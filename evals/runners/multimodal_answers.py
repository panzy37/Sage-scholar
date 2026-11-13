"""Real QAService ablations with a frozen retrieval boundary and resumable outputs."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from sage_scholar.infrastructure.json_store import write_json
from .multimodal_final import group


def completed(path, retry_failed):
    if not path.exists():
        return False
    previous = json.loads(path.read_text())
    if not retry_failed or 'error' not in previous:
        return True
    # Never discard error records or silently regenerate a successful answer.
    archive = path.parent / 'attempts'
    archive.mkdir(exist_ok=True)
    path.replace(archive / f'{path.stem}-{time.time_ns()}.json')
    return False


def provider_halted(output, error=None):
    """Share a circuit breaker between concurrently running answer/judge phases."""
    path = output / 'provider-halted.json'
    status = getattr(error, 'status_code', None)
    if status in (401, 402, 403):
        write_json(path, {'status_code': status, 'reason': type(error).__name__, 'created_ns': time.time_ns(),
                          'action': 'Fix credentials/balance, then explicitly --retry-failed'})
    return path.exists()


def gold_evidence(adapter, doc, pages):
    # Gold answers/queries never enter this operation: only original physical pages.
    text = [e for e in adapter._sections(doc) if e.page in pages]
    manifest = adapter.visual._manifest(doc)
    images = [adapter.visual._evidence(doc, manifest, item) for item in manifest.elements
              if item.kind == 'page' and item.page in pages]
    return text + images


class SelectedEvidence:
    """Controlled retrieval intervention, not an alternative answer generator."""
    def __init__(self, adapter, evidence, visual):
        self.adapter, self.evidence = adapter, evidence
        self.visual = adapter.visual if visual else None

    def retrieve_with_warnings(self, query, policy, **kwargs):
        if any(not policy.permits_document(e.document_id) for e in self.evidence):
            raise ValueError('Evidence outside SourcePolicy')
        return self.evidence, []

    async def read_documents(self, *, policy):
        # Even overview questions stay within the specified experimental arm.
        return self.retrieve_with_warnings('', policy)[0]

    def permitted_documents(self, policy):
        # QAService overview supplementation must not bypass the frozen arm.
        return []


async def answer_one(args, adapter, row, arm, output):
    from sage_scholar.application.qa_service import QAService
    from sage_scholar.infrastructure.model_gateway import LegacyModelGateway
    from sage_scholar.config import Config
    from sage_scholar.context.manager import ContextManager, request_context
    from sage_scholar.domain.execution import ExecutionRequest
    from sage_scholar.base.llm_client import LLMClient
    directory = output / 'answers' / row['id']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (arm + '.json')
    if completed(path, args.retry_failed):
        return
    start = time.perf_counter()
    record = {'id': row['id'], 'arm': arm, 'group': group(row), 'query': row['query'],
              'model': args.model, 'calls': []}
    try:
        retrieval = json.loads((output / 'retrieval' / (row['id'] + '.json')).read_text())
        doc = next(d for d in adapter.documents.list() if d.title == row['doc_name'])
        if arm == 'gold':
            evidence = await asyncio.to_thread(gold_evidence, adapter, doc, row['evidence_page'])
        else:
            observed = retrieval['results']['text' if arm == 'text' else 'page_rrf']
            if 'error' in observed:
                raise RuntimeError(observed['error'])
            evidence = [adapter.evidence_store.get(e) for e in observed['evidence_ids']]
        if not evidence or any(e is None for e in evidence):
            raise ValueError('Missing original evidence')
        request = ExecutionRequest(query=row['query'], workflow='qa', source_scope='selected_documents',
            document_ids=(doc.document_id,), allow_web=False, execution_mode='inline',
            request_id=f'eval-{row["id"]}-{arm}', metadata={'visual_mode': 'off' if arm == 'text' else 'on'})
        context = ContextManager(str(output / 'context'), max_tokens=30000)
        class MeasuredClient(LLMClient):
            def invoke(self, *a, **kw):
                if provider_halted(output):
                    raise RuntimeError('Evaluation provider halted; no further API requests')
                try:
                    return super().invoke(*a, **kw)
                except Exception as exc:
                    provider_halted(output, exc)
                    raise
            def _track_usage(self, prompt_tokens, completion_tokens, elapsed, tag, msg, **kwargs):
                record['calls'].append({'tag': tag, 'input_tokens': prompt_tokens, 'output_tokens': completion_tokens,
                    'seconds': elapsed, 'finish_reason': getattr(msg, 'finish_reason', None), **kwargs})
                super()._track_usage(prompt_tokens, completion_tokens, elapsed, tag, msg, **kwargs)
        gateway = LegacyModelGateway(Config(data_dir=str(adapter.visual.data_dir)), context_manager=context)
        client = MeasuredClient(model=args.model, timeout=120, context_manager=context)
        client.client = client.client.with_options(max_retries=0)
        gateway._client = gateway._vision_client = client
        service = QAService(SelectedEvidence(adapter, evidence, arm != 'text'), gateway, context_manager=context)
        try:
            with request_context(request):
                result = await service.answer(request)
        finally:
            client.client.close()
        record.update(answer=result.answer, citations=result.citations, citation_issues=result.citation_issues,
            evidence=[e.model_dump(mode='json', exclude={'text'}) for e in result.evidence],
            reading_strategy=result.reading_strategy,
            visual_fallback='视觉模型不可用' in result.answer or '原图无法读取' in result.answer,
            gold_answer=row['answer'], gold_pages=row['evidence_page'], document_version=doc.active_version,
            pdf_sha256=doc.content_hash)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['seconds'] = time.perf_counter() - start
    # Unknown provider cost is not zero; store actual usage separately from estimates.
    record['cost_usd'] = None
    write_json(path, record)
    print('Answer', row['id'], arm, record.get('error', 'ok'), f'{record["seconds"]:.1f}s', flush=True)
    return record


async def judge_one(args, adapter, row, arm, output):
    from evals.judges.multimodal import judge_answer
    path = output / 'judgments' / row['id'] / (arm + '.json')
    answer_path = output / 'answers' / row['id'] / (arm + '.json')
    if not answer_path.exists():
        return
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('answer_sha256') != hashlib.sha256(answer_path.read_bytes()).hexdigest():
            # An answer may have been retried independently. Its previous judge
            # result is no longer valid, even if that judgment was successful.
            archive = path.parent / 'attempts'
            archive.mkdir(exist_ok=True)
            path.replace(archive / f'{path.stem}-stale-{time.time_ns()}.json')
    if completed(path, args.retry_failed):
        return
    answer = json.loads(answer_path.read_text())
    if 'error' in answer:
        return
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(judge_answer, adapter, row, answer, args.judge_model)
    except Exception as exc:
        provider_halted(output, exc)
        result = {'error': f'{type(exc).__name__}: {exc}'}
    result.update(id=row['id'], arm=arm, group=group(row), seconds=time.perf_counter()-start,
                  answer_sha256=hashlib.sha256(answer_path.read_bytes()).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    print('Judge', row['id'], arm, result.get('error', 'ok'), flush=True)
    return result


async def execute(args, adapter, rows, output):
    if not 1 <= args.concurrency <= 16:
        raise ValueError('concurrency must be between 1 and 16')
    semaphore = asyncio.Semaphore(args.concurrency)
    halt = output / 'provider-halted.json'
    if args.retry_failed and halt.exists():
        halt.replace(output / f'provider-halted-{time.time_ns()}.json')
    async def task(row, arm):
        dependency = (output / 'retrieval' / (row['id'] + '.json') if args.phase == 'answers'
                      else output / 'answers' / row['id'] / (arm + '.json'))
        deadline = time.monotonic() + 3600
        while args.wait_for_inputs and not dependency.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Upstream evaluation did not produce {dependency.name}')
            await asyncio.sleep(2)
        async with semaphore:
            if provider_halted(output):
                return
            await (answer_one if args.phase == 'answers' else judge_one)(args, adapter, row, arm, output)
    await asyncio.gather(*(task(row, arm) for row in rows for arm in ('text','multimodal','gold')))


def report(output, rows):
    answers, judgments = [], []
    selected = {r['id'] for r in rows}
    invalidated = []
    for folder, target in (('answers', answers), ('judgments', judgments)):
        for path in (output / folder).glob('*/*.json'):
            record = json.loads(path.read_text())
            if record['id'] not in selected:
                continue
            if folder == 'judgments':
                source = output / 'answers' / record['id'] / (record['arm'] + '.json')
                if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != record.get('answer_sha256'):
                    invalidated.append(str(path))
                    continue
            target.append(record)
    combined = []
    for kind in ('all','text','image','table','mixed'):
        for arm in ('text','multimodal','gold'):
            expected = sum(kind == 'all' or group(r) == kind for r in rows)
            valid = [r for r in answers if r['arm'] == arm and (kind == 'all' or r['group'] == kind)]
            scored = [r for r in judgments if r['arm'] == arm and (kind == 'all' or r['group'] == kind) and 'metrics' in r]
            item = {'group': kind, 'arm': arm, 'expected': expected, 'answered': sum('answer' in r for r in valid),
                    'failures': sum('error' in r for r in valid), 'judged': len(scored), 'metrics': {}}
            for metric in ('answer_correctness','faithfulness','citation_accuracy'):
                values = [r['metrics'][metric]['score'] for r in scored if r['metrics'][metric]['score'] is not None]
                item['metrics'][metric] = {'mean': statistics.mean(values) if values else None, 'n': len(values)}
            item['mean_seconds'] = statistics.mean(r['seconds'] for r in valid) if valid else None
            combined.append(item)
    # Partial API runs must not compare means over different question sets.
    scored_by_key = {(r['id'], r['arm']): r for r in judgments if 'metrics' in r}
    arms = ('text', 'multimodal', 'gold')
    paired_ids = {r['id'] for r in rows if all((r['id'], arm) in scored_by_key for arm in arms)}
    paired = []
    for kind in ('all', 'text', 'image', 'table', 'mixed'):
        ids = [r['id'] for r in rows if r['id'] in paired_ids and (kind == 'all' or group(r) == kind)]
        for arm in arms:
            metrics = {}
            for metric in ('answer_correctness', 'faithfulness', 'citation_accuracy'):
                # Abstentions can have null support scores. Use the SAME
                # assessable questions across all arms for a paired metric.
                usable = [i for i in ids if all(scored_by_key[i, a]['metrics'][metric]['score'] is not None for a in arms)]
                values = [scored_by_key[i, arm]['metrics'][metric]['score'] for i in usable]
                metrics[metric] = {'mean': statistics.mean(values) if values else None, 'n': len(values)}
            paired.append({'group': kind, 'arm': arm, 'paired_questions': len(ids), 'metrics': metrics})
    expected = len(rows) * len(arms)
    successful_answers = sum('answer' in r for r in answers)
    result = {'method': 'LLM judge, not human gold accuracy', 'dataset_role': 'development_only', 'groups': combined,
              'invalidated_judgments': invalidated, 'paired_groups': paired,
              'paired_question_ids': sorted(paired_ids),
              'status': 'complete' if len(scored_by_key) == expected and successful_answers == expected else 'partial',
              'expected_answers': expected, 'successful_answers': successful_answers,
              'expected_judgments': expected, 'successful_judgments': len(scored_by_key),
              'unscored_judgments': expected - len(scored_by_key)}
    write_json(output / 'answer-summary.json', result)
    lines = ['# Multimodal QA evaluation', '', 'Scores are model judgments, not independently human-verified accuracy.', '',
        f'Status: {result["status"]}; successful judgments {len(scored_by_key)}/{expected}.',
        'Use paired_groups in JSON for matched comparisons when scores are missing or null.', '',
        '| Group | Arm | Answered / selected | Judged | Correctness | Faithfulness | Citation accuracy |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in combined:
        metrics = [row['metrics'][m]['mean'] for m in ('answer_correctness','faithfulness','citation_accuracy')]
        lines.append(f'| {row["group"]} | {row["arm"]} | {row["answered"]}/{row["expected"]} | {row["judged"]} | ' +
                     ' | '.join('not evaluated' if v is None else f'{v:.4f}' for v in metrics) + ' |')
    (output / 'answer-summary.md').write_text('\n'.join(lines)+'\n')
