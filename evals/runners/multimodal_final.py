"""Resumable SciEGQA development diagnostics, reusing actual retrieval and QA.

The frozen baseline candidates are reused only when their document version still
matches. This is explicitly a development evaluation, never an unseen test set.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

from evals.sciegqa import load, digest, ranked_pages, REVISION
from evals.metrics import retrieval_metrics
from evals.runners.sciegqa import summarize


def group(row):
    types = {x for values in row['subimg_type'] for x in values}
    return next(iter(types)) if len(types) == 1 else 'mixed'


def stratify(rows, count, seed):
    if count % 4 or count < 4:
        raise ValueError('Sample size must be a positive multiple of four')
    rng = random.Random(seed)
    selected = []
    for kind in ('text', 'image', 'table', 'mixed'):
        docs = defaultdict(list)
        for row in rows:
            if group(row) == kind:
                docs[row['doc_name']].append(row)
        names = sorted(docs)
        rng.shuffle(names)
        for bucket in docs.values():
            rng.shuffle(bucket)
        chosen = []
        while len(chosen) < count // 4:
            added = False
            for name in names:
                if docs[name] and len(chosen) < count // 4:
                    chosen.append(docs[name].pop())
                    added = True
            if not added:
                raise ValueError(f'Insufficient {kind} questions')
        selected.extend(chosen)
    return selected


def initialize(args):
    baseline = json.loads(Path(args.baseline).read_text())
    os.environ.update(baseline['models']['configuration'])
    os.environ.update(OBSERVABILITY_ENABLED='false', TOKENIZERS_PARALLELISM='false')
    from sage_scholar.config import Config
    from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
    import torch
    torch.set_num_threads(4)
    adapter = LegacyRetrievalAdapter(Config(data_dir=str(Path(args.index).resolve())))
    rows, pdfs = load(args.dataset_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return baseline, adapter, rows, pdfs, output


def freeze(args, rows, pdfs, output, baseline):
    from sage_scholar.infrastructure.json_store import write_json
    from evals.runners.runner import code_version
    chosen = stratify(rows, args.samples, args.seed)
    protocol = {'dataset_role': 'development_only', 'dataset_revision': REVISION,
        'baseline_sha256': digest(args.baseline), 'seed': args.seed,
        'selected_ids': [r['id'] for r in chosen], 'groups': {r['id']: group(r) for r in chosen},
        'sample_count': args.samples, 'candidate_scope': 'given_document',
        'k': 3, 'arms': ['text', 'multimodal', 'gold'], 'answer_model': args.model,
        'judge_model': args.judge_model, 'fusion': 'page',
        'models': baseline['models'], 'code': code_version(),
        'pdf_sha256': {name: digest(path) for name, path in pdfs.items()}}
    path = output / 'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for field in ('selected_ids','answer_model','judge_model','pdf_sha256','baseline_sha256','fusion'):
            if previous[field] != protocol[field]:
                raise ValueError(f'Frozen protocol mismatch: {field}; use a new output directory')
    else:
        write_json(path, protocol)
    # Keep the original selection immutable; append execution provenance rather
    # than overwriting the preselection timestamp/fingerprint on resume.
    executions = output / 'executions'
    executions.mkdir(exist_ok=True)
    write_json(executions / f'{args.phase}-{time.time_ns()}.json',
        {'phase': args.phase, 'code': code_version(), 'answer_model': args.model,
         'judge_model': args.judge_model, 'samples': args.samples,
         'baseline_sha256': protocol['baseline_sha256'], 'index': str(Path(args.index).resolve())})
    return chosen


def alignment(rows, pdfs, output):
    import pymupdf
    from sage_scholar.infrastructure.json_store import write_json
    documents = {}
    for name, path in pdfs.items():
        with pymupdf.open(path) as pdf:
            related = [r for r in rows if r['doc_name'] == name]
            bad = [r['id'] for r in related if any(p > len(pdf) or p < 1 for p in r['evidence_page'])]
            documents[name] = {'pdf_sha256': digest(path), 'pages': len(pdf), 'out_of_range': bad,
                'gold_pages': sorted({p for r in related for p in r['evidence_page']}),
                'validation': 'official PDF archive identity + one-based physical page bounds; not bbox correctness'}
            if bad:
                raise ValueError(f'Gold/PDF mismatch: {name}: {bad}')
    write_json(output / 'page-alignment.json', documents)


def prepare(adapter, rows, pdfs, output):
    from sage_scholar.library.library_manager import LibraryManager
    from sage_scholar.research.runtime import ResearchRuntimeProvider
    from sage_scholar.config import Config
    from sage_scholar.infrastructure.json_store import write_json
    provider = ResearchRuntimeProvider(Config(data_dir=str(adapter.visual.data_dir)), lambda: None, adapter)
    results = {}
    try:
        for name, path in pdfs.items():
            identity = f'sciegqa:{REVISION}:{name}:{digest(path)}'
            doc = next((d for d in adapter.documents.list() if d.source_identifier == identity), None)
            if not doc or not doc.active_version:
                print('Repairing import', name, flush=True)
                manager = provider.get_library_manager('pdf')
                manager.visual = adapter.visual
                result = manager.ingest(str(path), custom_title=name, overwrite=True, source_identifier=identity)
                doc = adapter.documents.get(result.document_id)
            manifest = adapter.visual._manifest(doc)
            if not manifest or manifest.status != 'ready':
                print('Repairing visual index', name, flush=True)
                status = adapter.visual.rebuild(doc.document_id)
                if status['status'] != 'ready':
                    raise RuntimeError(f'{name}: {status}')
            results[name] = {'document_id': doc.document_id, 'version': doc.active_version,
                'sha256': doc.content_hash, 'text': doc.metadata.get('text_index', {'status': 'ready'}),
                'visual': 'ready'}
            write_json(output / 'imports.json', results)
    finally:
        provider.close()


def retrieval(args, baseline, adapter, rows, output):
    from sage_scholar.domain.execution import SourcePolicy
    from sage_scholar.visual.fusion import fuse
    from sage_scholar.infrastructure.json_store import write_json
    records = {r['id']: r for r in baseline['samples']}
    docs = {d.title: d for d in adapter.documents.list() if d.active_version}
    names = {d.document_id: name for name, d in docs.items()}
    modes = ('text', 'visual_assisted', 'visual_pure', 'evidence_rrf', 'page_rrf', 'page_rrf_pure')
    directory = output / 'retrieval'
    directory.mkdir(exist_ok=True)
    completed = []
    for number, row in enumerate(rows, 1):
        path = directory / (row['id'] + '.json')
        previous = json.loads(path.read_text()) if path.exists() else None
        if path.exists():
            if not getattr(args, 'retry_failed', False) or not any('error' in r for r in previous['results'].values()):
                completed.append(previous)
                continue
            archive = directory / 'attempts'
            archive.mkdir(exist_ok=True)
            path.replace(archive / f'{row["id"]}-{time.time_ns()}.json')
        tick = time.perf_counter()
        doc = docs.get(row['doc_name'])
        result = {'id': row['id'], 'doc_name': row['doc_name'], 'group': group(row),
                  'gold_pages': [f'{row["doc_name"]}:{p}' for p in row['evidence_page']], 'results': {}}
        try:
            if doc is None:
                raise ValueError('No committed document')
            policy = SourcePolicy(source_scope='selected_documents', document_ids=(doc.document_id,))
            rankings, times = {}, {}
            for channel, old in (('text','text'), ('visual_assisted','visual'), ('visual_pure',None)):
                start = time.perf_counter()
                try:
                    prior = previous['results'].get(channel, {}) if previous else {}
                    stored = prior if 'metrics' in prior else records[row['id']]['results'].get(old, {})
                    candidates = [adapter.evidence_store.get(e) for e in stored.get('evidence_ids', [])]
                    valid = bool(candidates) and all(e and e.document_id == doc.document_id and e.source_version == doc.active_version for e in candidates)
                    if valid:
                        rankings[channel], times[channel] = candidates, stored['seconds']
                    else:
                        candidates, warnings = (adapter.retrieve_with_warnings(row['query'], policy, 3, visual_mode='off')
                            if channel == 'text' else adapter.visual.search(row['query'], [doc], 3,
                                locator_assistance=channel == 'visual_assisted'))
                        if warnings:
                            raise RuntimeError(';'.join(warnings))
                        rankings[channel], times[channel] = candidates, time.perf_counter() - start
                    result['results'][channel] = {'candidate_origin': ('previous_attempt' if stored is prior else 'frozen_baseline') if valid else 'live'}
                except Exception as exc:
                    result['results'][channel] = {'error': f'{type(exc).__name__}: {exc}', 'seconds': time.perf_counter()-start}
            for mode, strategy, visual in (('evidence_rrf','evidence','visual_assisted'),
                ('page_rrf','page','visual_assisted'), ('page_rrf_pure','page','visual_pure')):
                if 'text' in rankings and visual in rankings:
                    rankings[mode] = fuse(rankings['text'], rankings[visual], 6, strategy=strategy, max_pages=3)
                    times[mode] = times['text'] + times[visual]
                else:
                    result['results'][mode] = {'error': 'Required component retrieval failed'}
            for mode, ranking in rankings.items():
                pages = ranked_pages(ranking, names)
                result['results'].setdefault(mode, {}).update(retrieved_pages=pages,
                    evidence_ids=[e.evidence_id for e in ranking], seconds=times[mode],
                    metrics=retrieval_metrics(pages, result['gold_pages'], 3))
        except Exception as exc:
            result['results'] = {mode: {'error': f'{type(exc).__name__}: {exc}'} for mode in modes}
        result['wall_seconds'] = time.perf_counter() - tick
        write_json(path, result)
        completed.append(result)
        if number % 20 == 0:
            print(f'Retrieval {number}/{len(rows)}', flush=True)
    summary = summarize(completed, modes, 3)
    regressions = [r for r in completed if r['id'] in records and
        records[r['id']]['results'].get('fused',{}).get('metrics',{}).get('recall@3', 0) <
        records[r['id']]['results']['text'].get('metrics',{}).get('recall@3', 0)]
    write_json(output / 'retrieval-summary.json', {'dataset_role': 'development_only', 'summary': summary,
        'baseline_regressions': regressions, 'samples': completed})
    print(json.dumps(summary, indent=2), flush=True)


def add_parser(commands):
    p = commands.add_parser('multimodal-final', help='SciEGQA development ablations and real QA; explicit paid opt-in')
    p.add_argument('--phase', choices=['prepare','retrieval','answers','judge','report'], required=True)
    p.add_argument('--baseline', default='eval-results/sciegqa/full/report.json')
    p.add_argument('--index', default='eval-results/sciegqa/full/indexes/62854fbcf1d5b147')
    p.add_argument('--dataset-root', default='eval-results/sciegqa/downloads')
    p.add_argument('--output', default='eval-results/sciegqa/final')
    p.add_argument('--samples', type=int, default=240)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--model', default='deepseek-flash')
    p.add_argument('--judge-model', default='deepseek-v4-pro')
    p.add_argument('--allow-paid', action='store_true')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='Smoke only: process first N selected questions; selection stays frozen')
    p.add_argument('--wait-for-inputs', action='store_true', help='Wait up to one hour for concurrently running retrieval/answers')
    p.add_argument('--retry-failed', action='store_true', help='Retry only recorded failures; preserve previous attempts')


def run(args):
    if args.phase in {'answers','judge'} and not args.allow_paid:
        raise ValueError('Real QA/judge requires --allow-paid')
    baseline, adapter, rows, pdfs, output = initialize(args)
    chosen = freeze(args, rows, pdfs, output, baseline)
    if args.phase == 'prepare':
        alignment(rows, pdfs, output)
        prepare(adapter, rows, pdfs, output)
    elif args.phase == 'retrieval':
        retrieval(args, baseline, adapter, rows, output)
    else:
        from evals.runners.multimodal_answers import execute, report
        if args.phase == 'report':
            report(output, chosen)
        else:
            import asyncio
            asyncio.run(execute(args, adapter, chosen[:args.limit] if args.limit else chosen, output))
