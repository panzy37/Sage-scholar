"""Real PDF retrieval ablation; no answer model or gold-derived inputs."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

from evals.metrics import retrieval_metrics
from evals.sciegqa import REVISION, REPO, digest, download, extract_pdfs, load, select, ranked_pages


def configure(args):
    # Explicit before importing application config (which loads .env).
    values = {
        'OBSERVABILITY_ENABLED': 'false', 'HF_HUB_OFFLINE': '1',
        'EMBEDDING_PROVIDER': 'local', 'EMBEDDING_LOCAL_ONLY': 'true',
        'EMBEDDING_LOCAL_MODEL': args.embedding, 'EMBEDDING_DEVICE': args.device,
        'RERANKER_MODEL': args.reranker, 'RERANKER_DEVICE': args.device,
        'RERANKER_LOCAL_ONLY': 'true', 'RERANKER_FALLBACK': 'raise',
        'VISUAL_EMBEDDING_MODEL': args.visual_model, 'VISUAL_EMBEDDING_DEVICE': args.device,
        'VISUAL_EMBEDDING_PROVIDER': args.visual_provider,
        'VISUAL_EMBEDDING_BASE_URL': args.visual_base_url,
        'VISUAL_EMBEDDING_REVISION': args.visual_revision,
        'VISUAL_EMBEDDING_LOCAL_ONLY': 'true', 'PDF_VISUAL_ENABLED': 'true',
        'PDF_VISUAL_PARSER': 'page', 'PDF_VISUAL_MAX_SIDE': str(args.max_side),
    }
    os.environ.update(values)
    return values


def model_versions(models):
    """Read local cache metadata only; never query the Hub or load weights."""
    from huggingface_hub import try_to_load_from_cache
    from importlib.metadata import version, PackageNotFoundError
    result = {'snapshots': {}, 'packages': {}}
    for model in models:
        path = str(Path(model) / 'config.json') if Path(model).is_dir() else try_to_load_from_cache(model, 'config.json')
        if isinstance(path, str) and Path(path).is_file():
            parts = Path(path).parts
            result['snapshots'][model] = {'config_sha256': digest(path),
                'revision': parts[parts.index('snapshots') + 1] if 'snapshots' in parts else 'local-directory'}
        else:
            result['snapshots'][model] = {'revision': 'unavailable'}
    for package in ('transformers', 'sentence-transformers', 'faiss-cpu', 'PyMuPDF'):
        try:
            result['packages'][package] = version(package)
        except PackageNotFoundError:
            result['packages'][package] = 'unavailable'
    return result


def retrieve_rankings(adapter, query, policy, k, modes, unavailable=()):
    """Run production components once; caller scores only afterwards against gold."""
    from sage_scholar.visual.fusion import fuse
    permitted = adapter.permitted_documents(policy)
    result, rankings, elapsed = {}, {}, {}
    for channel in ('text', 'visual'):
        if channel not in modes and 'fused' not in modes:
            continue
        t = time.perf_counter()
        try:
            if channel == 'text':
                ranking, warnings = adapter.retrieve_with_warnings(query, policy, k, visual_mode='off')
            else:
                if any(d.document_id in unavailable for d in permitted):
                    raise RuntimeError('Visual index unavailable')
                ranking, warnings = adapter.visual.search(query, permitted, adapter.visual.config.max_images)
            if warnings:
                raise RuntimeError('; '.join(warnings))
            rankings[channel] = ranking
            elapsed[channel] = time.perf_counter() - t
        except Exception as exc:
            result[channel] = {'error': f'{type(exc).__name__}: {exc}', 'seconds': time.perf_counter() - t}
    if 'fused' in modes:
        if all(channel in rankings for channel in ('text', 'visual')):
            t = time.perf_counter()
            rankings['fused'] = fuse(rankings['text'], rankings['visual'], k + adapter.visual.config.max_images)
            elapsed['fused'] = sum(elapsed.values()) + time.perf_counter() - t
        else:
            result['fused'] = {'error': 'Component retrieval failed; not counted as a fused result'}
    return result, rankings, elapsed


def summarize(rows, modes, k):
    result = {}
    for mode in modes:
        observations = [r['results'][mode] for r in rows]
        valid = [r for r in observations if 'metrics' in r]
        metrics = {key: statistics.mean(r['metrics'][key] for r in valid)
                   for key in (valid[0]['metrics'] if valid else [])}
        result[mode] = {'status': ('evaluated' if len(valid) == len(observations) else 'partial') if valid else 'not_evaluated',
                        'samples': len(valid), 'errors': len(observations) - len(valid),
                        'metrics': metrics,
                        'metrics_including_execution_failures': {
                            key: value * len(valid) / len(observations) for key, value in metrics.items()},
                        'mean_latency_seconds': statistics.mean(r['seconds'] for r in valid) if valid else None}
    return result


def save_report(report, output):
    from sage_scholar.infrastructure.json_store import write_json
    write_json(output / 'report.json', report)
    lines = ['# SciEGQA RAG retrieval evaluation', '',
        f"Mode: **live models**; scope: **{report['scope']}**; "
        f"papers: {len(report['documents'])}; questions: {len(report['samples'])} / {report['dataset_questions']}.", '',
        'This is a page-retrieval adaptation, not the official answer/region-grounding score. '
        'English original questions; no translated or generated gold. No answer LLM/API calls.', '',
        'Metrics in the table use successful queries only; errors are listed separately. '
        'JSON also includes metrics_including_execution_failures (failed queries contribute zero).', '',
        f"Seed: {report['seed']}; page-render max side: {report['max_side']}; "
        f"visual result limit: {report['visual_limit']}; K: {report['k']}.", '',
        '| Mode | Evaluated | Errors | Recall@K | MRR@K | nDCG@K | Mean seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    if report.get('fatal_error'):
        lines += ['', 'Execution failed: ' + report['fatal_error'], '']
    for mode, value in report['summary'].items():
        m = value['metrics']
        vals = [m.get(f"recall@{report['k']}"), m.get('mrr'), m.get('ndcg'), value['mean_latency_seconds']]
        lines.append(f"| {mode} | {value['samples']} | {value['errors']} | " +
                     ' | '.join(f'{v:.4f}' if v is not None else 'not evaluated' for v in vals) + ' |')
    lines += ['', '## Protocol and limitations', '',
        '- Candidate pages include every page of each selected PDF; gold is used only after retrieval.',
        '- SourcePolicy restricts retrieval to the given document (or explicitly selected corpus mode).',
        '- Text: production FTS5/FAISS/RRF/BGE + section expansion. Visual: production Qwen image retrieval.',
        '- Fusion: production rank fusion of those same lists; component calls are reused to avoid duplicate inference.',
        '- Page ranking deduplicates Evidence start pages; it does not expand section spans to claim extra hits.',
        '- Fusion latency is the sum of measured text and visual retrieval plus fusion; timings include cold loads.',
        '- Visual limit remains the configured application limit, even if K is larger. No missing ranks are filled.',
        '- Model/index failures are reported separately, never scored as a successful text-only visual run.',
        '- Dataset coverage: ' + ('full' if len(report['samples']) == report['dataset_questions']
            and len(report['documents']) == report.get('dataset_papers') else 'partial/sample')
            + '. This measures page retrieval, not official answer correctness / bbox IoU.', '',
        '## Documents', '']
    lines += [f"- {name}: {doc['pages']} pages, `{doc['sha256']}`, parser `{doc['parser']}`"
              for name, doc in report['documents'].items()]
    lines += [f"- FAILED {name}: {error}" for name, error in report.get('document_errors', {}).items()]
    lines += ['', '## Models and environment', '', '```json',
              json.dumps(report['models'], indent=2, ensure_ascii=False), '```', '',
              'See report.json for individual questions, predictions, gold pages, errors, hashes and code version.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def run(args):
    if args.k < 1 or args.papers < 0 or args.questions < 0 or args.threads < 1:
        raise ValueError('Invalid K or sample limits')
    root, output = Path(args.dataset_root).resolve(), Path(args.output).resolve()
    if args.download:
        download(root)
    if not (root / 'PDF').exists():
        extract_pdfs(root / 'PDF.tar', root)
    try:
        rows, pdfs = load(root)
    except ValueError as exc:
        if not str(exc).startswith('Missing PDFs:') or not (root / 'PDF.tar').is_file():
            raise
        extract_pdfs(root / 'PDF.tar', root)
        rows, pdfs = load(root)
    chosen, names = select(rows, args.papers, args.seed, args.questions)
    if not chosen:
        raise ValueError('No selected questions')
    env = configure(args)
    from sage_scholar.config import Config
    from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
    from sage_scholar.domain.execution import SourcePolicy
    from sage_scholar.research.runtime import ResearchRuntimeProvider
    from sage_scholar.infrastructure.json_store import write_json, file_lock
    from evals.runners.runner import code_version
    import pymupdf
    import torch
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=True)
    # Guard isolated evaluation storage. No application data-dir argument is accepted.
    versions = model_versions((args.embedding, args.reranker) +
                              ((args.visual_model,) if args.visual_provider == 'local' else ()))
    settings = {**env, 'revision': REVISION, 'annotations': digest(root / 'SciEGQA_Bench.jsonl')}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    data = output / 'indexes' / key
    data.mkdir(parents=True, exist_ok=True)
    config = Config(data_dir=str(data))
    adapter = LegacyRetrievalAdapter(config)
    from sage_scholar.visual.embedding import VisualEmbedding
    class ProgressEmbedding(VisualEmbedding):
        def encode(self, inputs):
            if len(inputs) <= 1 or not all(isinstance(value, dict) for value in inputs):
                return super().encode(inputs)
            vectors = []
            for number, value in enumerate(inputs, 1):
                tick = time.perf_counter()
                vectors.extend(super().encode([value]))
                print(f'  Image {number}/{len(inputs)} encoded in {time.perf_counter()-tick:.1f}s', flush=True)
            return vectors
    adapter.visual.encoder = ProgressEmbedding(adapter.visual.config)
    def no_llm():
        raise RuntimeError('Answer LLM calls are forbidden in retrieval evaluation')
    provider = ResearchRuntimeProvider(config, no_llm, adapter)
    report = {'dataset': REPO, 'revision': REVISION, 'annotation_sha256': settings['annotations'],
        'dataset_questions': len(rows), 'dataset_papers': len(pdfs), 'mode': 'live',
        'created_at': datetime.now(timezone.utc).isoformat(), 'scope': args.scope, 'seed': args.seed,
        'k': args.k, 'max_side': args.max_side, 'visual_limit': adapter.visual.config.max_images,
        'code': code_version(), 'models': {'configuration': env, 'python': platform.python_version(),
            'torch': torch.__version__, 'device': args.device, 'threads': args.threads,
            'reranker': asdict(adapter.pipeline.reranker.config),
            'local_versions': versions, 'text_embedding': adapter.pipeline.embedding.identity,
            'visual_embedding': adapter.visual.encoder.identity, 'visual_config': asdict(adapter.visual.config)},
        'documents': {}, 'document_errors': {}, 'samples': [], 'summary': {}}
    modes = args.modes
    docs, visual_errors = {}, {}
    started = time.perf_counter()
    try:
        with file_lock(output / '.sciegqa-eval'):
            for name in names:
                with pymupdf.open(pdfs[name]) as pdf:
                    pages = len(pdf)
                if any(p > pages for r in chosen if r['doc_name'] == name for p in r['evidence_page']):
                    raise ValueError(f'Gold page outside PDF: {name}')
                identity = f'sciegqa:{REVISION}:{name}:{digest(pdfs[name])}'
                doc = next((d for d in adapter.documents.list() if d.source_identifier == identity), None)
                t = time.perf_counter()
                if not doc or not doc.active_version:
                    print(f'Importing {name}: all {pages} pages', flush=True)
                    manager = provider.get_library_manager('pdf')
                    # Text and visual indexes are prepared separately to expose failures and timings.
                    manager.visual = None
                    try:
                        # A previous failed import leaves a catalog entry without a version.
                        # Retry that entry rather than treating overwrite=False's skip as success.
                        result = manager.ingest(str(pdfs[name]), custom_title=name, overwrite=doc is not None,
                                                source_identifier=identity)
                        doc = adapter.documents.get(result.document_id)
                        if not doc or not doc.active_version or not doc.converted_path:
                            raise ValueError('Import did not produce a committed document version')
                    except Exception as exc:
                        from sage_scholar.runtime.journal import WorkflowInterrupted
                        if isinstance(exc, WorkflowInterrupted):
                            raise
                        report['document_errors'][name] = f'{type(exc).__name__}: {exc}'
                        print(f'Import failed {name}: {report["document_errors"][name]}', flush=True)
                        write_json(output / 'progress.json', report)
                        continue
                docs[name] = doc
                info = {'pages': pages, 'sha256': digest(pdfs[name]), 'document_id': doc.document_id,
                        'version': doc.active_version, 'parser': doc.parser_version,
                        'text_prepare_seconds': time.perf_counter() - t}
                if any(mode != 'text' for mode in modes):
                    manifest = adapter.visual._manifest(doc)
                    compatible = (manifest and manifest.status == 'ready'
                        and {k: v for k, v in manifest.embedding_identity.items() if k != 'dimension'}
                        == adapter.visual.encoder.identity and manifest.page_count == pages)
                    if not compatible:
                        print(f'Encoding {name}: {pages} page images ({args.visual_provider} model)', flush=True)
                        t = time.perf_counter()
                        status = adapter.visual.rebuild(doc.document_id)
                        info['visual_prepare_seconds'] = time.perf_counter() - t
                        if status['status'] != 'ready':
                            visual_errors[name] = status.get('error_type', 'visual_index_failed')
                    info['visual_status'] = 'failed' if name in visual_errors else 'ready'
                report['documents'][name] = info
                write_json(output / 'progress.json', report)
            mapping = {doc.document_id: name for name, doc in docs.items()}
            for index, row in enumerate(chosen, 1):
                print(f'Retrieving {index}/{len(chosen)} {row["id"]}', flush=True)
                if row['doc_name'] not in docs or (args.scope == 'corpus' and report['document_errors']):
                    error = report['document_errors'].get(row['doc_name'], 'Corpus incomplete after document import failure')
                    result = {mode: {'error': 'Document import failed: ' + error} for mode in modes}
                    rankings, elapsed = {}, {}
                else:
                    permitted = [docs[row['doc_name']]] if args.scope == 'document' else list(docs.values())
                    policy = SourcePolicy(source_scope='selected_documents',
                        document_ids=tuple(d.document_id for d in permitted), allow_web=False)
                    result, rankings, elapsed = retrieve_rankings(adapter, row['query'], policy, args.k,
                        modes, {docs[name].document_id for name in visual_errors})
                gold = [f'{row["doc_name"]}:{p}' for p in row['evidence_page']]
                for mode in modes:
                    if mode in rankings:
                        predicted = ranked_pages(rankings[mode], mapping)
                        result[mode] = {'retrieved_pages': predicted, 'seconds': elapsed[mode],
                            'evidence_ids': [e.evidence_id for e in rankings[mode]],
                            'metrics': retrieval_metrics(predicted, gold, args.k)}
                report['samples'].append({'id': row['id'], 'query': row['query'],
                    'doc_name': row['doc_name'], 'category': row['category'], 'subimg_type': row.get('subimg_type'),
                    'gold_pages': gold, 'gold_answer': row['answer'],
                    'results': {mode: result[mode] for mode in modes}})
                report['summary'] = summarize(report['samples'], modes, args.k)
                report['elapsed_seconds'] = time.perf_counter() - started
                report['models']['reranker_status'] = adapter.pipeline.reranker.last_status
                save_report(report, output)
    except Exception as exc:
        report['fatal_error'] = f'{type(exc).__name__}: {exc}'
        report['elapsed_seconds'] = time.perf_counter() - started
        save_report(report, output)
        raise
    finally:
        provider.close()
    print(json.dumps(report['summary'], indent=2), flush=True)
    return report
