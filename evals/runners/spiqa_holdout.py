"""Independent, preselected SPIQA test-A PDF page-retrieval validation.

SPIQA labels figure filenames, not physical PDF pages. Map only a unique exact
normalized caption prefix in the version-pinned PDF; abstain on ambiguous matches.
The protocol is frozen before downloading/reading retrieval results.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import random
import re
import unicodedata

from evals.sciegqa import digest
from sage_scholar.infrastructure.json_store import write_json

REVISION = '1774b71511f029b82089a069d75328f25fbf0705'


def normalize(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', value).lower())


def locate(caption, texts):
    # Exact caption-prefix match, no QA/gold-answer matching and no ranking score.
    normalized = normalize(caption)
    prefix = normalized[:100]
    if len(prefix) < 20:
        return []
    return [i + 1 for i, text in enumerate(texts) if prefix in normalize(text)]


def run(args):
    from evals.runners.multimodal_final import retrieval
    from sage_scholar.config import Config
    from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
    from sage_scholar.research.runtime import ResearchRuntimeProvider
    from evals.runners.runner import code_version
    import httpx
    import pymupdf
    import torch
    baseline = json.loads(Path(args.baseline).read_text())
    os.environ.update(baseline['models']['configuration'])
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = Path(args.metadata)
    corpus = json.loads(metadata.read_text())
    excluded = {r['doc_name'] for r in baseline['samples']}
    candidates = sorted(name for name in corpus if re.sub(r'v\d+$','',name) not in excluded)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:args.papers]
    plan = []
    for name in selected:
        indices = list(range(len(corpus[name]['qa'])))
        rng.shuffle(indices)
        plan.extend({'paper':name,'index':i} for i in indices[:args.questions])
    protocol = {'dataset':'google/spiqa', 'revision': REVISION, 'split':'test-A',
        'role':'independent_validation', 'metadata_sha256':digest(metadata), 'selection':plan,
        'excluded_sciegqa_papers':sorted(excluded), 'seed':args.seed,
        'mapping_rule':'unique exact normalized caption prefix (first 100 alphanumeric characters; minimum 20)',
        'unmapped':'excluded from labeled metrics, counted in coverage; never replaced',
        'code_before_results':code_version(), 'candidate_scope':'given_document_all_pages'}
    frozen = output / 'protocol.json'
    if frozen.exists():
        previous = json.loads(frozen.read_text())
        if previous['selection'] != plan or previous['metadata_sha256'] != digest(metadata):
            raise ValueError('SPIQA frozen selection mismatch')
    else:
        write_json(frozen, protocol)
    pdf_dir = output / 'pdfs'
    pdf_dir.mkdir(exist_ok=True)
    def download(name):
        path = pdf_dir / (name + '.pdf')
        try:
            if not path.exists():
                with httpx.Client(timeout=90, follow_redirects=True) as client:
                    response = client.get(f'https://arxiv.org/pdf/{name}')
                    response.raise_for_status()
                if not response.content.startswith(b'%PDF'):
                    raise ValueError('Not a PDF')
                temp = path.with_suffix('.part')
                temp.write_bytes(response.content)
                temp.replace(path)
            with pymupdf.open(path) as pdf:
                return name, [p.get_text() for p in pdf], None
        except Exception as exc:
            return name, [], str(exc)
    downloaded = list(concurrent.futures.ThreadPoolExecutor(max_workers=3).map(download, selected))
    papers = {name:(texts,error) for name,texts,error in downloaded}
    rows, alignment = [], []
    for choice in plan:
        name, index = choice['paper'], choice['index']
        item = corpus[name]['qa'][index]
        reference = item['reference']
        caption = corpus[name]['all_figures'][reference]['caption']
        texts, error = papers[name]
        pages = locate(caption, texts)
        identity = f'spiqa-{name}-{index:03}'
        entry = {'id':identity, 'reference':reference, 'caption':caption, 'pages':pages,
            'pdf_sha256':digest(pdf_dir / (name+'.pdf')) if texts else None,
            'status':'mapped' if len(pages)==1 else 'unresolved', 'download_error':error}
        alignment.append(entry)
        if len(pages) == 1:
            kind = corpus[name]['all_figures'][reference]['content_type']
            rows.append({'id':identity,'doc_name':name,'query':item['question'],'answer':item['answer'],
                'evidence_page':pages,'subimg_type':[['table' if kind=='table' else 'image']], 'category':'cs'})
    write_json(output / 'page-alignment.json', alignment)
    write_json(output / 'dataset.json', rows)
    config = Config(data_dir=str(output / 'indexes'))
    adapter = LegacyRetrievalAdapter(config)
    provider = ResearchRuntimeProvider(config, lambda:None, adapter)
    try:
        for name in sorted({r['doc_name'] for r in rows}):
            doc = next((d for d in adapter.documents.list() if d.title == name and d.active_version), None)
            if doc is None:
                print('SPIQA import',name,flush=True)
                manager = provider.get_library_manager('pdf')
                manager.visual = adapter.visual
                manager.ingest(str(pdf_dir/(name+'.pdf')),custom_title=name,
                    source_identifier=f'spiqa:{REVISION}:{name}:{digest(pdf_dir/(name+".pdf"))}')
    finally:
        provider.close()
    # No existing predictions: all component retrieval runs live on unseen papers.
    empty = {'samples':[{'id':r['id'],'results':{'text':{},'visual':{}}} for r in rows]}
    retrieval(args, empty, adapter, rows, output)
    report_path = output/'retrieval-summary.json'
    report = json.loads(report_path.read_text())
    report.update(dataset_role='independent_validation', selected=len(plan), mapped=len(rows),
                  selection_coverage=len(rows)/len(plan) if plan else 0)
    write_json(report_path,report)


def add_parser(commands):
    p=commands.add_parser('spiqa-holdout',help='Independent, preselected real-PDF validation; no paid LLM calls')
    p.add_argument('--baseline',default='eval-results/sciegqa/full/report.json')
    p.add_argument('--metadata',default='eval-results/spiqa/metadata.json')
    p.add_argument('--output',default='eval-results/spiqa/holdout')
    p.add_argument('--papers',type=int,default=20)
    p.add_argument('--questions',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--retry-failed',action='store_true')
