"""SciEGQA data adapter. Gold pages never enter the retrieval query or index."""
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tarfile
import urllib.request

REPO = 'Yuwh07/SciEGQA-Bench'
REVISION = '9bdca77be633cc3259d7e3739d35b38ab00bf5d9'
FILES = ('README.md', 'SciEGQA_Bench.jsonl', 'PDF.tar')


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def download(root):
    """Explicit download only; existing files are reused, incomplete files replaced."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = root / name
        if target.is_file():
            continue
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}'
        temporary = target.with_suffix(target.suffix + '.part')
        print(f'Downloading {name}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        temporary.replace(target)


def extract_pdfs(archive, root):
    """Accept only regular PDF members under PDF/, never links or path traversal."""
    root = Path(root).resolve()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        seen = set()
        for member in members:
            path = Path(member.name)
            target = (root / path).resolve()
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] != 'PDF' or not target.is_relative_to(root)
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f'Unsafe PDF archive member: {member.name}')
            if member.isfile() and (path.suffix.lower() != '.pdf' or member.name in seen):
                raise ValueError(f'Unexpected PDF archive member: {member.name}')
            seen.add(member.name)
        for member in members:
            if not member.isfile():
                continue
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.pdf.part')
            with bundle.extractfile(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)


def load(root):
    root = Path(root)
    rows = []
    for number, line in enumerate((root / 'SciEGQA_Bench.jsonl').read_text().splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row.get('query'), str) or not row['query'].strip():
            raise ValueError(f'Missing query at line {number}')
        if not re.fullmatch(r'\d{4}\.\d{4,5}', row.get('doc_name', '')):
            raise ValueError(f'Invalid document identity at line {number}')
        pages = row.get('evidence_page')
        if not pages or any(type(p) is not int or p < 1 for p in pages):
            raise ValueError(f'Gold pages must be 1-based at line {number}')
        row = {**row, 'id': f'sciegqa-{number:04d}'}
        rows.append(row)
    if not rows:
        raise ValueError('Empty SciEGQA annotations')
    pdfs = {}
    for path in sorted((root / 'PDF').rglob('*.pdf')):
        name = path.stem.split('_', 1)[0]
        if name in pdfs:
            raise ValueError(f'Ambiguous PDF identity: {name}')
        pdfs[name] = path
    missing = {r['doc_name'] for r in rows} - pdfs.keys()
    if missing:
        raise ValueError(f'Missing PDFs: {sorted(missing)}')
    return rows, pdfs


def select(rows, papers, seed, questions=0):
    if papers < 0 or questions < 0:
        raise ValueError('Sample limits cannot be negative; 0 means all')
    names = sorted({r['doc_name'] for r in rows})
    random.Random(seed).shuffle(names)
    names = names[:papers] if papers else names
    chosen = [r for r in rows if r['doc_name'] in names]
    if questions:
        random.Random(seed).shuffle(chosen)
        chosen = chosen[:questions]
    return chosen, [name for name in names if any(r['doc_name'] == name for r in chosen)]


def ranked_pages(evidence, names):
    """Start-page projection; never award hits for guessed section/page ranges."""
    return list(dict.fromkeys(f'{names[e.document_id]}:{e.page}' for e in evidence
                             if e.document_id in names and e.page is not None))
