"""Adapter/metric contracts; these tests are not real model accuracy measurements."""
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from evals.sciegqa import extract_pdfs, load, ranked_pages, select
from evals.metrics import retrieval_metrics
from evals.runners.sciegqa import summarize, save_report


def sample(name='2401.12345', page=2):
    return {'doc_name': name, 'query': 'What does the plot show?', 'answer': 'a trend',
            'evidence_page': [page], 'category': 'cs', 'subimg_type': [['image']]}


def dataset(tmp_path, rows):
    (tmp_path / 'SciEGQA_Bench.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    folder = tmp_path / 'PDF' / 'cs'
    folder.mkdir(parents=True)
    for r in rows:
        (folder / f'{r["doc_name"]}_title.pdf').write_bytes(b'pdf fixture')
    return tmp_path


def test_real_format_one_based_pages_and_stable_line_ids(tmp_path):
    rows, pdfs = load(dataset(tmp_path, [sample()]))
    assert rows[0]['evidence_page'] == [2]  # no mistaken +1
    assert rows[0]['id'] == 'sciegqa-0001'
    assert pdfs['2401.12345'].name == '2401.12345_title.pdf'


@pytest.mark.parametrize('page', [0, -1, '1', True])
def test_invalid_gold_pages_fail(tmp_path, page):
    with pytest.raises(ValueError, match='1-based'):
        load(dataset(tmp_path, [sample(page=page)]))


def test_missing_or_ambiguous_pdfs_fail(tmp_path):
    root = dataset(tmp_path, [sample()])
    (root / 'PDF' / 'cs' / '2401.12345_other.pdf').write_bytes(b'other')
    with pytest.raises(ValueError, match='Ambiguous'):
        load(root)
    for p in (root / 'PDF').rglob('*.pdf'):
        p.unlink()
    with pytest.raises(ValueError, match='Missing PDFs'):
        load(root)


def test_sampling_is_seeded_not_conditioned_on_gold():
    rows = [sample(f'2401.{i:05}', page=i+1) for i in range(10)]
    first, names = select(rows, 3, 42)
    altered = [{**r, 'answer': 'changed gold', 'evidence_page': [99]} for r in rows]
    second, other = select(altered, 3, 42)
    assert names == other and len(names) == 3
    assert [r['query'] for r in first] == [r['query'] for r in second]
    assert len(select(rows, 0, 42)[0]) == 10


def test_page_projection_deduplicates_and_does_not_invent_section_pages():
    def ev(doc, page):
        return SimpleNamespace(document_id=doc, page=page, location={'page_end': 9})
    pages = ranked_pages([ev('a', 2), ev('a', 2), ev('b', 2), ev('a', None)], {'a': 'paper-a', 'b': 'paper-b'})
    assert pages == ['paper-a:2', 'paper-b:2']
    scores = retrieval_metrics(pages, ['paper-a:3'], 3)
    assert scores['recall@3'] == 0


def test_multi_page_gold_scoring():
    scores = retrieval_metrics(['a:1', 'a:1', 'a:3'], ['a:2', 'a:3'], 3)
    assert scores['recall@3'] == .5
    assert scores['mrr'] == .5


def archive(tmp_path, name, *, symlink=False):
    path = tmp_path / 'archive.tar'
    with tarfile.open(path, 'w') as file:
        item = tarfile.TarInfo(name)
        if symlink:
            item.type, item.linkname = tarfile.SYMTYPE, '/etc/passwd'
            file.addfile(item)
        else:
            data = b'pdf fixture'
            item.size = len(data)
            file.addfile(item, io.BytesIO(data))
    return path


@pytest.mark.parametrize('name,symlink', [('../bad.pdf', False), ('/tmp/bad.pdf', False),
    ('PDF/../../bad.pdf', False), ('PDF/cs/link.pdf', True), ('PDF/cs/exec.py', False)])
def test_archive_rejects_unsafe_members(tmp_path, name, symlink):
    with pytest.raises(ValueError):
        extract_pdfs(archive(tmp_path, name, symlink=symlink), tmp_path / 'out')


def test_archive_regular_pdf(tmp_path):
    extract_pdfs(archive(tmp_path, 'PDF/cs/2401.12345_title.pdf'), tmp_path / 'out')
    assert (tmp_path / 'out/PDF/cs/2401.12345_title.pdf').read_bytes() == b'pdf fixture'


def test_failed_visual_is_not_scored_as_text_success():
    rows = [{'results': {'visual': {'error': 'missing encoder'}, 'text': {
        'metrics': {'recall@3': .5}, 'seconds': 1.2}}}]
    summary = summarize(rows, ['text', 'visual'], 3)
    assert summary['text']['samples'] == 1
    assert summary['visual']['status'] == 'not_evaluated'
    assert summary['visual']['errors'] == 1
    assert summary['visual']['metrics'] == {}


def test_import_failure_remains_in_aggregate_denominator():
    rows = [{'results': {'text': {'error': 'Document import failed'}}},
            {'results': {'text': {'metrics': {'recall@3': 1}, 'seconds': 1}}}]
    summary = summarize(rows, ['text'], 3)['text']
    assert summary['status'] == 'partial'
    assert summary['samples'] == 1 and summary['errors'] == 1
    assert summary['metrics']['recall@3'] == 1
    assert summary['metrics_including_execution_failures']['recall@3'] == .5


def test_production_fusion_reuses_calls_and_preserves_policy():
    from evals.runners.sciegqa import retrieve_rankings
    from sage_scholar.domain.execution import SourcePolicy
    calls = []
    doc = SimpleNamespace(document_id='allowed')
    text = SimpleNamespace(evidence_id='text', document_id='allowed', page=2)
    image = SimpleNamespace(evidence_id='image', document_id='allowed', page=3)
    policy = SourcePolicy(source_scope='selected_documents', document_ids=('allowed',), allow_web=False)
    def retrieve(query, supplied_policy, k, *, visual_mode):
        assert supplied_policy == policy and visual_mode == 'off'
        calls.append(('text', query))
        return [text], []
    def search(query, docs, k):
        assert docs == [doc] and k == 3
        calls.append(('visual', query))
        return [image], []
    adapter = SimpleNamespace(permitted_documents=lambda p: [doc] if p == policy else [],
        retrieve_with_warnings=retrieve,
        visual=SimpleNamespace(search=search, config=SimpleNamespace(max_images=3)))
    errors, ranking, elapsed = retrieve_rankings(adapter, 'original query', policy, 3, ['text','visual','fused'])
    assert not errors
    assert calls == [('text', 'original query'), ('visual', 'original query')]
    assert ranking['fused'] == [text, image]
    assert elapsed['fused'] >= elapsed['text'] + elapsed['visual']
    calls.clear()
    errors, ranking, _ = retrieve_rankings(adapter, 'q', policy, 3, ['text','visual','fused'], {'allowed'})
    assert 'visual' in errors and 'fused' in errors
    assert 'fused' not in ranking and len(calls) == 1


def test_visual_fallback_warning_fails_ablation():
    from evals.runners.sciegqa import retrieve_rankings
    adapter = SimpleNamespace(permitted_documents=lambda p: [], visual=SimpleNamespace(
        config=SimpleNamespace(max_images=3), search=lambda *args: ([], ['encoder unavailable'])))
    errors, rankings, _ = retrieve_rankings(adapter, 'q', object(), 3, ['visual'])
    assert 'visual' in errors and not rankings
