from types import SimpleNamespace
import pytest

from test_seven_structured_rag import pipeline, paper
from test_ingestion_rollback import manager_for, paths, disk
from sage_scholar.rag.chunker import TextChunker
from sage_scholar.domain.evidence import EvidenceRef
from sage_scholar.visual.fusion import fuse
from evals.metrics import retrieval_metrics
from test_pdf_visual import library


@pytest.mark.parametrize('header', ['Name | Detail', 'Name | ' + 'long header ' * 100])
def test_oversized_cells_keep_all_text_and_column_locators(header):
    value = ' escaped \\| pipe and long value ' * 100
    source = SimpleNamespace(content=f'| {header} |\n| --- | --- |\n| row-key |{value}|', metadata={})
    parts = list(TextChunker(160, count_tokens=len)._table(source))
    cells = [(text, meta) for text, meta in parts if meta.get('table_part') == 'cell' and meta['column_index'] == 1]
    assert cells
    restored = ''.join(value[m['cell_char_start']:m['cell_char_end']] for _, m in cells)
    assert restored == value
    assert all(len(text) <= 160 for text, _ in parts)
    assert all(m['row_start'] == 0 and m['row_key'] == 'row-key' for _, m in cells)


def test_text_failure_still_commits_independent_visual_index(pipeline, paper):
    pipeline.embedding.fail = True
    manager = manager_for(pipeline, SimpleNamespace(build=lambda *a: {'status': 'ready'}))
    result = manager.ingest(str(paper))
    doc = manager.store.get(result.document_id)
    assert doc.active_version == 'v1'
    assert doc.metadata['text_index']['status'] == 'failed'
    assert doc.metadata['visual']['status'] == 'ready'
    from sage_scholar.rag.pipeline import Pipeline
    assert paths(pipeline.vector_store) == paths(Pipeline(pipeline.data_dir).vector_store) == set()


def test_both_channels_failed_do_not_commit(pipeline, paper):
    pipeline.embedding.fail = True
    manager = manager_for(pipeline, SimpleNamespace(build=lambda *a: {'status': 'failed'}))
    with pytest.raises(RuntimeError, match='embedding failure'):
        manager.ingest(str(paper))
    assert not manager.store.list()[0].active_version


def ev(name, page, version='v1', visual=False):
    return EvidenceRef(evidence_id=name, source_id='pdf', document_id='doc', source_version=version,
                       page=page, original_text=name, metadata={'kind': 'visual_source'} if visual else {})


def test_page_votes_deduplicate_each_channel_preserve_versions_and_ids():
    text = [ev('a', 1), ev('a-section', 1), ev('b', 2)]
    images = [ev('b-image', 2, visual=True), ev('c', 3, visual=True)]
    result = fuse(text, images, 6, strategy='page', max_pages=3)
    assert result[0].page == 2
    assert {x.evidence_id for x in result} == {x.evidence_id for x in text + images}
    assert len({x.evidence_id for x in result}) == len(result)
    # Another version never supplies a second vote for v1 page 1.
    result = fuse(text, [ev('other-version', 1, 'v2')], 6, strategy='page', max_pages=3)
    assert result[0].source_version == 'v1' and result[0].page == 1
    assert retrieval_metrics(['p:1'], ['p:1','p:2'], 3)['all_evidence_hit@3'] == 0


def test_pure_visual_ignores_page_hint(library):
    adapter, doc, _ = library
    # Only page 1 exists. Locator-assisted search correctly honors page 99,
    # while pure vectors must still rank the existing page from the image index.
    assisted, _ = adapter.visual.search('page 99', [doc], 3)
    pure, warnings = adapter.visual.search('page 99', [doc], 3, locator_assistance=False)
    assert assisted == [] and not warnings and pure[0].page == 1


def test_caption_alignment_is_unique_not_fuzzy():
    from evals.runners.spiqa_holdout import locate
    assert locate('Figure 1: A description of the architecture', ['different', 'Figure 1: A description of the\narchitecture']) == [2]
    assert locate('Figure 1: A description of the architecture', ['Figure 1: A description of the architecture']*2) == [1,2]
    assert not locate('short', ['short'])


def test_selection_is_stratified_and_does_not_use_answer_or_gold_pages():
    from evals.runners.multimodal_final import stratify, group
    rows = [{'id':f'{kind}{i}', 'doc_name':str(i), 'subimg_type':types, 'answer':'gold'}
        for kind,types in [('text',[['text']]),('image',[['image']]),('table',[['table']]),('mixed',[['image','text']])]
        for i in range(10)]
    selected = stratify(rows, 20, 42)
    changed = [{**r,'answer':'wrong','evidence_page':[999]} for r in rows]
    assert [r['id'] for r in selected] == [r['id'] for r in stratify(changed,20,42)]
    assert all(sum(group(r)==kind for r in selected)==5 for kind in ['text','image','table','mixed'])


def test_gold_arm_does_not_expose_reference_answer(library):
    from evals.runners.multimodal_answers import gold_evidence, SelectedEvidence
    from sage_scholar.domain.execution import SourcePolicy
    adapter, doc, _ = library
    gold = gold_evidence(adapter, doc, [1])
    assert gold and all(e.page == 1 for e in gold)
    scoped = SelectedEvidence(adapter, gold, True)
    with pytest.raises(ValueError, match='SourcePolicy'):
        scoped.retrieve_with_warnings('query',SourcePolicy(source_scope='selected_documents',document_ids=('other',)))


def test_eval_component_failure_does_not_erase_other_channels(tmp_path):
    import json
    from evals.runners.multimodal_final import retrieval
    text, visual = ev('a',1), ev('b',1,visual=True)
    adapter = SimpleNamespace(documents=SimpleNamespace(list=lambda:[SimpleNamespace(title='paper', document_id='doc',active_version='v1')]),
        evidence_store=SimpleNamespace(get=lambda key:{'a':text,'b':visual}[key]),
        visual=SimpleNamespace(search=lambda *a,**k:([],['encoder disconnected'])))
    row={'id':'q','doc_name':'paper','subimg_type':[['text']],'query':'question','evidence_page':[1]}
    baseline={'samples':[{'id':'q','results':{'text':{'evidence_ids':['a'],'seconds':1},'visual':{'evidence_ids':['b'],'seconds':1}}}]}
    retrieval(SimpleNamespace(retry_failed=False),baseline,adapter,[row],tmp_path)
    result=json.loads((tmp_path/'retrieval/q.json').read_text())['results']
    assert result['text']['metrics']['recall@3']==1
    assert result['page_rrf']['metrics']['recall@3']==1
    assert 'error' in result['visual_pure'] and 'error' in result['page_rrf_pure']


def test_eval_resume_preserves_success_and_failure_attempts(tmp_path):
    import json
    from evals.runners.multimodal_answers import completed,provider_halted
    p=tmp_path/'answer.json'
    p.write_text(json.dumps({'answer':'done'}))
    assert completed(p,True) and p.exists()
    p.write_text(json.dumps({'error':'timeout'}))
    assert not completed(p,True) and not p.exists()
    assert json.loads(next((tmp_path/'attempts').glob('*.json')).read_text())['error']=='timeout'
    assert not provider_halted(tmp_path)
    assert provider_halted(tmp_path,SimpleNamespace(status_code=402))


def test_judgment_resume_invalidates_replaced_answer(tmp_path, monkeypatch):
    import asyncio
    import hashlib
    import json
    from evals.runners.multimodal_answers import judge_one
    from evals.judges import multimodal
    answer = tmp_path / 'answers/q/text.json'
    judgment = tmp_path / 'judgments/q/text.json'
    answer.parent.mkdir(parents=True)
    judgment.parent.mkdir(parents=True)
    answer.write_text(json.dumps({'answer': 'replacement'}))
    judgment.write_text(json.dumps({'answer_sha256': 'previous', 'metrics': {}}))
    calls = []
    def score(*args):
        calls.append(args)
        return {'metrics': {}}
    monkeypatch.setattr(multimodal, 'judge_answer', score)
    args = SimpleNamespace(retry_failed=True, judge_model='mock')
    row = {'id': 'q', 'subimg_type': [['text']]}
    asyncio.run(judge_one(args, None, row, 'text', tmp_path))
    asyncio.run(judge_one(args, None, row, 'text', tmp_path))
    assert len(calls) == 1
    assert json.loads(judgment.read_text())['answer_sha256'] == hashlib.sha256(answer.read_bytes()).hexdigest()
    assert len(list((judgment.parent / 'attempts').glob('*-stale-*.json'))) == 1


def test_partial_report_compares_only_paired_assessable_questions(tmp_path):
    import hashlib
    import json
    from evals.runners.multimodal_answers import report
    rows = [{'id': q, 'subimg_type': [['text']]} for q in ('q1', 'q2')]
    for q in ('q1', 'q2'):
        for arm in ('text', 'multimodal', 'gold'):
            answer = tmp_path / 'answers' / q / (arm + '.json')
            answer.parent.mkdir(parents=True, exist_ok=True)
            answer.write_text(json.dumps({'id': q, 'arm': arm, 'group': 'text', 'answer': 'done', 'seconds': 1}))
            if q == 'q2' and arm == 'gold':
                continue
            judgment = tmp_path / 'judgments' / q / (arm + '.json')
            judgment.parent.mkdir(parents=True, exist_ok=True)
            metrics = {m: {'score': 1 if q == 'q1' else 0} for m in ('answer_correctness', 'faithfulness', 'citation_accuracy')}
            if arm == 'text':
                metrics['faithfulness']['score'] = None
            judgment.write_text(json.dumps({'id': q, 'arm': arm, 'group': 'text', 'metrics': metrics,
                'answer_sha256': hashlib.sha256(answer.read_bytes()).hexdigest()}))
    report(tmp_path, rows)
    result = json.loads((tmp_path / 'answer-summary.json').read_text())
    assert result['status'] == 'partial' and result['unscored_judgments'] == 1
    assert result['paired_question_ids'] == ['q1']
    for row in result['paired_groups'][:3]:
        assert row['metrics']['answer_correctness'] == {'mean': 1, 'n': 1}
        assert row['metrics']['faithfulness'] == {'mean': None, 'n': 0}
