import asyncio
import json
from sage_scholar.context.source_view import compact_location, evidence_windows, text_windows
from sage_scholar.context.token_counter import TokenCounter
from sage_scholar.domain import EvidenceRef
from sage_scholar.application.qa_service import QAService
from sage_scholar.context.manager import ContextManager
from test_reading_citations import request
from test_query_service import FakeModel


def test_character_coordinates_do_not_enter_prompt_and_original_remains():
    location={'chunk_index':3,'source_locations':[{'bbox':[1.12345,2.45678,3.45678,4.5]}]*12000}
    item=EvidenceRef(evidence_id='ev_layout',source_id='source',text='Important result',page=7,location=location)
    counter=TokenCounter();windows=list(evidence_windows([item],counter,256))
    assert len(windows)==1 and counter.count(windows[0])<=256
    assert 'Important result' in windows[0] and 'page=7' in windows[0] and '[ev_layout]' in windows[0]
    assert 'bbox' not in windows[0] and len(item.location['source_locations'])==12000


def test_long_unicode_source_is_partitioned_without_loss():
    text=('表格:α β 123.45 中文边界\n'*1800)+'END_EXPERIMENT_VALUE=42'
    counter=TokenCounter();parts=list(text_windows(text,counter,200))
    assert ''.join(part for _,part in parts)==text
    assert all(counter.count(part)<=200 for _,part in parts)
    assert parts[-1][1].endswith('END_EXPERIMENT_VALUE=42')


def test_qa_context_failure_is_explicit_partial_before_model_call(tmp_path):
    class Retrieval:
        async def retrieve(self,*a,**kw):return [EvidenceRef(evidence_id='ev_x',source_id='s',text='fact')]
    model=FakeModel('should not be called')
    manager=ContextManager(tmp_path,max_tokens=300)
    result=asyncio.run(QAService(Retrieval(),model,context_manager=manager).answer(request('给定论文有哪些实验结果？','d')))
    assert result.partial_reason and not model.calls
    assert '未能' in result.answer
    assert result.evidence[0].evidence_id == 'ev_x'


def test_provider_guard_counts_tools_history_and_output(monkeypatch):
    import pytest
    from sage_scholar.context.provider_budget import check_provider_budget
    from sage_scholar.context.manager import ContextBudgetExceeded
    monkeypatch.delenv('LLM_TOKENIZER_PATH', raising=False)
    messages = [{'role': 'system', 'content': '规则'},
                {'role': 'user', 'content': '历史' * 100}]
    small = check_provider_budget('deepseek-flash', messages, None, None, 100, 10000)
    assert small['total_bound'] <= small['context_limit']
    assert 'not exact' in small['method']
    with pytest.raises(ContextBudgetExceeded):
        check_provider_budget('deepseek-flash', messages,
            [{'description': '工具 schema ' * 1000}], 'auto', 100, 10000)
    with pytest.raises(ContextBudgetExceeded):
        check_provider_budget('deepseek-flash', messages, None, None, 9500, 10000)


def test_mismatched_provider_tokenizer_fails_closed(monkeypatch):
    import pytest
    from sage_scholar.context.provider_budget import check_provider_budget
    from sage_scholar.context.manager import ContextBudgetExceeded
    monkeypatch.setenv('LLM_TOKENIZER_PATH', '/not-loaded.json')
    monkeypatch.setenv('LLM_TOKENIZER_MODEL', 'another-model')
    with pytest.raises(ContextBudgetExceeded, match='一致'):
        check_provider_budget('deepseek-flash', [], None, None, 100, 30000)


def test_llm_gate_rejects_before_network_io(tmp_path, monkeypatch):
    import pytest
    from sage_scholar.base.llm_client import LLMClient
    from sage_scholar.context.manager import ContextBudgetExceeded
    monkeypatch.setenv('LLM_CONTEXT_WINDOW', '3000')
    monkeypatch.delenv('LLM_TOKENIZER_PATH', raising=False)
    client = LLMClient(model='deepseek-flash', api_key='offline-test', base_url='http://127.0.0.1:1',
                       context_manager=ContextManager(tmp_path, max_tokens=30000))
    def forbidden(**kwargs):
        raise AssertionError('over-limit request reached network')
    monkeypatch.setattr(client.client.chat.completions, 'create', forbidden)
    with pytest.raises(ContextBudgetExceeded):
        client.invoke([{'role':'user','content':'中' * 900}], max_tokens=100)


def test_tool_reread_geometry_is_archived_not_reinserted_into_prompt(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=8000)
    original = json.dumps({'evidence_id':'ev_long', 'page':7, 'source_id':'paper',
        'original_text':'The final experiment yielded 42.',
        'location':{'page_end':7, 'source_locations':[{'bbox':[1,2,3,4]}]*12000}})
    messages = [{'role':'user','content':'Read the paper'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'c1','type':'function',
          'function':{'name':'document_sections','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'c1','content':original}]
    prepared = manager.prepare(messages, output_reserve=1000)
    pointer = json.loads(prepared.messages[-1]['content'])
    assert pointer['locators'][0]['location']['page_end'] == 7
    assert pointer['locators'][0]['location']['source_location_count'] == 12000
    assert 'bbox' not in json.dumps(pointer['locators'])
    assert prepared.accounting['total_tokens'] < 8000
    restored = ''.join(manager.artifacts.read(pointer['artifact_id'], 'standalone', offset=i, limit=12000)['text']
                       for i in range(0, len(original), 12000))
    assert restored == original


def test_qa_truncated_output_retains_answer_but_is_partial():
    from types import SimpleNamespace
    from sage_scholar.application.ports import ModelText
    class Retrieval:
        async def retrieve(self, *args, **kwargs):
            return [EvidenceRef(evidence_id='ev_x', source_id='s', text='The answer is 42.')]
    text = ModelText(SimpleNamespace(content='42 [ev_x]. More detail:',finish_reason='length'))
    result = asyncio.run(QAService(Retrieval(),FakeModel(text)).answer(request('具体结果是多少？','d')))
    assert result.partial_reason and '输出上限' in result.partial_reason
    assert '42 [1]' in result.answer and result.evidence


def test_model_text_survives_snapshot_deepcopy():
    from copy import deepcopy
    from types import SimpleNamespace
    from sage_scholar.application.ports import ModelText
    value = ModelText(SimpleNamespace(content="partial answer", finish_reason="length"))
    copied = deepcopy({"answer": value})["answer"]
    assert copied == "partial answer" and copied.finish_reason == "length"
