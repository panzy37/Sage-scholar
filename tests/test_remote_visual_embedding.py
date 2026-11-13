import json

import httpx
import numpy as np
import pytest

from sage_scholar.visual.config import VisualConfig
from sage_scholar.visual.embedding import VisualEmbedding
from sage_scholar.visual import remote_embedding as remote


def mock_client(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(remote.httpx, "Client", lambda **kw: client(
        transport=httpx.MockTransport(handler), **kw))


def config(**kw):
    return VisualConfig(embedding_provider="remote", embedding_dimensions=3, **kw)


def test_images_and_queries_share_remote_template_without_local_load(tmp_path, monkeypatch):
    path = tmp_path / 'page.png'
    path.write_bytes(b'png fixture')
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        assert request.headers['authorization'] == 'Bearer private-key'
        return httpx.Response(200, json={'data': [{'embedding': [3, 4, 0]}]})
    monkeypatch.setenv('VISUAL_EMBEDDING_API_KEY', 'private-key')
    mock_client(monkeypatch, handle)
    encoder = VisualEmbedding(config())
    vectors = encoder.encode([{'image': str(path)}, 'question'])
    assert encoder._model is None
    assert np.allclose(vectors, [[.6, .8, 0], [.6, .8, 0]])
    assert requests[0]['messages'][1]['content'][0]['image_url']['url'].startswith('data:image/png;base64,')
    assert str(path) not in json.dumps(requests)
    assert requests[0]['messages'][0] == requests[1]['messages'][0]
    assert all(r['continue_final_message'] and r['add_special_tokens'] for r in requests)
    assert 'private-key' not in repr(encoder.identity)
    assert encoder.identity != VisualEmbedding(VisualConfig(embedding_provider='local')).identity


@pytest.mark.parametrize('vector', [[1, 2], [0, 0, 0], ['bad', 1, 2]])
def test_invalid_vectors_fail_without_fallback(monkeypatch, vector):
    mock_client(monkeypatch, lambda r: httpx.Response(200, json={'data': [{'embedding': vector}]}))
    with pytest.raises(ValueError, match='invalid vectors'):
        VisualEmbedding(config()).encode(['query'])


def test_timeout_is_bounded_and_not_retried(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout('private server response', request=request)
    mock_client(monkeypatch, handle)
    with pytest.raises(RuntimeError, match='ReadTimeout') as error:
        VisualEmbedding(config()).encode(['query', 'other'])
    assert len(calls) == 1 and 'private server response' not in str(error.value)


def test_http_error_does_not_expose_response(monkeypatch):
    mock_client(monkeypatch, lambda r: httpx.Response(500, text='sensitive body'))
    with pytest.raises(RuntimeError, match='HTTP 500') as error:
        VisualEmbedding(config()).encode(['query'])
    assert 'sensitive' not in str(error.value)


def test_cancellation_stops_next_request(monkeypatch):
    from sage_scholar.runtime.journal import WorkflowInterrupted
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={'data': [{'embedding': [1, 0, 0]}]})
    def check():
        if calls:
            raise WorkflowInterrupted('cancelled')
    mock_client(monkeypatch, handle)
    monkeypatch.setattr(remote, 'check_control', check)
    with pytest.raises(WorkflowInterrupted):
        VisualEmbedding(config()).encode(['query', 'other'])
    assert len(calls) == 1


def test_remote_url_rejects_embedded_secrets():
    with pytest.raises(ValueError, match='base URL'):
        config(embedding_base_url='http://user:password@localhost:8000/v1')
