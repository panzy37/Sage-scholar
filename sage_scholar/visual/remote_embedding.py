"""vLLM chat embeddings for aligned text/image retrieval; no local fallback."""
import base64
import mimetypes
import os
from pathlib import Path

import httpx
import numpy as np

from sage_scholar.runtime.control import check_control


def request_body(value, model):
    if isinstance(value, str):
        content = [{"type": "text", "text": value}]
    elif isinstance(value, dict) and set(value) == {"image"}:
        path = Path(value["image"])
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("Unsupported visual embedding image format")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Visual embedding image exceeds 20 MiB")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                   {"type": "text", "text": ""}]
    else:
        raise ValueError("Visual embedding expects query text or a local image")
    # Official Qwen/vLLM pooling template: include the empty assistant prefix.
    return {"model": model, "messages": [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]}],
        "encoding_format": "float", "continue_final_message": True, "add_special_tokens": True}


def encode_remote(inputs, config):
    headers = {}
    key = os.getenv(config.embedding_api_key_env, "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    base = config.embedding_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    vectors = []
    # Explicit tunnel/service address must not be sent through a global HTTP proxy.
    # One bounded request per item; no implicit retry after cancellation/failure.
    with httpx.Client(timeout=httpx.Timeout(config.embedding_timeout, connect=10),
                      trust_env=False, headers=headers) as client:
        for value in inputs:
            check_control()
            body = request_body(value, config.embedding_model)
            try:
                response = client.post(base + "/embeddings", json=body)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(f"Remote visual embedding HTTP {exc.response.status_code}") from None
            except httpx.RequestError as exc:
                raise RuntimeError(f"Remote visual embedding connection failed ({type(exc).__name__})") from None
            check_control()
            try:
                data = response.json()["data"]
                if len(data) != 1:
                    raise ValueError("Unexpected embedding count")
                vector = np.asarray(data[0]["embedding"], dtype=float)
                norm = np.linalg.norm(vector)
                if (vector.shape != (config.embedding_dimensions,) or not np.isfinite(vector).all()
                        or not np.isfinite(norm) or norm == 0):
                    raise ValueError("Invalid embedding vector")
            except (KeyError, TypeError, ValueError, IndexError):
                raise ValueError("Remote visual embedding returned invalid vectors/dimensions") from None
            vectors.append((vector / norm).tolist())
    return vectors
