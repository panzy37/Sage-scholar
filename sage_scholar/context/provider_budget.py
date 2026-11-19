# 功能：在模型请求前检查完整请求是否满足提供商上下文上限。
"""Provider ceiling guard, separate from the estimated working-context budget.

An OpenAI-compatible API does not imply an OpenAI tokenizer. A deployment may
pin a matching tokenizer.json via LLM_TOKENIZER_PATH + LLM_TOKENIZER_MODEL.
Without it, UTF-8 bytes conservatively bound byte-based text tokenization;
this is deliberately reported as a bound, never as exact provider usage.
"""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from .manager import ContextBudgetExceeded


@lru_cache(maxsize=8)
def _tokenizer(path, modified_ns):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_provider_budget(model, messages, tools, tool_choice, output_tokens, limit):
    payload = json.dumps({"messages": messages, "tools": tools, "tool_choice": tool_choice},
                         ensure_ascii=False)
    path = os.getenv("LLM_TOKENIZER_PATH")
    if path:
        if os.getenv("LLM_TOKENIZER_MODEL") != model:
            raise ContextBudgetExceeded("LLM_TOKENIZER_MODEL 必须与当前调用模型一致")
        tokenizer, digest = _tokenizer(path, Path(path).stat().st_mtime_ns)
        count = len(tokenizer.encode(payload).ids)
        method = "pinned tokenizer.json + serialization/framing margin"
    else:
        count, digest = len(payload.encode("utf-8")), None
        method = "UTF-8 byte upper bound + framing margin (not exact usage)"
    # Provider chat/tool templates are not exposed by compatible APIs. Keep a
    # separate explicit margin even with a matching text tokenizer.
    margin = max(1024, 32 * len(messages))
    total = count + margin + output_tokens
    if total > limit:
        raise ContextBudgetExceeded(
            f"模型 {model} 单次请求安全边界超限：文本 {count} + 模板余量 {margin} "
            f"+ 输出 {output_tokens} > {limit}；计数方式 {method}")
    return {"model": model, "input_bound": count, "framing_margin": margin,
            "output_reserve": output_tokens, "total_bound": total,
            "context_limit": limit, "method": method, "tokenizer_sha256": digest}
