from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.execution import ExecutionRequest, SourcePolicy, RunOutcome


class ModelText(str):
    """String-compatible answer retaining the provider's completion boundary."""
    def __new__(cls, response):
        # copy/deepcopy reconstruct str subclasses from a plain string.
        value = super().__new__(cls, response if isinstance(response, str) else response.content or "")
        value.finish_reason = getattr(response, "finish_reason", None)
        return value


class ModelGateway(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        tag: str = "",
        images: list[dict] | None = None,
    ) -> str: ...


class RetrievalService(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        policy: SourcePolicy,
        top_k: int = 5,
    ) -> list[EvidenceRef]: ...

    async def read_documents(self, *, policy: SourcePolicy) -> list[EvidenceRef]: ...


EventEmitter = Callable[[str, dict], Awaitable[None]]


class LongTaskRunner(Protocol):
    async def run(self, request: ExecutionRequest, emit: EventEmitter, cancelled: Callable[[], bool]) -> RunOutcome: ...
