from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..contracts import (
    ContextSnapshot,
    ProviderEvent,
    ShareEvent,
    TaskInteractionReply,
    TaskQuery,
    TaskRequest,
    TaskResult,
    TaskUpdate,
    WorkState,
)
from ..coordination import BackendCapabilities, ProjectRecord, WorkerRequest
from ..gateway import WorkerControl, WorkerRunChannel
from .registry import ProviderBuildContext, ProviderRegistration

LOGGER = logging.getLogger(__name__)


OPENAI_VLLM_WORKER_CAPABILITIES = BackendCapabilities(
    steering="native",
    side_queries="none",
    terminal_side_queries="none",
    interactions=False,
    blocking_granularity="run",
    authority_enforcement="gateway",
    structured_events="none",
    trusted_risk_signals=False,
    session_resume=False,
    modalities=frozenset({"text"}),
    max_parallel_projects=1,
    context_provisioning="push_bounded",
    session="stateful",
)


@dataclass(frozen=True)
class OpenAIVLLMProviderConfig:
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "Qwen3-30B-Omni-A3B-Instruct"
    timeout_sec: float = 120.0
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_parallel_projects: int = 1


@dataclass(frozen=True)
class OpenAIVLLMProviderSettings:
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "Qwen3-30B-Omni-A3B-Instruct"
    timeout_sec: float = 120.0
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_parallel_projects: int = 1


class OpenAIVLLMWorkerProvider:
    name = "openai-vllm"
    capabilities = OPENAI_VLLM_WORKER_CAPABILITIES

    def __init__(self, config: OpenAIVLLMProviderConfig) -> None:
        if not config.base_url:
            raise ValueError("worker.settings.base_url must not be empty")
        if not config.model:
            raise ValueError("worker.settings.model must not be empty")
        if config.timeout_sec <= 0:
            raise ValueError("worker.settings.timeout_sec must be positive")
        if config.max_parallel_projects < 1:
            raise ValueError("worker.settings.max_parallel_projects must be positive")
        self.config = config
        self.capabilities = dataclasses.replace(
            OPENAI_VLLM_WORKER_CAPABILITIES,
            max_parallel_projects=config.max_parallel_projects,
        )
        self._project_lock = asyncio.Lock()
        self._projects: dict[str, _OpenAIVLLMWorkerProject] = {}
        self._closed = False

    async def open_project(
        self, project: ProjectRecord
    ) -> "_OpenAIVLLMWorkerProject":
        if self._closed:
            raise RuntimeError("OpenAI vLLM WorkerProvider is closed")
        if project.provider_name != self.name:
            raise ValueError(
                "project provider does not match OpenAI vLLM WorkerProvider"
            )
        async with self._project_lock:
            existing = self._projects.get(project.project_id)
            if existing is not None:
                return existing
            opened = _OpenAIVLLMWorkerProject(self, project)
            self._projects[project.project_id] = opened
            return opened

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for project in tuple(self._projects.values()):
            await project.close()
        self._projects.clear()


class _OpenAIVLLMWorkerProject:
    def __init__(
        self, provider: OpenAIVLLMWorkerProvider, project: ProjectRecord
    ) -> None:
        self.provider = provider
        self.project = project
        self._closed = False

    async def start(
        self, request: WorkerRequest, control: WorkerControl
    ) -> WorkerRunChannel:
        if self._closed:
            raise RuntimeError("OpenAI vLLM worker project is closed")
        if request.project_id != self.project.project_id:
            raise ValueError("worker request belongs to another project")
        run = _OpenAIVLLMRun(
            TaskRequest(
                task_id=request.task_id,
                session_id=request.lineage_id,
                generation=request.generation,
                instruction=request.instruction,
                context=ContextSnapshot(request.lineage_id, ()),
                work_state=WorkState(),
                request_id=request.run_id,
                metadata={
                    "owner_id": request.owner_id,
                    "project_id": request.project_id,
                    "lineage_id": request.lineage_id,
                },
            ),
            self.provider.config,
        )
        await run.start()
        return WorkerRunChannel(request, control, run, self.provider.capabilities)

    async def close(self) -> None:
        self._closed = True


class _OpenAIVLLMRun:
    def __init__(self, request: TaskRequest, config: OpenAIVLLMProviderConfig) -> None:
        self.request = request
        self.config = config
        self.generation = request.generation
        self.thread_id = request.session_id
        self._queue: asyncio.Queue[ProviderEvent | None] = asyncio.Queue()
        self._terminal = False
        self._closed = False
        self._active: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self._submit(self.request.instruction)

    def events(self) -> AsyncIterator[ProviderEvent]:
        return self._iterate()

    async def steer(self, update: TaskUpdate) -> None:
        instruction = (update.instruction or update.event.text).strip()
        if not instruction:
            instruction = "Continue."
        await self._submit(instruction)

    async def query(self, query: TaskQuery) -> None:
        del query
        raise RuntimeError("OpenAI vLLM provider does not support side queries")

    async def respond(self, reply: TaskInteractionReply) -> bool:
        del reply
        return False

    async def cancel(self) -> None:
        async with self._lock:
            if self._terminal:
                return
            task = self._active
            self._active = None
            self._terminal = True
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._queue.put(
            ProviderEvent(
                kind="result",
                generation=self.generation,
                result=TaskResult(
                    task_id=self.request.task_id,
                    session_id=self.request.session_id,
                    generation=self.generation,
                    status="cancelled",
                    full_result="Cancelled",
                ),
            )
        )
        await self._queue.put(None)

    async def close(self) -> None:
        self._closed = True
        await self.cancel()

    async def _submit(self, instruction: str) -> None:
        async with self._lock:
            if self._terminal or self._closed:
                raise RuntimeError("provider run is closed")
            prior = self._active
            if prior is not None:
                prior.cancel()
            self._active = asyncio.create_task(self._run_completion(instruction))
        if prior is not None:
            await asyncio.gather(prior, return_exceptions=True)

    async def _run_completion(self, instruction: str) -> None:
        await self._queue.put(
            ProviderEvent(
                kind="share",
                generation=self.generation,
                share=ShareEvent(
                    task_id=self.request.task_id,
                    session_id=self.request.session_id,
                    generation=self.generation,
                    kind="activity",
                    text=f"Querying local vLLM model {self.config.model}",
                ),
            )
        )
        try:
            output = await asyncio.to_thread(self._request_completion, instruction)
            status = "completed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.warning("openai vllm request failed", exc_info=True)
            output = f"OpenAI-compatible vLLM request failed: {exc}"
            status = "failed"
        async with self._lock:
            if self._terminal:
                return
            self._terminal = True
            self._active = None
        await self._queue.put(
            ProviderEvent(
                kind="result",
                generation=self.generation,
                result=TaskResult(
                    task_id=self.request.task_id,
                    session_id=self.request.session_id,
                    generation=self.generation,
                    status=status,
                    full_result=output,
                    provider_metadata={
                        "provider": "openai-vllm",
                        "model": self.config.model,
                        "base_url": self.config.base_url,
                    },
                ),
            )
        )
        await self._queue.put(None)

    def _request_completion(self, instruction: str) -> str:
        base = self.config.base_url.rstrip("/")
        url = f"{base}/chat/completions"
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": instruction})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_sec,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"{exc.code} {exc.reason}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc
        body = json.loads(raw)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        text = _extract_text(content).strip()
        if not text:
            raise RuntimeError("response has empty content")
        return text

    async def _iterate(self) -> AsyncIterator[ProviderEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return ""


def _build_openai_vllm_provider(
    context: ProviderBuildContext,
    settings: OpenAIVLLMProviderSettings,
) -> OpenAIVLLMWorkerProvider:
    del context
    return OpenAIVLLMWorkerProvider(
        OpenAIVLLMProviderConfig(
            base_url=settings.base_url,
            model=settings.model,
            timeout_sec=settings.timeout_sec,
            api_key_env=settings.api_key_env,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
            max_parallel_projects=settings.max_parallel_projects,
        )
    )


OPENAI_VLLM_PROVIDER_REGISTRATION = ProviderRegistration(
    key="openai_vllm",
    provider_name=OpenAIVLLMWorkerProvider.name,
    settings_type=OpenAIVLLMProviderSettings,
    build=_build_openai_vllm_provider,
)
