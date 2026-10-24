"""Direct, testable DeepSeek Flash client without a provider adapter."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from st_agent.safety import CircuitBreaker, CircuitOpenError, OutboundGuard
from st_agent.trace import emit


class ProviderToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    name: str
    arguments: str


class ProviderReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    response_id: str = "fake-response"
    content: str | None = None
    tool_calls: tuple[ProviderToolCall, ...] = ()
    finish_reason: str = "stop"


class DeepSeekProtocol(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> ProviderReply: ...


class DeepSeekError(RuntimeError):
    pass


class DeepSeekAuthenticationError(DeepSeekError):
    pass


class DeepSeekProviderPaused(DeepSeekError):
    pass


class DeepSeekMalformedResponse(DeepSeekError):
    pass


class DeepSeekRequestError(DeepSeekError):
    pass


class DeepSeekBudgetExceeded(DeepSeekError):
    """Local request budget exhausted; no HTTP request was sent."""


class DeepSeekClient:
    def close(self) -> None:
        self._client.close()

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        timeout_seconds: float = 30.0,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
        outbound_guard: OutboundGuard | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        vision_enabled: bool = False,
        max_requests: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
            raise ValueError("max_requests must be a positive integer")
        self.max_requests = max_requests
        self._request_lock = Lock()
        self._request_records: list[dict[str, Any]] = []
        self._api_key = api_key
        self.model = model
        self.retries = retries
        self.outbound_guard = outbound_guard or OutboundGuard()
        self.circuit_breaker = circuit_breaker
        self.vision_enabled = vision_enabled
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict], *, required_tool: str | None = None
    ) -> ProviderReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": "auto" if tools else "none",
            "thinking": {"type": "disabled"},
            "temperature": 0,
        }
        if required_tool is not None:
            payload["tool_choice"] = {"type": "function", "function": {"name": required_tool}}
        # Validate the final tool choice too, before telemetry or transport.
        payload = self.outbound_guard.validate(payload)
        if self.circuit_breaker is not None:
            try:
                self.circuit_breaker.allow()
            except CircuitOpenError as error:
                raise DeepSeekProviderPaused("DeepSeek circuit is open") from error
        for attempt in range(self.retries + 1):
            try:
                response = self._metered_post(payload, attempt + 1)
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                if attempt == self.retries:
                    if self.circuit_breaker is not None:
                        self.circuit_breaker.failure()
                    raise DeepSeekProviderPaused("DeepSeek network unavailable") from error
                time.sleep(0.05 * (2**attempt))
                continue
            if response.status_code in {401, 403}:
                if self.circuit_breaker is not None:
                    self.circuit_breaker.failure()
                raise DeepSeekAuthenticationError("DeepSeek authentication failed")
            if response.status_code in {408, 429, 500, 502, 503, 504}:
                if attempt == self.retries:
                    if self.circuit_breaker is not None:
                        self.circuit_breaker.failure()
                    raise DeepSeekProviderPaused(
                        f"DeepSeek temporarily unavailable ({response.status_code})"
                    )
                time.sleep(0.05 * (2**attempt))
                continue
            try:
                response.raise_for_status()
                payload_out = response.json()
                choice = payload_out["choices"][0]
                message = choice["message"]
                calls = tuple(
                    ProviderToolCall(
                        id=item["id"],
                        name=item["function"]["name"],
                        arguments=item["function"]["arguments"],
                    )
                    for item in message.get("tool_calls", ())
                )
                reply = ProviderReply(
                    response_id=payload_out["id"],
                    content=message.get("content"),
                    tool_calls=calls,
                    finish_reason=choice["finish_reason"],
                )
                if self.circuit_breaker is not None:
                    self.circuit_breaker.success()
                return reply
            except httpx.HTTPStatusError as error:
                if self.circuit_breaker is not None:
                    self.circuit_breaker.failure()
                raise DeepSeekRequestError(
                    f"DeepSeek rejected the request ({response.status_code})"
                ) from error
            except (KeyError, IndexError, TypeError, ValueError) as error:
                if self.circuit_breaker is not None:
                    self.circuit_breaker.failure()
                raise DeepSeekMalformedResponse("DeepSeek returned an invalid response") from error
        raise AssertionError("retry loop must return or raise")

    def request_metrics(self) -> list[dict[str, Any]]:
        with self._request_lock:
            return deepcopy(self._request_records)

    def _metered_post(self, payload: dict, attempt: int) -> httpx.Response:
        image_count = sum(
            part.get('type') == 'image_url'
            for message in payload.get('messages', [])
            if isinstance(message.get('content'), list)
            for part in message['content'] if isinstance(part, dict)
        )
        with self._request_lock:
            if self.max_requests is not None and len(self._request_records) >= self.max_requests:
                raise DeepSeekBudgetExceeded("Local provider request budget exhausted")
            record = {
                "request_index": len(self._request_records) + 1,
                "attempt": attempt,
                "category": "vision" if image_count else ("tool_intent" if payload.get("tools") else "text"),
                "tool_names": [t.get("function", {}).get("name") for t in (payload.get("tools") or [])],
                "image_count": image_count,
                "status_code": None,
                "error_type": None,
                "elapsed_seconds": None,
                "usage": None,
            }
            self._request_records.append(record)
        started = time.perf_counter()
        if getattr(self, "trace", None) is not None:
            from st_agent.trace import redact
            safe_envelope = redact(payload)
            # Image bytes are attachments, not text telemetry. Preserve hashes for
            # image requests without duplicating base64 data in every attempt.
            for message in safe_envelope.get("messages", []):
                if not isinstance(message.get("content"), list):
                    continue
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        image_url = part["image_url"].get("url", "")
                        part["image_url"]["url"] = "[IMAGE_ATTACHMENT]"
                        part["image_url"]["attachment_sha256"] = hashlib.sha256(image_url.encode()).hexdigest()
            encoded = json.dumps(safe_envelope, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()
            wire_encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()
            emit(self, "llm.request", model=self.model, category=record["category"], attempt=attempt,
                 request_index=record["request_index"], image_count=image_count,
                 tool_names=record["tool_names"], outbound_envelope=safe_envelope,
                 outbound_sha256=hashlib.sha256(encoded).hexdigest(),
                 wire_sha256=hashlib.sha256(wire_encoded).hexdigest(),
                 outbound_complete=safe_envelope == payload)
        updates: dict[str, Any] = {}
        try:
            response = self._client.post("/chat/completions", json=payload)
            updates["status_code"] = response.status_code
            try:
                body = response.json()
                usage = body.get("usage") if isinstance(body, dict) else None
                if isinstance(usage, dict):
                    updates["usage"] = {
                        key: usage[key]
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                        if type(usage.get(key)) is int and usage[key] >= 0
                    }
            except ValueError:
                pass
            return response
        except httpx.HTTPError as error:
            updates["error_type"] = type(error).__name__
            raise
        finally:
            updates["elapsed_seconds"] = time.perf_counter() - started
            with self._request_lock:
                record.update(updates)
            emit(self, "llm.response", status="failed" if record["error_type"] or
                 (record["status_code"] is not None and record["status_code"] >= 400) else "ok",
                 duration_ms=record["elapsed_seconds"] * 1000, model=self.model, **record)

    def complete_batch(
        self,
        requests: list[tuple[list[dict[str, Any]], list[dict]]],
        *,
        max_workers: int = 4,
    ) -> list[ProviderReply]:
        if not 1 <= max_workers <= 8:
            raise ValueError("max_workers must be between 1 and 8")
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="deepseek") as pool:
            futures = [pool.submit(self.complete, messages, tools) for messages, tools in requests]
            return [future.result() for future in futures]

    def capabilities(self) -> dict[str, bool | str]:
        return {
            "model": self.model,
            "tool_calling": True,
            "vision": self.vision_enabled,
        }

    def probe(self, tools: list[dict]) -> dict[str, Any]:
        reply = self.complete(
            [{"role": "user", "content": "Call dataset_inspect with no arguments."}], tools
        )
        valid = bool(reply.tool_calls and reply.tool_calls[0].name == "dataset_inspect")
        if not valid:
            raise DeepSeekMalformedResponse("Strict Tool Calling capability probe failed")
        return {"model": self.model, "strict_tool_calling": True, "response_id": reply.response_id}


class FakeDeepSeek:
    """Deterministic provider double used by all mandatory Agent tests."""

    def __init__(self, script: Iterable[ProviderReply | BaseException]) -> None:
        self.script = deque(script)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> ProviderReply:
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            raise DeepSeekMalformedResponse("FakeDeepSeek script exhausted")
        item = self.script.popleft()
        if isinstance(item, BaseException):
            if isinstance(item, DeepSeekError):
                raise item
            raise DeepSeekProviderPaused("DeepSeek unavailable") from item
        return item
