"""Minimal OpenAI-compatible chat client that keeps what agents depend on: streamed
tool-call deltas assembled by index, reasoning kept apart from content, and timing."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import httpx


@dataclass
class ToolCall:
    index: int | None
    id: str | None
    name: str | None
    arguments: str = ""


@dataclass
class Completion:
    ok: bool
    status: int | None = None
    error: str | None = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None
    timings: dict | None = None  # llama.cpp extension
    ttft: float | None = None  # seconds until the first generated token of any kind
    total: float = 0.0
    token_times: list[float] = field(default_factory=list)
    stream_issues: list[str] = field(default_factory=list)

    @property
    def completion_tokens(self) -> int | None:
        if self.usage and isinstance(self.usage.get("completion_tokens"), int):
            return self.usage["completion_tokens"]
        return len(self.token_times) or None

    @property
    def prompt_tokens(self) -> int | None:
        if self.usage and isinstance(self.usage.get("prompt_tokens"), int):
            return self.usage["prompt_tokens"]
        return None

    def decode_rate(self) -> float | None:
        """Tokens per second after the first token."""
        n = self.completion_tokens
        if not n or n < 2 or len(self.token_times) < 2:
            return None
        span = self.token_times[-1] - self.token_times[0]
        return (n - 1) / span if span > 0 else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("token_times")
        d["completion_tokens"] = self.completion_tokens
        d["decode_tps"] = self.decode_rate()
        return d


class Client:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key or 'none'}"}
        self.http = httpx.Client(headers=headers, timeout=httpx.Timeout(timeout, connect=10.0))

    def close(self) -> None:
        self.http.close()

    def get_json(self, url: str) -> dict | list | None:
        try:
            r = self.http.get(url, timeout=10.0)
            if r.status_code == 200:
                return r.json()
        except (httpx.HTTPError, ValueError):
            pass
        return None

    def chat(self, body: dict, headers: dict | None = None) -> Completion:
        body = dict(body)
        if body.get("stream"):
            body.setdefault("stream_options", {"include_usage": True})
            return self._stream(body, headers or {})
        return self._plain(body, headers or {})

    def _plain(self, body: dict, headers: dict) -> Completion:
        t0 = time.perf_counter()
        try:
            r = self.http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            return Completion(ok=False, error=f"{type(e).__name__}: {e}", total=time.perf_counter() - t0)
        total = time.perf_counter() - t0
        if r.status_code != 200:
            return Completion(ok=False, status=r.status_code, error=r.text[:500], total=total)
        try:
            data = r.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError):
            return Completion(ok=False, status=r.status_code, error=f"unparseable response: {r.text[:300]}", total=total)
        msg = choice.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):  # some servers send an object
                args = json.dumps(args) if args is not None else ""
            calls.append(ToolCall(index=tc.get("index", i), id=tc.get("id"), name=fn.get("name"), arguments=args))
        return Completion(
            ok=True,
            status=200,
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            timings=data.get("timings"),
            ttft=total,
            total=total,
            token_times=[total],
        )

    def _stream(self, body: dict, headers: dict) -> Completion:
        out = Completion(ok=True)
        calls: dict[int, ToolCall] = {}
        order: list[int] = []
        t0 = time.perf_counter()
        try:
            with self.http.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as r:
                out.status = r.status_code
                if r.status_code != 200:
                    out.ok = False
                    out.error = r.read().decode("utf-8", "replace")[:500]
                    out.total = time.perf_counter() - t0
                    return out
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        out.stream_issues.append("unparseable SSE chunk")
                        continue
                    if "error" in chunk and not chunk.get("choices"):
                        out.ok = False
                        out.error = json.dumps(chunk["error"])[:500]
                        continue
                    if chunk.get("usage"):
                        out.usage = chunk["usage"]
                    if chunk.get("timings"):
                        out.timings = chunk["timings"]
                    for choice in chunk.get("choices") or []:
                        now = time.perf_counter() - t0
                        delta = choice.get("delta") or {}
                        got = False
                        if delta.get("content"):
                            out.content += delta["content"]
                            got = True
                        r_text = delta.get("reasoning_content") or delta.get("reasoning")
                        if r_text:
                            out.reasoning += r_text
                            got = True
                        for tc in delta.get("tool_calls") or []:
                            got = True
                            idx = tc.get("index")
                            if idx is None:
                                out.stream_issues.append("tool_call delta without index")
                                idx = order[-1] if order and not tc.get("id") else len(order)
                            call = calls.get(idx)
                            if call is None:
                                call = ToolCall(index=idx, id=tc.get("id"), name=None)
                                calls[idx] = call
                                order.append(idx)
                            elif tc.get("id") and call.id and tc["id"] != call.id:
                                out.stream_issues.append("tool_call index reused by a different call id")
                            if tc.get("id") and not call.id:
                                call.id = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                if call.name and call.name != fn["name"]:
                                    out.stream_issues.append("tool_call name changed mid-stream")
                                    call.name = call.name + fn["name"]
                                else:
                                    call.name = fn["name"]
                            if fn.get("arguments"):
                                a = fn["arguments"]
                                call.arguments += a if isinstance(a, str) else json.dumps(a)
                        if got:
                            if out.ttft is None:
                                out.ttft = now
                            out.token_times.append(now)
                        if choice.get("finish_reason"):
                            out.finish_reason = choice["finish_reason"]
        except httpx.HTTPError as e:
            out.ok = False
            out.error = f"{type(e).__name__}: {e}"
        out.total = time.perf_counter() - t0
        out.tool_calls = [calls[i] for i in order]
        for c in out.tool_calls:
            if not c.id:
                out.stream_issues.append("tool call without id")
                break
        if out.ok and out.finish_reason is None:
            out.stream_issues.append("stream ended without finish_reason")
        return out
