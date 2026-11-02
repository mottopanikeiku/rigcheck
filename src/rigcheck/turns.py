"""Agent-turn timing. A scripted five-turn debugging session is replayed with the agent's
real prompt and tools. Turn 1 starts with a unique nonce, so nothing can come from cache;
turns 2-5 only append a tool call and its result, so a working prefix cache should make
them pay only for the new tokens.

Prefix reuse is estimated from timing alone, so it works on any engine: turn 1 gives the
cold prefill speed, and a warm turn's first-token time says how much of its prompt was
really processed. 100% means everything sent before was reused; 0% means the whole
prompt was processed again."""

from __future__ import annotations

import json
import secrets

from . import project
from .client import Client
from .profiles import Profile

TASK = "The per-customer totals in the billing report are off by a cent now and then. Find the cause and fix it."


def script(profile: Profile) -> list[dict]:
    """History appended after the task: one (assistant call, tool result) pair per turn."""
    steps = [
        ("search", {"pattern": "round("}, project.grep_output()),
        ("read", {"path": "src/app/billing.py"}, project.billing_module()),
        (
            "edit",
            {
                "path": "src/app/billing.py",
                "old": "return round(float(self.total()) / cached_rate(self.currency), 2)",
                "new": "return float(self.total()) / cached_rate(self.currency)",
            },
            "Edit applied successfully.",
        ),
        ("shell", {"command": "pytest -q", "description": "Run the test suite"}, project.pytest_failure_output()),
    ]
    history = []
    for i, (kind, fields, result) in enumerate(steps, 1):
        cid = f"call_turn_{i}"
        history.append({"role": "assistant", "content": "", "tool_calls": [profile.call(kind, cid, **fields)]})
        history.append({"role": "tool", "tool_call_id": cid, "content": result})
    return history


def run_session(client: Client, profile: Profile, base: dict, session: int, log=None) -> list[dict]:
    nonce = secrets.token_hex(6)
    preamble = profile.build_messages(TASK, nonce=nonce)
    history = script(profile)
    rows = []
    tools_chars = len(json.dumps(base.get("tools") or []))
    for turn in range(1, len(history) // 2 + 2):
        messages = preamble + history[: 2 * (turn - 1)]
        body = {**base, "messages": messages, "stream": True}
        c = client.chat(body, headers={"X-Rigcheck-Probe": f"turns/{turn}", "X-Rigcheck-Agent": profile.name})
        t = c.timings or {}
        row = {
            "session": session,
            "turn": turn,
            "ok": c.ok,
            "error": c.error,
            "prompt_chars": tools_chars + len(json.dumps(messages)),
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "ttft": c.ttft,
            "total": c.total,
            "decode_tps": c.decode_rate(),
            "finish_reason": c.finish_reason,
            "engine_prompt_n": t.get("prompt_n"),
            "engine_cache_n": t.get("cache_n"),
            "engine_prompt_ms": t.get("prompt_ms"),
        }
        rows.append(row)
        if log:
            tt = f"{c.ttft:.2f}s" if c.ttft is not None else "-"
            log(f"  {profile.name} session {session} turn {turn}: prompt {c.prompt_tokens} tok, first token {tt}" + ("" if c.ok else f"  ERROR {c.error}"))
        if not c.ok:
            break
    return rows


def prefix_reuse(session_rows: list[dict]) -> list[float]:
    """Estimated share of the previous prompt each warm turn reused (see module doc)."""
    rows = sorted((r for r in session_rows if r["ok"] and r["ttft"]), key=lambda r: r["turn"])
    if not rows or rows[0]["turn"] != 1:
        return []
    t1, c1 = rows[0]["ttft"], rows[0]["prompt_chars"]
    out = []
    for prev, cur in zip(rows, rows[1:]):
        if cur["turn"] != prev["turn"] + 1:
            break
        processed = cur["ttft"] / t1 * c1
        out.append(min(1.0, max(0.0, (cur["prompt_chars"] - processed) / prev["prompt_chars"])))
    return out
