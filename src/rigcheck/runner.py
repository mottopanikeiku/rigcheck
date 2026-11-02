"""Run probes and timed sessions against one endpoint and collect raw results."""

from __future__ import annotations

import datetime as dt
import json
import platform
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import __version__, engine, turns
from .client import Client
from .probes import PROBES, score
from .profiles import Profile, load


@dataclass
class RunConfig:
    base_url: str
    model: str | None = None
    agents: list[str] = field(default_factory=lambda: ["opencode"])
    probes: list[str] = field(default_factory=lambda: list(PROBES))
    trials: int = 3
    sessions: int = 2
    stream: bool = True
    temperature: float | None = None
    top_p: float | None = None
    seed: int = 1
    max_tokens: int = 4096
    turn_tokens: int = 256
    concurrency: int = 1
    extra: dict = field(default_factory=dict)
    api_key: str | None = None
    timeout: float = 900.0
    label: str | None = None
    fit: bool = True


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _base_body(cfg: RunConfig, profile: Profile) -> dict:
    body: dict = {"model": cfg.model, "tools": profile.tools, "tool_choice": "auto"}
    if "parallel_tool_calls" in profile.params:
        body["parallel_tool_calls"] = profile.params["parallel_tool_calls"]
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        body["top_p"] = cfg.top_p
    body.update(cfg.extra)
    return body


def _min_prompt_tokens(profile: Profile) -> int:
    """A floor on the prompt size that holds for any tokenizer and chat template: the
    message text plus the tool descriptions, at six characters per token."""
    chars = 0
    for m in profile.build_messages("Say ok."):
        c = m["content"]
        chars += len(c) if isinstance(c, str) else sum(len(p.get("text", "")) for p in c if isinstance(p, dict))

    def walk(x):
        nonlocal chars
        if isinstance(x, dict):
            if isinstance(x.get("description"), str):
                chars += len(x["description"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(profile.tools)
    return chars // 6


def fit_check(client: Client, cfg: RunConfig, profile: Profile, info: dict) -> dict:
    """One cold request: does the engine see the whole agent prompt?"""
    nonce = secrets.token_hex(6)
    body = {**_base_body(cfg, profile), "messages": profile.build_messages("Say ok.", nonce=nonce), "tools": profile.build_tools(nonce)}
    body.update({"stream": True, "max_tokens": 8})
    c = client.chat(body, headers={"X-Rigcheck-Probe": "fit/0", "X-Rigcheck-Agent": profile.name})
    floor = _min_prompt_tokens(profile)
    ctx = engine.loaded_context(client, info)
    out = {"ok": c.ok, "error": c.error, "prompt_tokens": c.prompt_tokens, "floor_tokens": floor, "context": ctx, "ttft": c.ttft}
    if not c.ok:
        out["verdict"] = "error"
    elif c.prompt_tokens is not None and c.prompt_tokens < floor:
        out["verdict"] = "truncated"
    else:
        out["verdict"] = "fits"
    return out


def _probe_trial(client: Client, cfg: RunConfig, profile: Profile, base: dict, probe_id: str, trial: int) -> dict:
    probe = PROBES[probe_id]
    idx = trial % len(probe.instances)
    inst = probe.instances[idx]
    task, history, overrides = probe.build(profile, inst)
    body = {**base, "messages": profile.build_messages(task) + history, "stream": cfg.stream, "max_tokens": cfg.max_tokens}
    body.update(overrides)
    body["seed"] = cfg.seed * 1000 + trial
    c = client.chat(body, headers={"X-Rigcheck-Probe": f"{probe_id}/{idx}", "X-Rigcheck-Agent": profile.name})
    row = {
        "probe": probe_id,
        "instance": idx,
        "trial": trial,
        "ok": c.ok,
        "status": c.status,
        "error": c.error,
        "content": c.content,
        "reasoning_chars": len(c.reasoning),
        "tool_calls": [{"index": t.index, "id": t.id, "name": t.name, "arguments": t.arguments} for t in c.tool_calls],
        "finish_reason": c.finish_reason,
        "stream_issues": c.stream_issues,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "ttft": c.ttft,
        "total": c.total,
        "engine_prompt_n": (c.timings or {}).get("prompt_n"),
        "engine_cache_n": (c.timings or {}).get("cache_n"),
    }
    row.update(score(row, profile))
    return row


def run(cfg: RunConfig) -> dict:
    client = Client(cfg.base_url, cfg.api_key, cfg.timeout)
    try:
        if not cfg.model:
            cfg.model = engine.first_model(client)
            if not cfg.model:
                raise SystemExit(f"no model given and {cfg.base_url}/models listed none; pass --model")
        info = engine.describe(client, cfg.model)
        _log(f"rigcheck {__version__}: {info['engine']} {info.get('version') or ''} at {cfg.base_url}, model {cfg.model}")
        warm = client.chat({"model": cfg.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": True})
        if not warm.ok:
            raise SystemExit(f"endpoint error on a one-token request: {warm.status} {warm.error}")
        results = {
            "rigcheck": __version__,
            "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "label": cfg.label,
            "endpoint": info,
            "host": {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version()},
            "config": {k: v for k, v in asdict(cfg).items() if k not in ("api_key",)},
            "agents": {},
        }
        for name in cfg.agents:
            profile = load(name)
            base = _base_body(cfg, profile)
            entry: dict = {
                "profile": {"name": profile.name, "source": name, "title": profile.title, "version": profile.version, "prompt_chars": profile.prompt_chars()},
            }
            if cfg.fit:
                _log(f"{profile.title}: checking the prompt fits")
                entry["fit"] = fit_check(client, cfg, profile, info)
                f = entry["fit"]
                _log(f"  {f['verdict']}: engine counted {f['prompt_tokens']} prompt tokens (floor {f['floor_tokens']}, context {f['context']})")
            jobs = [(p, t) for p in cfg.probes for t in range(cfg.trials)]
            rows: list[dict] = []
            if jobs:
                _log(f"{profile.title}: {len(jobs)} probe trials")
                with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
                    # the first trial alone, so the parallel ones find the agent's prompt in cache
                    first = pool.submit(_probe_trial, client, cfg, profile, base, *jobs[0])
                    first.result()
                    futs = [first] + [pool.submit(_probe_trial, client, cfg, profile, base, p, t) for p, t in jobs[1:]]
                    for fut in futs:
                        r = fut.result()
                        rows.append(r)
                        mark = "pass" if r["pass"] else f"FAIL {r['reason']}"
                        _log(f"  {r['probe']:<16} #{r['trial']:<3} {mark}")
            entry["probes"] = rows
            sessions = []
            for s in range(cfg.sessions):
                _log(f"{profile.title}: timed session {s + 1}/{cfg.sessions}")
                sessions += turns.run_session(client, profile, {**base, "max_tokens": cfg.turn_tokens, "seed": cfg.seed}, s, _log)
            entry["turns"] = sessions
            results["agents"][name] = entry
        results["endpoint"]["context_loaded"] = engine.loaded_context(client, info)
        results["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        return results
    finally:
        client.close()


def save(results: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
