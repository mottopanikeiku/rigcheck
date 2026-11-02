from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from . import __version__, capture, profiles, report, runner, stats
from .probes import PROBES


def _cmd_run(a: argparse.Namespace) -> None:
    cfg = runner.RunConfig(
        base_url=a.url,
        model=a.model,
        agents=a.agent or ["opencode"],
        probes=a.probe or list(PROBES),
        trials=a.trials,
        sessions=a.sessions,
        stream=not a.no_stream,
        temperature=a.temperature,
        top_p=a.top_p,
        seed=a.seed,
        max_tokens=a.max_tokens,
        turn_tokens=a.turn_tokens,
        concurrency=a.concurrency,
        extra=json.loads(a.extra) if a.extra else {},
        api_key=a.api_key,
        timeout=a.timeout,
        label=a.label,
    )
    for p in cfg.probes:
        if p not in PROBES:
            raise SystemExit(f"unknown probe '{p}'. Probes: {', '.join(PROBES)}")
    res = runner.run(cfg)
    out = a.output
    if not out:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{res['endpoint']['engine']}-{cfg.model}")
        out = f"rigcheck-{slug}-{stamp}.json"
    runner.save(res, out)
    rows = report.summarize(res)
    print(report.terminal(rows))
    print(f"\nraw results: {out}", file=sys.stderr)
    if a.md:
        Path(a.md).write_text(report.markdown(rows))
    if a.html:
        Path(a.html).write_text(report.html(rows))


def _load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        rows += report.summarize(json.loads(Path(p).read_text()))
    return rows


def _cmd_report(a: argparse.Namespace) -> None:
    rows = _load_rows(a.results)
    if a.agent:
        rows = [r for r in rows if r["agent"] in a.agent]
    if a.svg:
        Path(a.svg).write_text(report.svg(rows, a.title or ""))
    if a.md:
        Path(a.md).write_text(report.markdown(rows, a.svg if a.svg else None))
    if a.html:
        Path(a.html).write_text(report.html(rows, a.title or "rigcheck report"))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    print(report.terminal(rows))


def _cmd_compare(a: argparse.Namespace) -> None:
    ra, rb = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    na = ra.get("label") or f"{ra['endpoint']['engine']} / {ra['endpoint']['model']}"
    nb = rb.get("label") or f"{rb['endpoint']['engine']} / {rb['endpoint']['model']}"
    print(f"A = {na}\nB = {nb}\n")
    for agent in sorted(set(ra["agents"]) & set(rb["agents"])):
        pa = {(r["probe"], r["instance"], r["trial"]): r["pass"] for r in ra["agents"][agent].get("probes", [])}
        pb = {(r["probe"], r["instance"], r["trial"]): r["pass"] for r in rb["agents"][agent].get("probes", [])}
        keys = sorted(set(pa) & set(pb))
        pairs = [(pa[k], pb[k]) for k in keys]
        print(f"{agent}:")
        if pairs:
            b = sum(x and not y for x, y in pairs)
            c = sum(y and not x for x, y in pairs)
            est, lo, hi = stats.paired_diff(pairs)
            p = stats.mcnemar_exact(b, c)
            print(
                f"  tool calls: A - B = {100 * est:+.0f} points [{100 * lo:+.0f}, {100 * hi:+.0f}] over {len(pairs)} matched trials;"
                f" A-only passes {b}, B-only {c}, McNemar p = {p:.3g}"
            )
        for label, turn_ok in (("cold turn", lambda t: t == 1), ("warm turns", lambda t: t > 1)):
            ga, gb = (
                [
                    [t["ttft"] for t in r["agents"][agent].get("turns", []) if t["session"] == s and turn_ok(t["turn"]) and t["ttft"]]
                    for s in {t["session"] for t in r["agents"][agent].get("turns", [])}
                ]
                for r in (ra, rb)
            )
            est, lo, hi = stats.ratio_of_medians(ga, gb)
            if est == est:
                print(f"  {label} first token: A/B = {est:.2f}x [{lo:.2f}, {hi:.2f}]")


def _cmd_capture(a: argparse.Namespace) -> None:
    capture.run(Path(a.out), a.host, a.port)


def _cmd_profile(a: argparse.Namespace) -> None:
    rec = json.loads(Path(a.request).read_text())
    replace = dict(r.split("=", 1) for r in a.replace or [])
    meta = {"title": a.title or a.name, "version": a.agent_version or "", "license": a.license or "", "source": a.source or ""}
    prof = capture.to_profile(rec, a.name, a.task, replace, meta)
    Path(a.output).write_text(json.dumps(prof, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.output}: {len(prof['tools'])} tools, {len(json.dumps(prof['messages']))} chars of messages")
    missing = {"read", "shell", "edit", "write", "search"} - set(prof["canned"])
    if missing:
        print(f'no tool found for: {", ".join(sorted(missing))}; add them under "canned" before running multi-turn probes')


def _cmd_agents(a: argparse.Namespace) -> None:
    names = [a.name] if a.name else profiles.bundled()
    for n in names:
        p = profiles.load(n)
        tools = ", ".join(sorted(p.tool_names))
        print(f"{p.name:<10} {p.title} {p.version} ({p.license}), ~{p.prompt_chars() // 1000}k chars of prompt and tools")
        if a.name:
            print(f"  source: {p.source}\n  tools: {tools}\n  {p.notes}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rigcheck", description="Test a local model + engine + coding agent setup: tool calls and turn latency, with intervals.")
    ap.add_argument("--version", action="version", version=f"rigcheck {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="probe an OpenAI-compatible endpoint")
    r.add_argument("url", help="base URL, e.g. http://localhost:8080/v1")
    r.add_argument("--model", help="model id (default: first from /v1/models)")
    r.add_argument("--agent", action="append", help=f"agent profile or profile .json, repeatable (bundled: {', '.join(profiles.bundled())}; default opencode)")
    r.add_argument("--probe", action="append", help=f"run only these probes (default all: {', '.join(PROBES)})")
    r.add_argument("--trials", type=int, default=3, help="trials per probe (default 3)")
    r.add_argument("--sessions", type=int, default=2, help="timed 5-turn sessions per agent (default 2, 0 to skip)")
    r.add_argument("--no-stream", action="store_true", help="send probes without streaming")
    r.add_argument("--temperature", type=float)
    r.add_argument("--top-p", type=float)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--max-tokens", type=int, default=4096, help="output cap for probe replies (default 4096)")
    r.add_argument("--turn-tokens", type=int, default=256, help="output cap for timed turns (default 256)")
    r.add_argument("--concurrency", type=int, default=1, help="probe requests in flight (default 1)")
    r.add_argument("--extra", help='JSON merged into every request, e.g. \'{"reasoning_effort": "low"}\'')
    r.add_argument("--api-key")
    r.add_argument("--timeout", type=float, default=900.0, help="seconds per request (default 900)")
    r.add_argument("--label", help="name for this setup in reports")
    r.add_argument("-o", "--output", help="raw results JSON path")
    r.add_argument("--md", help="also write a Markdown report")
    r.add_argument("--html", help="also write an HTML report")
    r.set_defaults(fn=_cmd_run)

    p = sub.add_parser("report", help="summarize one or more results files")
    p.add_argument("results", nargs="+")
    p.add_argument("--agent", action="append", help="only these agents")
    p.add_argument("--md")
    p.add_argument("--html")
    p.add_argument("--svg")
    p.add_argument("--json", help="write the summary rows as JSON")
    p.add_argument("--title")
    p.set_defaults(fn=_cmd_report)

    c = sub.add_parser("compare", help="paired comparison of two results files")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=_cmd_compare)

    cp = sub.add_parser("capture", help="record the requests an agent sends (to build a profile)")
    cp.add_argument("--host", default="127.0.0.1")
    cp.add_argument("--port", type=int, default=9909)
    cp.add_argument("--out", default="captured")
    cp.set_defaults(fn=_cmd_capture)

    pr = sub.add_parser("profile", help="turn a captured request into an agent profile")
    pr.add_argument("request", help="a request-NNN.json written by capture")
    pr.add_argument("--name", required=True)
    pr.add_argument("--task", required=True, help="the exact prompt you gave the agent while capturing")
    pr.add_argument("--replace", action="append", help="OLD=NEW text replacement, e.g. your home path")
    pr.add_argument("--title")
    pr.add_argument("--agent-version")
    pr.add_argument("--license")
    pr.add_argument("--source")
    pr.add_argument("-o", "--output", required=True)
    pr.set_defaults(fn=_cmd_profile)

    ag = sub.add_parser("agents", help="list bundled agent profiles")
    ag.add_argument("name", nargs="?")
    ag.set_defaults(fn=_cmd_agents)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
