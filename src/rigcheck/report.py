"""Summaries and renderings: terminal table, Markdown, HTML and an SVG chart."""

from __future__ import annotations

import html as _html
import math
from collections import Counter, defaultdict

from . import stats, turns
from .probes import PROBES


def _model_name(res: dict) -> str:
    ep = res["endpoint"]
    return ep.get("model_path") or ep.get("model") or "?"


def summarize(res: dict) -> list[dict]:
    rows = []
    ep = res["endpoint"]
    for name, a in res["agents"].items():
        probes = a.get("probes", [])
        k = sum(r["pass"] for r in probes)
        n = len(probes)
        lo, hi = stats.wilson(k, n)
        by_probe: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in probes:
            by_probe[r["probe"]][0] += r["pass"]
            by_probe[r["probe"]][1] += 1
        reasons = Counter(r["reason"] for r in probes if not r["pass"])
        warnings = Counter(w.split(":")[0] for r in probes for w in r.get("warnings", []))
        sessions: dict[int, list[dict]] = defaultdict(list)
        for t in a.get("turns", []):
            if t["ok"] and t["ttft"] is not None:
                sessions[t["session"]].append(t)
        cold = [[t["ttft"] for t in s if t["turn"] == 1] for s in sessions.values()]
        warm = [[t["ttft"] for t in s if t["turn"] > 1] for s in sessions.values()]
        dec = [[t["decode_tps"] for t in s if t["decode_tps"] and (t["completion_tokens"] or 0) >= 16] for s in sessions.values()]
        session_wait = [[sum(t["ttft"] for t in s)] for s in sessions.values() if len(s) >= 2]
        reuse = [turns.prefix_reuse(s) for s in sessions.values()]
        cache_share = [
            t["engine_cache_n"] / (t["engine_cache_n"] + t["engine_prompt_n"])
            for s in sessions.values()
            for t in s
            if t["turn"] > 1
            and t.get("engine_cache_n") is not None
            and t.get("engine_prompt_n") is not None
            and (t["engine_cache_n"] + t["engine_prompt_n"]) > 0
        ]
        turn_prompt = [t["prompt_tokens"] for s in sessions.values() for t in s if t["prompt_tokens"]]
        rows.append(
            {
                "label": res.get("label"),
                "engine": ep.get("engine"),
                "engine_version": ep.get("version"),
                "model": _model_name(res),
                "agent": name,
                "agent_title": a["profile"]["title"],
                "fit": a.get("fit", {}).get("verdict"),
                "prompt_tokens": a.get("fit", {}).get("prompt_tokens"),
                "context": ep.get("context_loaded") or a.get("fit", {}).get("context"),
                "pass": k,
                "n": n,
                "rate": k / n if n else math.nan,
                "lo": lo,
                "hi": hi,
                "by_probe": dict(by_probe),
                "reasons": dict(reasons.most_common()),
                "warnings": dict(warnings.most_common()),
                "sessions": len(sessions),
                "cold_ttft": stats.bootstrap(cold),
                "warm_ttft": stats.bootstrap(warm),
                "warm_p90": stats.bootstrap(warm, stats.p90),
                "warm_over_cold": stats.ratio_of_medians(warm, cold),
                "decode_tps": stats.bootstrap(dec),
                "session_wait": stats.bootstrap(session_wait),
                "prefix_reuse": stats.bootstrap(reuse),
                "engine_cache_share": stats.quantile(cache_share, 0.5) if cache_share else None,
                "max_turn_prompt": max(turn_prompt) if turn_prompt else None,
            }
        )
    return rows


# ---------------------------------------------------------------- formatting helpers


def pct(x: float) -> str:
    return "-" if x is None or math.isnan(x) else f"{100 * x:.0f}%"


def secs(x: float) -> str:
    if x is None or math.isnan(x):
        return "-"
    return f"{x:.2f}s" if x < 10 else f"{x:.0f}s" if x >= 100 else f"{x:.1f}s"


def ktok(n: int | None) -> str:
    return "-" if not n else f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def interval(t: tuple[float, float, float], fmt=secs) -> str:
    est, lo, hi = t
    if est is None or math.isnan(est):
        return "-"
    if lo is None or math.isnan(lo):
        return fmt(est)
    return f"{fmt(est)} [{fmt(lo)}, {fmt(hi)}]"


def rate_ci(r: dict) -> str:
    if not r["n"]:
        return "-"
    return f"{pct(r['rate'])} [{pct(r['lo'])}, {pct(r['hi'])}] n={r['n']}"


def tps(x: float) -> str:
    return "-" if x is None or math.isnan(x) else f"{x:.1f}"


def setup_name(r: dict) -> str:
    return r.get("label") or f"{r['engine']} / {r['model']}"


# ---------------------------------------------------------------- terminal


def terminal(rows: list[dict]) -> str:
    head = ["setup", "agent", "prompt", "tool calls (95% CI)", "cold turn", "warm turn", "prefix reuse", "decode tok/s"]
    body = []
    for r in rows:
        fit = ktok(r["prompt_tokens"]) + ("" if r["fit"] == "fits" else f" {str(r['fit']).upper()}")
        body.append(
            [
                setup_name(r),
                r["agent"],
                fit,
                rate_ci(r),
                interval(r["cold_ttft"]),
                interval(r["warm_ttft"]),
                interval(r["prefix_reuse"], pct),
                interval(r["decode_tps"], tps),
            ]
        )
    widths = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    lines = ["  ".join(str(x).ljust(w) for x, w in zip(head, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(str(x).ljust(w) for x, w in zip(row, widths)) for row in body]
    for r in rows:
        if r["reasons"] or r["warnings"]:
            fails = ", ".join(f"{k} {v}" for k, v in r["reasons"].items()) or "none"
            warns = ", ".join(f"{k} {v}" for k, v in r["warnings"].items()) or "none"
            lines.append(f"{setup_name(r)} / {r['agent']}: failures: {fails}; warnings: {warns}")
    return "\n".join(lines)


# ---------------------------------------------------------------- markdown


def markdown(rows: list[dict], svg_path: str | None = None) -> str:
    out = ["# rigcheck report", ""]
    if svg_path:
        out += [f"![results]({svg_path})", ""]
    out += [
        "| Setup | Agent | Prompt | Tool calls pass (95% CI) | Cold turn | Warm turn | Prefix reuse | Decode tok/s |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        fit = ktok(r["prompt_tokens"]) + ("" if r["fit"] == "fits" else f" **{r['fit']}**")
        out.append(
            f"| {setup_name(r)} | {r['agent']} | {fit} | {rate_ci(r)} | {interval(r['cold_ttft'])} | {interval(r['warm_ttft'])} "
            f"| {interval(r['prefix_reuse'], pct)} | {interval(r['decode_tps'], tps)} |"
        )
    probe_ids = [p for p in PROBES if any(p in r["by_probe"] for r in rows)]
    if probe_ids:
        out += ["", "## Pass rate by probe", ""]
        out.append("| Setup | Agent | " + " | ".join(probe_ids) + " |")
        out.append("|---|---|" + "---:|" * len(probe_ids))
        for r in rows:
            cells = []
            for p in probe_ids:
                k, n = r["by_probe"].get(p, (0, 0))
                cells.append(f"{k}/{n}" if n else "-")
            out.append(f"| {setup_name(r)} | {r['agent']} | " + " | ".join(cells) + " |")
        out += ["", "Probes: " + "; ".join(f"`{p}` {PROBES[p].checks}" for p in probe_ids) + "."]
    fails = [r for r in rows if r["reasons"] or r["warnings"]]
    if fails:
        out += ["", "## Failures and warnings", ""]
        for r in fails:
            f = ", ".join(f"{k} ×{v}" for k, v in r["reasons"].items()) or "none"
            w = ", ".join(f"{k} ×{v}" for k, v in r["warnings"].items()) or "none"
            out.append(f"- {setup_name(r)} / {r['agent']}: failures {f}; warnings {w}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- svg


PALETTE = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#be123c"]


def svg(rows: list[dict], title: str = "") -> str:
    """Two panels per setup row: tool-call pass rate with its Wilson interval, and
    first-token latency for the cold turn vs warm turns (log scale)."""
    rows = [r for r in rows if r["n"] or r["sessions"]]
    engines = list(dict.fromkeys(r["engine"] for r in rows))
    color = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(engines)}
    row_h, top, left = 30, 62 if title else 44, 300
    pw, gap = 300, 60
    width = left + pw + gap + pw + 30
    height = top + row_h * len(rows) + 46
    x1 = left
    x2 = left + pw + gap
    times = [v for r in rows for v in (r["cold_ttft"][0], r["warm_ttft"][0], r["cold_ttft"][2], r["warm_ttft"][2]) if v and not math.isnan(v)]
    tmin = 10 ** math.floor(math.log10(max(min(times), 1e-3))) if times else 0.1
    tmax = 10 ** math.ceil(math.log10(max(times))) if times else 100

    def sx(p):
        return x1 + p * pw

    def tx(t):
        t = min(max(t, tmin), tmax)
        return x2 + (math.log10(t) - math.log10(tmin)) / (math.log10(tmax) - math.log10(tmin)) * pw

    e = []
    e.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">'
    )
    e.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    if title:
        e.append(f'<text x="16" y="24" font-size="15" font-weight="600" fill="#111827">{_html.escape(title)}</text>')
    hy = top - 16
    e.append(f'<text x="{x1}" y="{hy}" fill="#374151" font-weight="600">Tool-call probes passed (95% CI)</text>')
    e.append(f'<text x="{x2}" y="{hy}" fill="#374151" font-weight="600">First token: cold turn <tspan fill="#9ca3af">○</tspan> vs warm turns ●</text>')
    for p in (0, 0.25, 0.5, 0.75, 1.0):
        x = sx(p)
        e.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + row_h * len(rows)}" stroke="#e5e7eb"/>')
        e.append(f'<text x="{x:.1f}" y="{top + row_h * len(rows) + 16}" text-anchor="middle" fill="#6b7280">{int(p * 100)}%</text>')
    t = tmin
    while t <= tmax * 1.0001:
        x = tx(t)
        e.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + row_h * len(rows)}" stroke="#e5e7eb"/>')
        lab = f"{t:g}s"
        e.append(f'<text x="{x:.1f}" y="{top + row_h * len(rows) + 16}" text-anchor="middle" fill="#6b7280">{lab}</text>')
        t *= 10
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        c = color[r["engine"]]
        name = setup_name(r) + (f" · {r['agent']}" if len({x["agent"] for x in rows}) > 1 else "")
        e.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" fill="#111827">{_html.escape(name)}</text>')
        if r["n"]:
            e.append(f'<line x1="{sx(r["lo"]):.1f}" y1="{y:.1f}" x2="{sx(r["hi"]):.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2"/>')
            e.append(f'<circle cx="{sx(r["rate"]):.1f}" cy="{y:.1f}" r="5" fill="{c}"/>')
        ce, cl, ch = r["cold_ttft"]
        we, wl, wh = r["warm_ttft"]
        if ce and not math.isnan(ce):
            if cl and not math.isnan(cl):
                e.append(f'<line x1="{tx(cl):.1f}" y1="{y - 4:.1f}" x2="{tx(ch):.1f}" y2="{y - 4:.1f}" stroke="{c}" stroke-opacity="0.45" stroke-width="2"/>')
            e.append(f'<circle cx="{tx(ce):.1f}" cy="{y - 4:.1f}" r="5" fill="#ffffff" stroke="{c}" stroke-width="2"/>')
        if we and not math.isnan(we):
            if wl and not math.isnan(wl):
                e.append(f'<line x1="{tx(wl):.1f}" y1="{y + 4:.1f}" x2="{tx(wh):.1f}" y2="{y + 4:.1f}" stroke="{c}" stroke-width="2"/>')
            e.append(f'<circle cx="{tx(we):.1f}" cy="{y + 4:.1f}" r="5" fill="{c}"/>')
    ly = height - 12
    lx = 16
    for eng in engines:
        e.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{color[eng]}"/>')
        e.append(f'<text x="{lx + 14}" y="{ly}" fill="#374151">{_html.escape(str(eng))}</text>')
        lx += 22 + 7 * len(str(eng))
    e.append("</svg>")
    return "\n".join(e)


def html(rows: list[dict], title: str = "rigcheck report") -> str:
    md = markdown(rows)
    table_html = _md_tables_to_html(md)
    return (
        "<!doctype html><meta charset='utf-8'><title>" + _html.escape(title) + "</title>"
        "<style>body{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#111827}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #e5e7eb;padding:4px 8px;text-align:left}"
        "th{background:#f9fafb}code{background:#f3f4f6;padding:0 3px}</style>"
        f"<h1>{_html.escape(title)}</h1>" + svg(rows) + table_html
    )


def _md_tables_to_html(md: str) -> str:
    out, in_table = [], False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-:") for c in cells):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{_html.escape(c).replace('**', '')}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
            if line.startswith("## "):
                out.append(f"<h2>{_html.escape(line[3:])}</h2>")
            elif line.startswith("- "):
                out.append(f"<p>{_html.escape(line[2:])}</p>")
            elif line and not line.startswith("# ") and not line.startswith("!["):
                out.append(f"<p>{_html.escape(line)}</p>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)
