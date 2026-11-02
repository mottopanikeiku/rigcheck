"""Tool-call probes. Each probe is a short agent situation with a known right move.

A trial passes when the response clears every hard check (HTTP ok, no tool-call markup
leaked into the text, arguments are a JSON object that validates against the tool's
schema, the tool exists, output not cut off) and makes the expected move."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import jsonschema

from . import project
from .client import Completion, ToolCall
from .profiles import PROJECT, Action, Profile

HARD_FAILURES = ("http_error", "length", "markup_in_content", "call_as_text", "bad_json", "unknown_tool", "schema_violation")

_MARKUP = re.compile(
    r"<tool_call>|</tool_call>|<function=|</function>|<parameter=|<\|?tool_call|<tool_call\||\[TOOL_CALLS\]"
    r"|<\|channel\|>|<\|call\|>|<\|start\|>|<\|end\|>|<\|im_start\|>|<\|im_end\|>|<\|python_tag\|>|<\|tool_calls_section_begin\|>"
    r"|<start_function_call>|<｜tool▁call|<tool▁call|<\|tool_code\|>|<\|tool_call_start\|>|\bto=functions\.\w+|<\|\"\|>"
)
_THINK = re.compile(r"</?think>")


@dataclass
class Reply:
    """A reference answer: what a correct model would send. The mock server uses it."""

    content: str = ""
    calls: list[tuple[str, dict]] = field(default_factory=list)  # (action kind, fields)


@dataclass
class Probe:
    id: str
    checks: str
    instances: list[dict]
    build: Callable[[Profile, dict], tuple[str, list[dict], dict]]
    expect: Callable[[list[Action], Completion, dict], str | None]
    oracle: Callable[[dict], Reply]


# ---------------------------------------------------------------- history helpers


def _tool_turn(profile: Profile, kind: str, call_id: str, result: str, **fields) -> list[dict]:
    if kind == "shell":
        fields.setdefault("description", "Run command")
    return [
        {"role": "assistant", "content": "", "tool_calls": [profile.call(kind, call_id, **fields)]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _any(actions: list[Action], pred) -> bool:
    return any(pred(a) for a in actions)


EXPLORED = "explored"  # only looked around first; the probe's move was never tested


def _explored(actions: list[Action]) -> bool:
    return bool(actions) and all(a.kind in ("read", "list", "search") for a in actions)


# ---------------------------------------------------------------- probes


def _read_first() -> Probe:
    inst = [
        {"q": "What is RETRY_LIMIT set to in src/app/settings.py?", "file": "src/app/settings.py", "symbol": "RETRY_LIMIT"},
        {"q": "Which URL path does get_invoice() in src/app/http_client.py request?", "file": "src/app/http_client.py", "symbol": "get_invoice"},
        {"q": "What maxsize does the cache in src/app/cache.py use?", "file": "src/app/cache.py", "symbol": "lru_cache"},
    ]

    def expect(actions, c, i):
        if not actions:
            return "no_call"
        stem = i["file"].rsplit("/", 1)[-1].split(".")[0]
        if _any(actions, lambda a: a.touches(i["file"]) or i["symbol"] in (a.pattern + a.command)):
            return None
        if _any(actions, lambda a: a.kind in ("list", "search") and stem in a.pattern + " ".join(a.paths) + a.command):
            return None  # looking for the file is the right move too
        return EXPLORED if _explored(actions) else "wrong_action"

    return Probe(
        "read_first",
        "reads the file instead of guessing",
        inst,
        lambda p, i: (i["q"], [], {}),
        expect,
        lambda i: Reply(calls=[("read", {"path": i["file"]})]),
    )


def _run_command() -> Probe:
    inst = [
        {
            "q": "Run the parser tests (tests/test_parser.py) with pytest and tell me if they pass.",
            "needles": ["pytest", "test_parser"],
            "cmd": "pytest -q tests/test_parser.py",
        },
        {"q": "Run git status and tell me what changed.", "needles": ["git status"], "cmd": "git status"},
        {"q": "Show me the last three commits.", "needles": ["git log"], "cmd": "git log -3 --oneline"},
    ]

    def expect(actions, c, i):
        if not actions:
            return "no_call"
        if _any(actions, lambda a: a.command and all(n in a.command for n in i["needles"])):
            return None
        return "wrong_action"

    return Probe(
        "run_command",
        "runs the shell command it was asked to",
        inst,
        lambda p, i: (i["q"], [], {}),
        expect,
        lambda i: Reply(calls=[("shell", {"command": i["cmd"], "description": "Run command"})]),
    )


def _search() -> Probe:
    inst = [
        {"q": "Find where parse_header is defined.", "symbol": "parse_header", "file": "src/app/parser.py"},
        {"q": "Which files use REQUEST_TIMEOUT?", "symbol": "REQUEST_TIMEOUT", "file": "src/app/settings.py"},
        {"q": "Where do we import requests?", "symbol": "requests", "file": "src/app/http_client.py"},
    ]

    def expect(actions, c, i):
        if not actions:
            return "no_call"
        if _any(actions, lambda a: i["symbol"] in (a.pattern + a.command) or (a.kind == "read" and a.touches(i["file"]))):
            return None
        return EXPLORED if _explored(actions) else "wrong_action"

    return Probe(
        "search",
        "searches the code for a symbol",
        inst,
        lambda p, i: (i["q"], [], {}),
        expect,
        lambda i: Reply(calls=[("search", {"pattern": i["symbol"]})]),
    )


def _edit_after_read() -> Probe:
    inst = [
        {
            "q": "average() in src/app/mathutil.py returns the wrong value. Fix it.",
            "file": "src/app/mathutil.py",
            "old": "return sum(xs) / (len(xs) - 1)",
            "new": "return sum(xs) / len(xs)",
            "bad": "len(xs) - 1",
            "good": "/ len(xs)",
        },
        {
            "q": "Lower RETRY_LIMIT in src/app/settings.py to 3.",
            "file": "src/app/settings.py",
            "old": "RETRY_LIMIT = 5",
            "new": "RETRY_LIMIT = 3",
            "bad": "RETRY_LIMIT = 5",
            "good": "RETRY_LIMIT = 3",
        },
        {
            "q": "parse_header in src/app/parser.py should keep the key's original case. Fix it.",
            "file": "src/app/parser.py",
            "old": "return key.strip().lower(), value.strip()",
            "new": "return key.strip(), value.strip()",
            "bad": ".lower()",
            "good": "key.strip()",
        },
    ]

    def build(p: Profile, i):
        return i["q"], _tool_turn(p, "read", "call_read_1", project.numbered(i["file"]), path=i["file"]), {}

    def edited(a: Action, i) -> bool:
        if not a.touches(i["file"]):
            return False
        if a.kind == "edit":
            return i["good"] in a.new and i["bad"] not in a.new and (i["bad"] in a.old or not a.old)
        if a.kind == "write":
            return i["good"] in a.content and i["bad"] not in a.content
        if a.kind == "shell":  # sed -i, perl -pi, python -c ...
            return i["good"].strip("/ ") in a.command and re.search(r"sed -i|perl -pi|python", a.command) is not None
        return False

    def expect(actions, c, i):
        if not actions:
            return "no_call"
        return None if _any(actions, lambda a: edited(a, i)) else "no_correct_edit"

    return Probe(
        "edit_after_read",
        "uses a file it just read to make the right edit",
        inst,
        build,
        expect,
        lambda i: Reply(calls=[("edit", {"path": i["file"], "old": i["old"], "new": i["new"]})]),
    )


def _stop_when_done() -> Probe:
    inst = [
        {
            "q": "Run the tests.",
            "kind": "shell",
            "fields": {"command": "pytest -q"},
            "result": "..                                                                       [100%]\n2 passed in 0.03s",
            "answer": "Both tests pass.",
        },
        {
            "q": "What does src/app/utils.py do?",
            "kind": "read",
            "fields": {"path": "src/app/utils.py"},
            "result": project.numbered("src/app/utils.py"),
            "answer": "It defines chunks(), a generator that yields slices of n items.",
        },
        {
            "q": "Is the working tree clean?",
            "kind": "shell",
            "fields": {"command": "git status"},
            "result": "On branch main\nnothing to commit, working tree clean",
            "answer": "Yes, nothing to commit.",
        },
    ]

    def build(p: Profile, i):
        return i["q"], _tool_turn(p, i["kind"], "call_1", i["result"], **i["fields"]), {}

    def expect(actions, c, i):
        if actions:
            return "unneeded_call"
        return None if c.content.strip() else "empty_reply"

    return Probe(
        "stop_when_done",
        "answers once the tool result is in, without another call",
        inst,
        build,
        expect,
        lambda i: Reply(content=i["answer"]),
    )


def _no_tool() -> Probe:
    inst = [
        {"q": "In one sentence, what does HTTP status 404 mean?", "answer": "The server could not find the requested resource."},
        {"q": "In one sentence, what is the difference between a Python list and a tuple?", "answer": "Lists are mutable, tuples are not."},
        {"q": "Thanks, that's all for now.", "answer": "You're welcome."},
    ]

    def expect(actions, c, i):
        if actions:
            return "unneeded_call"
        return None if c.content.strip() else "empty_reply"

    return Probe("no_tool", "answers a plain question without calling tools", inst, lambda p, i: (i["q"], [], {}), expect, lambda i: Reply(content=i["answer"]))


def _parallel() -> Probe:
    inst = [
        {"q": "Read src/app/parser.py and src/app/utils.py, then tell me which one defines a generator.", "files": ["src/app/parser.py", "src/app/utils.py"]},
        {"q": "Show me both test files: tests/test_parser.py and tests/test_mathutil.py.", "files": ["tests/test_parser.py", "tests/test_mathutil.py"]},
        {
            "q": "Open src/app/settings.py and src/app/http_client.py. Does the client use the timeout setting?",
            "files": ["src/app/settings.py", "src/app/http_client.py"],
        },
    ]

    def expect(actions, c, i):
        if not actions:
            return "no_call"
        if all(_any(actions, lambda a, f=f: a.touches(f)) for f in i["files"]):
            return None
        return "one_at_a_time"

    return Probe(
        "parallel",
        "reads two files in one turn (parallel_tool_calls on)",
        inst,
        lambda p, i: (i["q"], [], {"parallel_tool_calls": True}),
        expect,
        lambda i: Reply(calls=[("read", {"path": f}) for f in i["files"]]),
    )


def _recover() -> Probe:
    inst = [
        {"q": "Show me src/app/util.py.", "bad": "src/app/util.py", "good": "src/app/utils.py"},
        {"q": "Open src/app/settings.yaml.", "bad": "src/app/settings.yaml", "good": "src/app/settings.py"},
        {"q": "Read the README file.", "bad": "README", "good": "README.md"},
    ]

    def build(p: Profile, i):
        err = f"Error: File not found: {PROJECT}/{i['bad']}"
        return i["q"], _tool_turn(p, "read", "call_read_1", err, path=i["bad"]), {}

    def expect(actions, c, i):
        if not actions:
            return "gave_up"
        if _any(actions, lambda a: a.touches(i["good"]) or a.kind in ("list", "search")):
            return None
        return "repeated_or_wrong"

    return Probe(
        "recover",
        "looks for the right file after a not-found error",
        inst,
        build,
        expect,
        lambda i: Reply(calls=[("read", {"path": i["good"]})]),
    )


_REPORT_PY = """import argparse
import csv
from collections import defaultdict
from decimal import Decimal

from app.cache import cached_rate


def main():
    parser = argparse.ArgumentParser(description="Per-customer invoice totals in EUR.")
    parser.add_argument("path", help="CSV file with id, customer, amount, currency")
    args = parser.parse_args()
    totals = defaultdict(Decimal)
    with open(args.path, newline="") as f:
        for row in csv.DictReader(f):
            totals[row["customer"]] += Decimal(row["amount"]) / Decimal(str(cached_rate(row["currency"])))
    for customer, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{customer:<30} {total:>12.2f}")


if __name__ == "__main__":
    main()
"""


def _long_write() -> Probe:
    inst = [
        {
            "q": "Create scripts/report.py: a command-line script that reads a CSV of invoices (columns: id, customer, amount, currency) "
            "from a path given as the first argument, converts each amount to EUR with app.cache.cached_rate, and prints a "
            "per-customer total, sorted by total, descending. Use argparse and the csv module, and put the logic in main().",
            "context": "src/app/cache.py",
            "file": "scripts/report.py",
            "needles": ["argparse", "csv", "def main"],
            "content": _REPORT_PY,
        },
        {
            "q": "Create tests/test_cache.py with pytest tests for app.cache.cached_rate: one test per currency (EUR, USD, GBP) "
            "and one that checks an unknown currency raises KeyError.",
            "context": "src/app/cache.py",
            "file": "tests/test_cache.py",
            "needles": ["def test_", "KeyError"],
            "content": 'import pytest\n\nfrom app.cache import cached_rate\n\n\ndef test_eur():\n    assert cached_rate("EUR") == 1.0\n\n\n'
            'def test_usd():\n    assert cached_rate("USD") == 1.08\n\n\ndef test_gbp():\n    assert cached_rate("GBP") == 0.86\n\n\n'
            'def test_unknown():\n    with pytest.raises(KeyError):\n        cached_rate("JPY")\n',
        },
        {
            "q": "Create src/app/retry.py with a retry(func, attempts, delay) helper that calls func, catches exceptions, sleeps "
            "`delay` seconds between attempts with time.sleep, and re-raises the last exception after the final attempt. "
            "Include a docstring and type hints.",
            "context": "src/app/http_client.py",
            "file": "src/app/retry.py",
            "needles": ["def retry", "time.sleep", "raise"],
            "content": 'import time\nfrom typing import Callable, TypeVar\n\nT = TypeVar("T")\n\n\ndef retry(func: Callable[[], T], attempts: int, delay: float) -> T:\n'
            '    """Call func up to `attempts` times, sleeping `delay` seconds between tries."""\n    last: Exception | None = None\n'
            "    for i in range(attempts):\n        try:\n            return func()\n        except Exception as e:\n            last = e\n"
            "            if i < attempts - 1:\n                time.sleep(delay)\n    assert last is not None\n    raise last\n",
        },
    ]

    def build(p: Profile, i):
        return i["q"], _tool_turn(p, "read", "call_read_1", project.numbered(i["context"]), path=i["context"]), {}

    def wrote(a: Action, i) -> bool:
        if not a.touches(i["file"]):
            return False
        body = a.content or a.new or a.command
        return all(n in body for n in i["needles"]) and body.count("\n") >= 6

    def expect(actions, c, i):
        if not actions:
            return "no_call"
        if _any(actions, lambda a: wrote(a, i)):
            return None
        return EXPLORED if _explored(actions) else "no_complete_file"

    return Probe(
        "long_write",
        "writes a whole new file in one long string argument, after reading what it needs",
        inst,
        build,
        expect,
        lambda i: Reply(calls=[("write", {"path": i["file"], "content": i["content"]})]),
    )


PROBES: dict[str, Probe] = {
    p.id: p for p in (_read_first(), _run_command(), _search(), _edit_after_read(), _stop_when_done(), _no_tool(), _parallel(), _recover(), _long_write())
}


# ---------------------------------------------------------------- checks


def hard_checks(c: Completion, profile: Profile) -> tuple[str | None, list[Action], list[dict]]:
    """Return (failure reason or None, parsed actions, per-call details)."""
    if not c.ok:
        return "http_error", [], []
    details = []
    actions = []
    reason = None
    if _MARKUP.search(c.content):
        reason = "markup_in_content"
    elif _call_in_text(c.content, profile):
        reason = "call_as_text"
    for tc in c.tool_calls:
        d = {"name": tc.name, "arguments": tc.arguments[:2000]}
        try:
            args = json.loads(tc.arguments) if tc.arguments.strip() else {}
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            d["error"] = "bad_json"
            reason = reason or "bad_json"
            details.append(d)
            continue
        if tc.name not in profile.tool_names:
            d["error"] = "unknown_tool"
            reason = reason or "unknown_tool"
        else:
            schema = profile.schema(tc.name)
            err = _schema_error(schema, args)
            if err:
                d["error"] = f"schema_violation: {err}"
                reason = reason or "schema_violation"
        a = profile.classify(tc.name, args)
        d["action"] = a.kind
        actions.append(a)
        details.append(d)
    if c.finish_reason == "length" and not c.tool_calls:
        reason = reason or "length"
    return reason, actions, details


def soft_warnings(c: Completion) -> list[str]:
    w = list(dict.fromkeys(c.stream_issues))
    if c.ok and _THINK.search(c.content):
        w.append("think_tags_in_content")
    if c.ok and c.tool_calls and c.finish_reason not in ("tool_calls", "function_call"):
        w.append(f"finish_reason={c.finish_reason}")
    return w


def _schema_error(schema: dict | None, args: dict) -> str | None:
    if not schema:
        return None
    try:
        cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
        err = jsonschema.exceptions.best_match(cls(schema).iter_errors(args))
    except Exception:  # a broken or unresolvable schema leaves nothing to check against
        return None
    return err.message[:200] if err else None


_JSON_CALL = re.compile(r'\{\s*"(?:name|function)"\s*:\s*"([\w.\-]+)"')


def _call_in_text(text: str, profile: Profile) -> bool:
    """A tool call written into the reply as text instead of sent as a structured call:
    JSON ({"name": "read", ...}) or ad-hoc syntax at the start of a line or a bracket
    (read{...}, [read:src/a.py], grep(pattern: ...))."""
    if any(m.group(1) in profile.tool_names for m in _JSON_CALL.finditer(text)):
        return True
    names = "|".join(re.escape(n) for n in sorted(profile.tool_names, key=len, reverse=True))
    if not names:
        return False
    adhoc = re.compile(rf"(?m)(?:^|\[)\s*(?:{names})\s*(?:\{{|\(\s*\w+\s*[:=]|:\s*[\w./-]+\.\w+)")
    return adhoc.search(text) is not None


def completion_of(row: dict) -> Completion:
    """Rebuild a reply from a stored probe row, so results can be re-scored later."""
    calls = [ToolCall(t.get("index"), t.get("id"), t.get("name"), t.get("arguments") or "") for t in row.get("tool_calls") or []]
    return Completion(
        ok=row["ok"],
        status=row.get("status"),
        error=row.get("error"),
        content=row.get("content") or "",
        tool_calls=calls,
        finish_reason=row.get("finish_reason"),
        stream_issues=list(row.get("stream_issues") or []),
    )


def score(row: dict, profile: Profile) -> dict:
    probe = PROBES[row["probe"]]
    c = completion_of(row)
    reason, actions, calls = hard_checks(c, profile)
    if reason is None:
        reason = probe.expect(actions, c, probe.instances[row["instance"]])
    passed = None if reason == EXPLORED else reason is None
    return {"pass": passed, "reason": reason, "warnings": soft_warnings(c), "calls": calls}
