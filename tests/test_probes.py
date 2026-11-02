"""Every probe's reference answer must pass, in every bundled agent's own tool format,
and the usual ways of getting it wrong must fail for the right reason."""

import json

import pytest

from rigcheck import profiles
from rigcheck.client import Completion, ToolCall
from rigcheck.probes import PROBES, hard_checks

AGENTS = profiles.bundled()


def _completion(prof, reply, **kw):
    calls = [prof.call(kind, f"c{i}", **fields) for i, (kind, fields) in enumerate(reply.calls)]
    tcs = [ToolCall(index=i, id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"]) for i, c in enumerate(calls)]
    return Completion(ok=True, content=reply.content, tool_calls=tcs, finish_reason="tool_calls" if tcs else "stop", **kw)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("probe_id", list(PROBES))
def test_reference_answers_pass(agent, probe_id):
    prof = profiles.load(agent)
    probe = PROBES[probe_id]
    for inst in probe.instances:
        task, history, _ = probe.build(prof, inst)
        msgs = prof.build_messages(task) + history
        assert any(task in json.dumps(m) for m in msgs)
        c = _completion(prof, probe.oracle(inst))
        reason, actions, details = hard_checks(c, prof)
        assert reason is None, (details, reason)
        assert probe.expect(actions, c, inst) is None


def test_markup_in_text_fails():
    prof = profiles.load("opencode")
    c = Completion(ok=True, content='<tool_call>\n{"name": "read", "arguments": {"filePath": "x"}}\n</tool_call>', finish_reason="stop")
    assert hard_checks(c, prof)[0] == "markup_in_content"
    c = Completion(ok=True, content='{"name": "read", "arguments": {"filePath": "/home/dev/project/x"}}', finish_reason="stop")
    assert hard_checks(c, prof)[0] == "markup_in_content"
    c = Completion(ok=True, content="<|channel|>commentary to=functions.read <|message|>{}", finish_reason="stop")
    assert hard_checks(c, prof)[0] == "markup_in_content"


def test_bad_json_unknown_tool_and_schema():
    prof = profiles.load("opencode")
    bad = Completion(ok=True, tool_calls=[ToolCall(0, "a", "read", '{"filePath": "x"')], finish_reason="tool_calls")
    assert hard_checks(bad, prof)[0] == "bad_json"
    unknown = Completion(ok=True, tool_calls=[ToolCall(0, "a", "read_file", '{"path": "x"}')], finish_reason="tool_calls")
    assert hard_checks(unknown, prof)[0] == "unknown_tool"
    schema = Completion(ok=True, tool_calls=[ToolCall(0, "a", "read", '{"path": "x"}')], finish_reason="tool_calls")
    assert hard_checks(schema, prof)[0] == "schema_violation"


def test_answering_without_reading_fails():
    prof = profiles.load("qwen-code")
    probe = PROBES["read_first"]
    c = Completion(ok=True, content="RETRY_LIMIT is 3.", finish_reason="stop")
    reason, actions, _ = hard_checks(c, prof)
    assert reason is None
    assert probe.expect(actions, c, probe.instances[0]) == "no_call"


def test_codex_shell_reads_and_patches_are_understood():
    prof = profiles.load("codex")
    a = prof.classify("exec_command", {"cmd": "sed -n '1,80p' src/app/settings.py"})
    assert a.kind == "read" and a.touches("src/app/settings.py")
    a = prof.classify("exec_command", {"cmd": "rg -n parse_header"})
    assert a.kind == "search" and "parse_header" in a.command
    patch = "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: src/app/mathutil.py\n@@\n-    return sum(xs) / (len(xs) - 1)\n+    return sum(xs) / len(xs)\n*** End Patch\nEOF"
    a = prof.classify("exec_command", {"cmd": patch})
    assert a.kind == "edit" and a.touches("src/app/mathutil.py") and "len(xs) - 1" in a.old and "- 1" not in a.new


def test_qwen_code_dispatcher_tool_is_unwrapped():
    prof = profiles.load("qwen-code")
    a = prof.classify("tool_call", {"name": "read_file", "arguments": {"file_path": "/home/dev/project/src/app/cache.py"}})
    assert a.kind == "read" and a.touches("src/app/cache.py")


def test_nonce_makes_prompt_unique():
    prof = profiles.load("opencode")
    a = prof.build_messages("hi", nonce="aaaa")
    b = prof.build_messages("hi", nonce="bbbb")
    assert a != b and a[1:] == b[1:]


def test_parallel_globs_for_both_files_count():
    prof = profiles.load("opencode")
    probe = PROBES["parallel"]
    inst = probe.instances[1]
    calls = [ToolCall(i, f"c{i}", "glob", json.dumps({"pattern": f"**/{f.rsplit('/', 1)[-1]}"})) for i, f in enumerate(inst["files"])]
    c = Completion(ok=True, tool_calls=calls, finish_reason="tool_calls")
    reason, actions, _ = hard_checks(c, prof)
    assert reason is None and probe.expect(actions, c, inst) is None
    assert probe.expect(actions[:1], c, inst) == "one_at_a_time"
