import json
import threading

import httpx

from rigcheck import capture, profiles


def test_capture_and_convert(tmp_path):
    server = capture.serve(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    body = {
        "model": "m",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are agent X. cwd: /home/someone/proj"},
            {"role": "system", "content": "Second system block."},
            {"role": "user", "content": "Fix the bug in a.py"},
        ],
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
        "parallel_tool_calls": True,
    }
    r = httpx.post(f"{url}/chat/completions", json=body)
    assert r.status_code == 200 and "data: [DONE]" in r.text
    r = httpx.post(f"{url}/responses", json={"model": "m", "input": []})
    assert r.json()["status"] == "completed"
    server.shutdown()

    rec = json.loads((tmp_path / "request-001.json").read_text())
    prof = capture.to_profile(rec, "x", "Fix the bug in a.py", {"/home/someone/proj": "/home/dev/project"})
    assert prof["messages"][0]["content"] == "You are agent X. cwd: /home/dev/project\n\nSecond system block."
    assert prof["messages"][-1] == {"role": "user", "content": "{task}"}
    assert prof["params"] == {"parallel_tool_calls": True}
    assert prof["canned"]["read"] == {"tool": "read_file", "args": {"path": "{abspath}"}}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(prof))
    p = profiles.load(str(path))
    assert p.build_messages("do it")[-1]["content"] == "do it"


def test_responses_api_requests_convert():
    rec = {
        "body": {
            "instructions": "base",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go now"}]},
            ],
            "tools": [
                {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
                {"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "spawn", "parameters": {"type": "object"}}]},
                {"type": "web_search"},
            ],
        }
    }
    prof = capture.to_profile(rec, "y", "go now", {})
    assert prof["messages"] == [{"role": "system", "content": "base\n\ndev"}, {"role": "user", "content": "{task}"}]
    assert [t["function"]["name"] for t in prof["tools"]] == ["exec_command", "ns__spawn"]
