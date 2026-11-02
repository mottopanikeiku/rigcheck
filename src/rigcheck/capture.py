"""Record the requests a coding agent sends, so its real prompt and tools can be replayed.

Point the agent at http://127.0.0.1:PORT/v1 and run one task. Every POST body is saved
as JSON. The server answers chat completions and Responses API calls with a short
"ok" so the agent finishes its turn cleanly.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPLY = "ok"


def _chat_reply(model: str, stream: bool) -> tuple[str, bytes]:
    created = int(time.time())
    if not stream:
        body = {
            "id": "chatcmpl-capture",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": REPLY},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return "application/json", json.dumps(body).encode()
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    ]
    out = []
    for c in chunks:
        c.update({"id": "chatcmpl-capture", "object": "chat.completion.chunk", "created": created, "model": model})
        out.append(f"data: {json.dumps(c)}\n\n")
    out.append("data: [DONE]\n\n")
    return "text/event-stream", "".join(out).encode()


def _responses_reply(model: str, stream: bool) -> tuple[str, bytes]:
    item = {
        "type": "message",
        "id": "msg_capture",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": REPLY, "annotations": []}],
    }
    response = {
        "id": "resp_capture",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [item],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    if not stream:
        return "application/json", json.dumps(response).encode()
    events = [
        ("response.created", {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}),
        ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": {**item, "status": "in_progress", "content": []}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_capture", "output_index": 0, "content_index": 0, "delta": REPLY}),
        ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
        ("response.completed", {"type": "response.completed", "response": response}),
    ]
    out = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return "text/event-stream", out.encode()


class _Recorder(BaseHTTPRequestHandler):
    out_dir: Path
    counter = 0
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, ctype: str, payload: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            body = {"object": "list", "data": [{"id": "capture", "object": "model", "owned_by": "rigcheck"}]}
            self._send("application/json", json.dumps(body).encode())
        else:
            self._send("application/json", b"{}")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        with _Recorder.lock:
            _Recorder.counter += 1
            n = _Recorder.counter
        record = {"path": self.path, "user_agent": self.headers.get("User-Agent"), "body": body}
        (self.out_dir / f"request-{n:03d}.json").write_text(json.dumps(record, indent=2))
        model = body.get("model", "capture") if isinstance(body, dict) else "capture"
        stream = bool(body.get("stream")) if isinstance(body, dict) else False
        if "/responses" in self.path:
            ctype, payload = _responses_reply(model, stream)
        else:
            ctype, payload = _chat_reply(model, stream)
        self._send(ctype, payload)


def serve(out_dir: Path, host: str = "127.0.0.1", port: int = 9909) -> ThreadingHTTPServer:
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = type("Recorder", (_Recorder,), {"out_dir": out_dir})
    return ThreadingHTTPServer((host, port), handler)


def run(out_dir: Path, host: str, port: int) -> None:
    server = serve(out_dir, host, port)
    print(f"recording to {out_dir}; point your agent at http://{host}:{port}/v1  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# Roles and canned calls for the bundled agents. For any other agent, roles are inferred
# from tool and argument names; check them with `rigcheck agents NAME`.
KNOWN = {
    "opencode": {
        "roles": {
            "read": {"kind": "read", "fields": {"filePath": "path"}},
            "bash": {"kind": "shell", "fields": {"command": "command", "workdir": "path"}},
            "edit": {"kind": "edit", "fields": {"filePath": "path", "oldString": "old", "newString": "new"}},
            "write": {"kind": "write", "fields": {"filePath": "path", "content": "content"}},
            "grep": {"kind": "search", "fields": {"pattern": "pattern", "path": "path", "include": "path"}},
            "glob": {"kind": "list", "fields": {"pattern": "pattern", "path": "path"}},
        },
        "canned": {
            "read": {"tool": "read", "args": {"filePath": "{abspath}"}},
            "shell": {"tool": "bash", "args": {"command": "{command}"}},
            "search": {"tool": "grep", "args": {"pattern": "{pattern}"}},
            "edit": {"tool": "edit", "args": {"filePath": "{abspath}", "oldString": "{old}", "newString": "{new}"}},
            "write": {"tool": "write", "args": {"filePath": "{abspath}", "content": "{content}"}},
        },
    },
    "codex": {
        "roles": {
            "exec_command": {"kind": "shell", "fields": {"cmd": "command"}},
            "view_image": {"kind": "read", "fields": {"path": "path"}},
        },
        "canned": {
            "read": {"tool": "exec_command", "args": {"cmd": "sed -n '1,200p' {path}"}},
            "shell": {"tool": "exec_command", "args": {"cmd": "{command}"}},
            "search": {"tool": "exec_command", "args": {"cmd": "rg -n '{pattern}'"}},
            "edit": {
                "tool": "exec_command",
                "args": {"cmd": "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: {path}\n@@\n-{old}\n+{new}\n*** End Patch\nEOF"},
            },
            "write": {
                "tool": "exec_command",
                "args": {"cmd": "apply_patch <<'EOF'\n*** Begin Patch\n*** Add File: {path}\n{content_plus}\n*** End Patch\nEOF"},
            },
        },
    },
    "qwen-code": {
        "roles": {
            "read_file": {"kind": "read", "fields": {"file_path": "path"}},
            "run_shell_command": {"kind": "shell", "fields": {"command": "command", "directory": "path"}},
            "edit": {"kind": "edit", "fields": {"file_path": "path", "old_string": "old", "new_string": "new"}},
            "write_file": {"kind": "write", "fields": {"file_path": "path", "content": "content"}},
            "grep_search": {"kind": "search", "fields": {"pattern": "pattern", "path": "path", "glob": "path"}},
            "glob": {"kind": "list", "fields": {"pattern": "pattern", "path": "path"}},
            "tool_call": {"kind": "meta", "fields": {"_name": "name", "_arguments": "arguments"}},
            "tool_search": {"kind": "other", "fields": {"query": "pattern"}},
        },
        "canned": {
            "read": {"tool": "read_file", "args": {"file_path": "{abspath}"}},
            "shell": {"tool": "run_shell_command", "args": {"command": "{command}", "description": "{description}"}},
            "search": {"tool": "grep_search", "args": {"pattern": "{pattern}"}},
            "edit": {"tool": "edit", "args": {"file_path": "{abspath}", "old_string": "{old}", "new_string": "{new}"}},
            "write": {"tool": "write_file", "args": {"file_path": "{abspath}", "content": "{content}"}},
        },
    },
}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n\n".join(p.get("text", "") for p in content if isinstance(p, dict))


def _from_responses(body: dict) -> tuple[list[dict], list[dict]]:
    """Map a Responses API request onto the chat-completions shape engines parse."""
    messages: list[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    for item in body.get("input", []):
        if item.get("type") != "message":
            continue
        role = "system" if item["role"] in ("developer", "system") else item["role"]
        messages.append({"role": role, "content": _text_of(item.get("content", []))})
    tools: list[dict] = []
    for t in body.get("tools", []):
        if t.get("type") == "function":
            fn = {k: t[k] for k in ("name", "description", "parameters", "strict") if k in t}
            tools.append({"type": "function", "function": fn})
        elif t.get("type") == "namespace":
            for inner in t.get("tools", []):
                if inner.get("type") == "function":
                    fn = {k: inner[k] for k in ("description", "parameters", "strict") if k in inner}
                    tools.append({"type": "function", "function": {"name": f"{t['name']}__{inner['name']}", **fn}})
        # hosted tools (web_search and friends) have no chat-completions equivalent
    return messages, tools


def to_profile(record: dict, name: str, task: str, replace: dict[str, str], meta: dict | None = None) -> dict:
    """Turn one recorded request into a profile. `task` is the prompt you gave the agent."""
    body = record["body"]
    if "input" in body:
        messages, tools = _from_responses(body)
    else:
        messages = json.loads(json.dumps(body["messages"]))
        tools = body.get("tools", [])
    # chat templates reject system messages after the first; merge the leading run
    lead = 0
    while lead < len(messages) and messages[lead]["role"] in ("system", "developer"):
        lead += 1
    if lead > 1:
        merged = "\n\n".join(_text_of(m["content"]) for m in messages[:lead])
        messages = [{"role": "system", "content": merged}] + messages[lead:]
    for m in messages:
        if m["role"] == "developer":
            m["role"] = "system"
    text = json.dumps({"messages": messages, "tools": tools})
    for old, new in replace.items():
        text = text.replace(old, new)
    data = json.loads(text)
    messages, tools = data["messages"], data["tools"]
    found = False
    for m in reversed(messages):
        if m["role"] != "user":
            continue
        c = m["content"]
        for variant in (f'"{task}"', task):
            if isinstance(c, str) and variant in c:
                m["content"] = c.replace(variant, "{task}")
                found = True
                break
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and variant in (p.get("text") or ""):
                        p["text"] = p["text"].replace(variant, "{task}")
                        found = True
                if found:
                    break
        if found:
            break
    if not found:
        raise ValueError("could not find the task text in the recorded user messages")
    params = {k: body[k] for k in ("parallel_tool_calls", "max_tokens", "max_completion_tokens", "tool_choice") if k in body}
    known = KNOWN.get(name, {})
    profile = {"name": name, **(meta or {}), "messages": messages, "tools": tools, "params": params}
    profile["roles"] = known.get("roles", {})
    profile["canned"] = known.get("canned") or _guess_canned(tools)
    return profile


def _guess_canned(tools: list[dict]) -> dict:
    """Pick read/shell/search/edit tools by name for histories of an unknown agent."""
    from .profiles import _infer_role

    canned: dict = {}
    for t in tools:
        fn = t["function"]
        props = (fn.get("parameters") or {}).get("properties", {})
        role = _infer_role(fn["name"], {k: None for k in props})
        kind = role["kind"]
        if kind in ("read", "shell", "search", "edit") and kind not in canned:
            tmpl = {"path": "{abspath}", "command": "{command}", "pattern": "{pattern}", "old": "{old}", "new": "{new}"}
            args = {k: tmpl[r] for k, r in role["fields"].items() if r in tmpl}
            canned[kind] = {"tool": fn["name"], "args": args}
    return canned
