"""A fake OpenAI-compatible server with known faults, so the suite can test itself
without a model. The model name picks the behaviour:

good      reference answers, streamed tool-call arguments split across chunks, prefix cache
nocache   like good, but every request pays for the whole prompt
leak      tool calls written into the text as <tool_call> markup
badjson   tool-call arguments cut off mid-string
index0    parallel calls all streamed with index 0 (a real-world bug)
truncate  reports a 4096-token prompt no matter what was sent
nocall    never calls a tool
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rigcheck import profiles
from rigcheck.probes import PROBES

SECONDS_PER_CHAR = 1e-5


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.cached = ""


def _reply_for(headers, body) -> tuple[str, list[dict]]:
    probe = headers.get("X-Rigcheck-Probe", "")
    agent = headers.get("X-Rigcheck-Agent")
    name, _, idx = probe.partition("/")
    if name not in PROBES or not agent:
        return "ok", []
    prof = profiles.load(agent)
    ref = PROBES[name].oracle(PROBES[name].instances[int(idx)])
    calls = [prof.call(kind, f"call_{i}", **fields) for i, (kind, fields) in enumerate(ref.calls)]
    return ref.content, calls


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.endswith("/models"):
                self._json({"object": "list", "data": [{"id": "good", "object": "model"}]})
            elif self.path == "/props":
                self._json({"build_info": "mock", "default_generation_settings": {"n_ctx": 65536}, "total_slots": 1})
            else:
                self._json({}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            mode = body.get("model", "good")
            msgs = body.get("messages") or []
            # tools, then the system prompt, then the conversation, as Qwen's chat templates render it
            prompt = json.dumps(body.get("tools")) + json.dumps(msgs)
            with state.lock:
                if mode == "nocache":
                    reused = 0
                else:
                    reused = 0
                    for a, b in zip(state.cached, prompt):
                        if a != b:
                            break
                        reused += 1
                state.cached = prompt
            new = len(prompt) - reused
            time.sleep(0.001 + new * SECONDS_PER_CHAR)
            content, calls = _reply_for(self.headers, body)
            if mode == "nocall":
                content, calls = "Sure, I can help with that.", []
            elif mode == "leak" and calls:
                content = "".join(
                    f"<tool_call>{json.dumps({'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments'])})}</tool_call>" for c in calls
                )
                calls = []
            elif mode == "badjson":
                for c in calls:
                    c["function"]["arguments"] = c["function"]["arguments"][:-7]
            prompt_tokens = 4096 if mode == "truncate" else len(prompt) // 4
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 24, "total_tokens": prompt_tokens + 24}
            timings = {"prompt_n": new // 4, "cache_n": reused // 4, "prompt_ms": 1.0}
            finish = "tool_calls" if calls else "stop"
            if not body.get("stream"):
                msg = {"role": "assistant", "content": content or None}
                if calls:
                    msg["tool_calls"] = calls
                self._json({"choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage, "timings": timings})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            def delta(d, finish_reason=None):
                send({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": d, "finish_reason": finish_reason}]})

            delta({"role": "assistant"})
            for word in (content or "").split(" "):
                if word:
                    delta({"content": word + " "})
                    time.sleep(0.0005)
            for i, c in enumerate(calls):
                idx = 0 if mode == "index0" else i
                args = c["function"]["arguments"]
                cuts = [0, len(args) // 3, 2 * len(args) // 3, len(args)]
                delta({"tool_calls": [{"index": idx, "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": ""}}]})
                for a, b in zip(cuts, cuts[1:]):
                    delta({"tool_calls": [{"index": idx, "function": {"arguments": args[a:b]}}]})
                    time.sleep(0.0005)
            delta({}, finish)
            send({"choices": [], "usage": usage, "timings": timings})
            self.wfile.write(b"data: [DONE]\n\n")

    return H


def start() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(State()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"
