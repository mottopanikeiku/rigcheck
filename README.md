# rigcheck

Test a local coding-agent setup the way the agent uses it. rigcheck sends your engine the real system prompt and tool schemas of OpenCode, Codex CLI or Qwen Code, checks every tool call that comes back, and times each turn of a scripted session with a stopwatch and error bars.

Point it at llama.cpp, Ollama, or anything else that speaks the OpenAI chat API.

## Quick start

```sh
llama-server -hf ggml-org/gpt-oss-20b-GGUF            # or ollama serve, or any OpenAI-compatible server
uvx --from git+https://github.com/mottopanikeiku/rigcheck rigcheck run http://localhost:8080/v1 --agent opencode
```

Add `--agent codex --agent qwen-code` to test more agents, `--trials 10 --sessions 5` for tighter intervals, `--md report.md --html report.html` for a report. Each run also writes its raw results as JSON.

## What it checks

First, one cold request with the agent's full prompt. If the engine counts fewer tokens than that prompt must contain, it was truncated and the run says so.

Then nine probes, each a small situation in a fake Python repo with one right move:

| Probe | Passes when the model |
|---|---|
| `read_first` | reads the file before answering a question about it |
| `run_command` | runs the shell command it was asked to run |
| `search` | searches for a symbol (or reads the file that defines it) |
| `edit_after_read` | makes the right edit using a file it just read |
| `stop_when_done` | answers once the tool result is in, without another call |
| `no_tool` | answers a plain question without calling anything |
| `parallel` | reads two files in one turn |
| `recover` | goes looking for the right file after "file not found" |
| `long_write` | writes a whole new file in one long string argument |

Every reply must also clear the hard checks: no tool-call markup leaked into the text (`<tool_call>`, `<|channel|>`, `<function=`) and no call written out as text (`{"name": "read"...}`, `[read:src/a.py]`), arguments that parse as a JSON object and validate against the agent's own schema for that tool, no made-up tool names, and no reply cut off by the token limit. Replies stream by default, and tool calls are assembled from the deltas strictly by index, as the OpenAI spec says and agents do. Missing ids, reused indexes and a wrong `finish_reason` are reported as warnings.

Last, a timed five-turn debugging session with the agent's prompt: search, read a 180-line file, edit, run the tests. Turn 1 starts with a random nonce so nothing can come from cache. Turns 2 to 5 only append to the conversation, so a working prefix cache should make them pay for the new tokens alone. rigcheck reports first-token time for the cold and the warm turns, decode speed, and **cache misses**: warm turns that took at least 75% of what reading their whole prompt cold would take. That test needs only a stopwatch, so it works on any engine.

Each agent's tools differ (`read` vs `read_file` vs `exec_command "sed -n ..."`), so each profile maps them onto generic actions, and a Codex `apply_patch` through the shell counts as an edit.

## Statistics

Pass rates come with 95% Wilson intervals and their n. Latency medians get 95% bootstrap intervals that resample whole sessions, since turns within a session are correlated. Probes run at the sampling settings the server is configured with and a different seed per trial, so trials are independent draws rather than one greedy answer repeated.

```sh
rigcheck compare a.json b.json
```

pairs two runs trial by trial: exact McNemar test and a paired bootstrap interval for the pass-rate difference, plus ratio-of-medians intervals for first-token times.

## Agents

| Profile | Agent | Prompt + tools |
|---|---|---|
| `opencode` | [OpenCode](https://github.com/sst/opencode) 1.19.5 | 10 tools, ~30k characters |
| `codex` | [Codex CLI](https://github.com/openai/codex) 0.166.0 | 12 tools, ~38k characters |
| `qwen-code` | [Qwen Code](https://github.com/QwenLM/qwen-code) 0.26.5 | 10 tools, ~69k characters |
| `minimal` | a 5-tool control prompt | ~2k characters |

The bundled profiles are the agents' own requests, recorded and replayed. Their prompts stay under the agents' licenses; see [`src/rigcheck/agents/NOTICE`](src/rigcheck/agents/NOTICE). Codex CLI only speaks the Responses API, so its request is converted to chat completions.

To add an agent, record what it sends:

```sh
rigcheck capture --port 9909                     # then point the agent at http://127.0.0.1:9909/v1 and give it one task
rigcheck profile captured/request-002.json --name myagent --task "the task you gave it" \
    --replace /home/you/project=/home/dev/project -o myagent.json
rigcheck run http://localhost:8080/v1 --agent myagent.json
```

Tool roles are guessed from names and argument names; check `roles` and `canned` in the JSON (the bundled profiles show the format).

Any OpenAI-compatible engine works as is. Engine detection only adds metadata (build, loaded context) for llama.cpp and Ollama; another engine needs a branch in [`engine.py`](src/rigcheck/engine.py).

## Development

```sh
uv sync
uv run pytest     # runs everything against a mock server with known faults; no model needed
```

MIT license.
