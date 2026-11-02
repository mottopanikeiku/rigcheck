#!/usr/bin/env python3
"""Run the README matrix: each model on llama.cpp and on Ollama, with every bundled agent.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server OLLAMA=~/ollama/bin/ollama MODELS=~/models \\
        uv run scripts/matrix.py --trials 12 --sessions 5

GGUF paths are relative to $MODELS; Ollama models are the library tags (pull them first).
Each setup starts its own server, runs, and shuts it down. With --lock FILE, every setup
holds an exclusive flock on FILE while it runs, and waits for the 1-minute load average to
drop below --max-load before starting."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from rigcheck import runner

QWEN = ["--temp", "1", "--top-k", "20", "--top-p", "0.95", "--presence-penalty", "1.5"]

# Sampling for llama-server matches the parameters Ollama ships with each library model,
# which are the model cards' recommended defaults.
MODELS = {
    "qwen3.5-4b": {"title": "Qwen3.5-4B", "gguf": "Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", "ollama": "qwen3.5:4b", "sampling": QWEN},
    "gemma-4-e4b": {
        "title": "Gemma 4 E4B",
        "gguf": "gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
        "ollama": "gemma4:e4b",
        "sampling": ["--temp", "1", "--top-k", "64", "--top-p", "0.95"],
    },
    "gpt-oss-20b": {
        "title": "gpt-oss-20b",
        "gguf": "gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf",
        "ollama": "gpt-oss:20b",
        "sampling": ["--temp", "1", "--top-p", "1", "--top-k", "0"],
    },
    "qwen3.6-35b-a3b": {
        "title": "Qwen3.6-35B-A3B",
        "gguf": "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "ollama": "qwen3.6:35b",
        "sampling": QWEN,
    },
}

ENGINES = {
    "llama.cpp": {"ctx": 65536},
    "ollama": {"env": {}},
    "ollama-64k": {"env": {"OLLAMA_CONTEXT_LENGTH": "65536"}},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def wait_ready(url: str, timeout: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise SystemExit(f"server at {url} not ready after {timeout:.0f}s")


def start_llama(model: dict, args) -> tuple[subprocess.Popen, str]:
    gguf = Path(os.environ["MODELS"]).expanduser() / model["gguf"]
    cmd = [
        os.path.expanduser(os.environ["LLAMA_SERVER"]),
        "-m",
        str(gguf),
        "-c",
        str(ENGINES["llama.cpp"]["ctx"]),
        "-t",
        str(args.threads),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--alias",
        model["title"],
        *model["sampling"],
    ]
    proc = subprocess.Popen(["taskset", "-c", args.cpus, *cmd], stdout=subprocess.DEVNULL, stderr=open(args.log, "a"))
    url = f"http://127.0.0.1:{args.port}"
    wait_ready(f"{url}/health")
    return proc, f"{url}/v1"


def start_ollama(engine: str, args) -> tuple[subprocess.Popen, str]:
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{args.port}", **ENGINES[engine]["env"]}
    cmd = [os.path.expanduser(os.environ["OLLAMA"]), "serve"]
    proc = subprocess.Popen(["taskset", "-c", args.cpus, *cmd], env=env, stdout=subprocess.DEVNULL, stderr=open(args.log, "a"))
    url = f"http://127.0.0.1:{args.port}"
    wait_ready(f"{url}/api/version")
    return proc, f"{url}/v1"


def stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def host_info(args) -> dict:
    cpu = ""
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                cpu = line.split(":", 1)[1].strip()
    except OSError:
        pass
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {"cpu": cpu, "cores_used": len(expand(args.cpus)), "memory_gib": round(mem), "machine": platform.machine(), "threads": args.threads}


def expand(cpus: str) -> list[int]:
    out = []
    for part in cpus.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def wait_quiet(max_load: float) -> None:
    while True:
        load = os.getloadavg()[0]
        if load < max_load:
            return
        log(f"load average {load:.1f} >= {max_load}; waiting")
        time.sleep(60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--agents", nargs="+", default=["opencode", "codex", "qwen-code", "minimal"])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--cpus", default="0-15")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--out", default="results")
    ap.add_argument("--lock", help="flock this file around each setup")
    ap.add_argument("--max-load", type=float, default=3.0)
    ap.add_argument("--log", default="/tmp/rigcheck-matrix-server.log")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        model = MODELS[key]
        for engine in args.engines:
            path = out / f"{engine}--{key}.json"
            if args.skip_existing and path.exists():
                continue
            lock = open(args.lock, "a") if args.lock else None
            if lock:
                log(f"waiting for {args.lock}")
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                wait_quiet(args.max_load)
                log(f"{engine} / {model['title']}: starting server")
                if engine == "llama.cpp":
                    proc, url = start_llama(model, args)
                    model_id = model["title"]
                else:
                    proc, url = start_ollama(engine, args)
                    model_id = model["ollama"]
                try:
                    cfg = runner.RunConfig(
                        base_url=url,
                        model=model_id,
                        agents=args.agents,
                        trials=args.trials,
                        sessions=args.sessions,
                        label=f"{engine} · {model['title']}",
                    )
                    res = runner.run(cfg)
                    res["host"].update(host_info(args))
                    res["setup"] = {
                        "engine": engine,
                        "model": key,
                        "title": model["title"],
                        "server_args": model["sampling"] if engine == "llama.cpp" else ENGINES[engine]["env"],
                    }
                    path.write_text(json.dumps(res, indent=1))
                    log(f"wrote {path}")
                finally:
                    stop(proc)
            finally:
                if lock:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    lock.close()


if __name__ == "__main__":
    main()
