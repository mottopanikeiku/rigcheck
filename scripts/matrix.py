#!/usr/bin/env python3
"""Run the README matrix: each model on llama.cpp and on Ollama, with the bundled agents.

    export LLAMA_SERVER=~/llama.cpp/build/bin/llama-server OLLAMA=~/ollama/bin/ollama MODELS=~/models
    uv run scripts/matrix.py --phase probes --cpus 8-15 --threads 8
    uv run scripts/matrix.py --phase timing --cpus 0-15 --threads 16 --lock /tmp/bench.lock

GGUF paths are relative to $MODELS; the Ollama models are library tags (pull them first).
Every setup gets a fresh server. Pass/fail probes don't care about timing noise, so they
can run on a few cores of a busy box; timed sessions want the whole machine, so with
--lock each setup holds an exclusive flock and first waits for the 1-minute load average
to drop below --max-load. Both phases write into results/<engine>--<model>.json."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from rigcheck import runner

QWEN = ["--temp", "1", "--top-k", "20", "--top-p", "0.95", "--presence-penalty", "1.5"]

# llama-server gets the sampling parameters Ollama ships with each library model (the
# model cards' recommended defaults); requests carry no sampling parameters of their own.
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

# llama.cpp and Ollama as installed, plus Ollama with the context its docs recommend for agents.
ENGINES = {
    "llama.cpp": {"env": {}},
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


def start_llama(model: dict, args) -> tuple[subprocess.Popen, str, str]:
    gguf = Path(os.environ["MODELS"]).expanduser() / model["gguf"]
    cmd = [
        os.path.expanduser(os.environ["LLAMA_SERVER"]),
        "-m",
        str(gguf),
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
    proc = subprocess.Popen(["taskset", "-c", args.cpus, *cmd], stdout=subprocess.DEVNULL, stderr=open(args.log, "a"), start_new_session=True)
    url = f"http://127.0.0.1:{args.port}"
    wait_ready(f"{url}/health")
    return proc, f"{url}/v1", model["title"]


def start_ollama(engine: str, model: dict, args) -> tuple[subprocess.Popen, str, str]:
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{args.port}", **ENGINES[engine]["env"]}
    exe = os.path.expanduser(os.environ["OLLAMA"])
    proc = subprocess.Popen(["taskset", "-c", args.cpus, exe, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=open(args.log, "a"), start_new_session=True)
    url = f"http://127.0.0.1:{args.port}"
    wait_ready(f"{url}/api/version")
    name = model["ollama"]
    if args.threads != os.cpu_count():
        # Ollama starts one thread per core it detects, ignoring the CPU mask; on fewer
        # cores that oversubscribes them. A derived model with num_thread fixes that.
        derived = f"{name.replace(':', '-')}-t{args.threads}"
        modelfile = f"FROM {name}\nPARAMETER num_thread {args.threads}\n"
        subprocess.run(
            [exe, "create", derived, "-f", "/dev/stdin"], input=modelfile, text=True, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        name = derived
    return proc, f"{url}/v1", name


class Yield(threading.Thread):
    """Freeze the server (SIGSTOP) while anyone else holds the bench lock exclusively,
    so pass/fail probes never disturb someone's timed run."""

    def __init__(self, lockfile: str, proc: subprocess.Popen):
        super().__init__(daemon=True)
        self.lockfile, self.proc, self.done = lockfile, proc, threading.Event()

    def run(self) -> None:
        paused = False
        with open(self.lockfile, "a") as fd:
            while not self.done.wait(5):
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    busy = False
                except BlockingIOError:
                    busy = True
                if busy != paused:
                    os.killpg(self.proc.pid, signal.SIGSTOP if busy else signal.SIGCONT)
                    log("bench lock taken: pausing server" if busy else "bench lock free: resuming server")
                    paused = busy
            if paused:
                os.killpg(self.proc.pid, signal.SIGCONT)


def stop(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGCONT)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def expand(cpus: str) -> list[int]:
    out = []
    for part in cpus.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def host_info(args) -> dict:
    cpu = ""
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
            if line.startswith("Model name:"):
                cpu = line.split(":", 1)[1].strip()
    except OSError:
        pass
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {
        "cpu": cpu,
        "cpus": args.cpus,
        "cores_used": len(expand(args.cpus)),
        "threads": args.threads,
        "memory_gib": round(mem),
        "machine": platform.machine(),
        "load_at_start": round(os.getloadavg()[0], 2),
    }


def acquire_quiet(path: str, max_load: float):
    """Take the bench lock once the machine is quiet. Holding the lock pauses our own
    probe runs, so give the load average five minutes to settle; if it doesn't, let go
    of the lock (someone else may need it) and try again later."""
    while True:
        lock = open(path, "a")
        log(f"waiting for {path}")
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(5):
            if os.getloadavg()[0] < max_load:
                return lock
            time.sleep(60)
        log(f"load average {os.getloadavg()[0]:.1f} >= {max_load} with the lock held; retrying in 10 minutes")
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        time.sleep(600)


def merge(path: Path, res: dict, phase: str) -> dict:
    old = json.loads(path.read_text()) if path.exists() else None
    if not old:
        res["phases"] = {phase: {"started": res["started"], "finished": res["finished"], "host": res["host"]}}
        return res
    old.setdefault("phases", {})[phase] = {"started": res["started"], "finished": res["finished"], "host": res["host"]}
    for name, entry in res["agents"].items():
        dst = old["agents"].setdefault(name, entry)
        if phase == "probes":
            dst["probes"] = entry["probes"]
            dst["fit"] = entry["fit"]
        else:
            dst["turns"] = entry["turns"]
    for k, v in res["endpoint"].items():
        old["endpoint"].setdefault(k, v)
    return old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["probes", "timing"], required=True)
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--agents", nargs="+", default=["opencode"])
    ap.add_argument("--trials", type=int, default=10, help="per probe, probes phase")
    ap.add_argument("--sessions", type=int, default=5, help="timed sessions per agent, timing phase")
    ap.add_argument("--cpus", default="0-15")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--out", default="results")
    ap.add_argument("--lock", help="flock this file around each setup")
    ap.add_argument("--max-load", type=float, default=3.0)
    ap.add_argument("--yield-to", help="pause while another process holds this lock (probes phase)")
    ap.add_argument("--log", default="/tmp/rigcheck-matrix-server.log")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally blocks: resume and stop the server
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        model = MODELS[key]
        for engine in args.engines:
            path = out / f"{engine}--{key}.json"
            lock = acquire_quiet(args.lock, args.max_load) if args.lock else None
            try:
                log(f"{args.phase}: {engine} / {model['title']}")
                if engine == "llama.cpp":
                    proc, url, model_id = start_llama(model, args)
                else:
                    proc, url, model_id = start_ollama(engine, model, args)
                yielder = Yield(args.yield_to, proc) if args.yield_to else None
                if yielder:
                    yielder.start()
                try:
                    cfg = runner.RunConfig(
                        base_url=url,
                        model=model_id,
                        agents=args.agents,
                        trials=args.trials if args.phase == "probes" else 0,
                        sessions=args.sessions if args.phase == "timing" else 0,
                        label=f"{engine} · {model['title']}",
                        fit=args.phase == "probes",
                        timeout=6 * 3600 if yielder else 900,
                    )
                    res = runner.run(cfg)
                    res["host"].update(host_info(args))
                    res["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                    res["setup"] = {
                        "engine": engine,
                        "model": key,
                        "title": model["title"],
                        "ollama_tag": model["ollama"],
                        "gguf": model["gguf"],
                        "llama_server_args": model["sampling"],
                        "env": ENGINES[engine]["env"],
                    }
                    path.write_text(json.dumps(merge(path, res, args.phase), indent=1))
                    log(f"wrote {path}")
                finally:
                    if yielder:
                        yielder.done.set()
                        yielder.join()
                    stop(proc)
            finally:
                if lock:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    lock.close()


if __name__ == "__main__":
    main()
