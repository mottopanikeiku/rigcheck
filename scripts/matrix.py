#!/usr/bin/env python3
"""Run the README matrix: each model on llama.cpp and on Ollama, with the bundled agents.

    export LLAMA_SERVER=~/llama.cpp/build/bin/llama-server OLLAMA=~/ollama/bin/ollama MODELS=~/models
    uv run scripts/matrix.py --phase timing --lock /tmp/bench.lock
    uv run scripts/matrix.py --phase probes --yield-to /tmp/bench.lock

GGUF paths are relative to $MODELS; the Ollama models are library tags (pull them first).
Every (engine, model, agent) gets a fresh server, and writes into
results/<engine>--<model>.json under that agent.

Timed sessions want the whole machine. With --lock, each one takes an exclusive flock,
waits until the 1-minute load average is below --max-load and other processes used less
than --max-others cores over a minute, and samples other processes' CPU use while it runs.
A run during which they averaged more than --max-others cores is thrown away and redone
later. Pass/fail probes don't care about timing noise; with --yield-to they freeze while
someone else holds the lock."""

from __future__ import annotations

import argparse
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

from rigcheck import __version__, runner

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
    "llama.cpp": {"name": "llama.cpp", "env": {}},
    "ollama": {"name": "Ollama default", "env": {}},
    "ollama-64k": {"name": "Ollama 64k", "env": {"OLLAMA_CONTEXT_LENGTH": "65536"}},
}

CLK_TCK = os.sysconf("SC_CLK_TCK")


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


# ---------------------------------------------------------------- servers

SIGNALS = {signal.SIGINT, signal.SIGTERM}
SERVERS: set[subprocess.Popen] = set()


def spawn(cmd: list[str], logfile: str, env: dict | None = None) -> subprocess.Popen:
    """Start a server in its own process group (so Ollama's runner child can be paused and
    stopped with it), with the signals main() blocks unblocked again."""
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(logfile, "a"),
        start_new_session=True,
        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, SIGNALS),
    )
    SERVERS.add(proc)
    return proc


def reaper() -> None:
    """Ctrl-C and SIGTERM land here, whatever the other threads are blocked on: resume and
    kill any server still running (a paused one would otherwise linger), then exit."""
    sig = signal.sigwait(SIGNALS)
    for proc in list(SERVERS):
        for s in (signal.SIGCONT, signal.SIGKILL):
            try:
                os.killpg(proc.pid, s)
            except ProcessLookupError:
                pass
    os._exit(128 + sig)


def start_llama(model: dict, args) -> tuple[subprocess.Popen, str, str]:
    gguf = Path(os.environ["MODELS"]).expanduser() / model["gguf"]
    exe = os.path.expanduser(os.environ["LLAMA_SERVER"])
    cmd = [exe, "-m", str(gguf), "-t", str(args.threads), "--host", "127.0.0.1", "--port", str(args.port), "--alias", model["title"], *model["sampling"]]
    proc = spawn(["taskset", "-c", args.cpus, *cmd], args.log)
    url = f"http://127.0.0.1:{args.port}"
    wait_ready(f"{url}/health")
    return proc, f"{url}/v1", model["title"]


def start_ollama(engine: str, model: dict, args) -> tuple[subprocess.Popen, str, str]:
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{args.port}", **ENGINES[engine]["env"]}
    exe = os.path.expanduser(os.environ["OLLAMA"])
    proc = spawn(["taskset", "-c", args.cpus, exe, "serve"], args.log, env=env)
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


def stop(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGCONT)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # whatever is left in the group, e.g. Ollama's runner
    except ProcessLookupError:
        pass
    proc.wait()
    SERVERS.discard(proc)


class Yield(threading.Thread):
    """Freeze the server (SIGSTOP) while anyone else holds the bench lock exclusively,
    so pass/fail probes never disturb someone's timed run."""

    def __init__(self, lockfile: str, proc: subprocess.Popen):
        super().__init__(daemon=True)
        self.lockfile, self.proc, self.done = lockfile, proc, threading.Event()
        self.paused_s = 0.0

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
                if busy:
                    self.paused_s += 5
                if busy != paused:
                    os.killpg(self.proc.pid, signal.SIGSTOP if busy else signal.SIGCONT)
                    log("bench lock taken: pausing server" if busy else "bench lock free: resuming server")
                    paused = busy
            if paused:
                os.killpg(self.proc.pid, signal.SIGCONT)


# ---------------------------------------------------------------- machine load


def busy_jiffies() -> int:
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return v[0] + v[1] + v[2] + v[5] + v[6] + v[7]  # user nice system irq softirq steal


def own_jiffies(pgid: int | None) -> int:
    """CPU time of this process plus the server's process group."""
    total = 0
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
        except OSError:
            continue
        rest = s[s.rindex(")") + 2 :].split()
        if int(d) == me or (pgid is not None and int(rest[2]) == pgid):
            total += int(rest[11]) + int(rest[12])
    return total


class Monitor(threading.Thread):
    """Samples how many cores everything except us and our server keeps busy."""

    def __init__(self, pgid: int | None, every: float = 5.0):
        super().__init__(daemon=True)
        self.pgid, self.every, self.done = pgid, every, threading.Event()
        self.samples: list[float] = []

    def run(self) -> None:
        t, b, o = time.monotonic(), busy_jiffies(), own_jiffies(self.pgid)
        while not self.done.wait(self.every):
            t2, b2, o2 = time.monotonic(), busy_jiffies(), own_jiffies(self.pgid)
            self.samples.append(max(0.0, ((b2 - b) - (o2 - o)) / CLK_TCK / (t2 - t)))
            t, b, o = t2, b2, o2

    def stop(self) -> dict:
        self.done.set()
        self.join()
        s = self.samples
        return {"others_cores_mean": round(sum(s) / len(s), 3) if s else None, "others_cores_max": round(max(s), 3) if s else None, "samples": len(s)}


def others_now(seconds: float = 60) -> float:
    m = Monitor(None)
    m.start()
    time.sleep(seconds)
    return m.stop()["others_cores_mean"] or 0.0


def acquire_quiet(path: str, max_load: float, max_others: float):
    """Take the bench lock once the machine is quiet. Holding the lock pauses our own
    probe runs, so give the machine five minutes to settle; if it doesn't, let go of the
    lock (someone else may need it) and try again in ten."""
    waited_since = time.time()
    while True:
        lock = open(path, "a")
        log(f"waiting for {path}")
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(5):
            load = os.getloadavg()[0]
            others = others_now(60) if load < max_load else None
            if others is not None and others < max_others:
                return lock, round((time.time() - waited_since) / 60, 1)
            log(f"not quiet: load {load:.1f}" + (f", others {others:.2f} cores" if others is not None else ""))
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        log(f"still busy after five minutes ({(time.time() - waited_since) / 60:.0f} min so far); retrying in 10 minutes")
        time.sleep(600)


# ---------------------------------------------------------------- results


def expand(cpus: str) -> list[int]:
    out = []
    for part in cpus.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def host_info() -> dict:
    cpu = ""
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
            if line.startswith("Model name:"):
                cpu = line.split(":", 1)[1].strip()
    except OSError:
        pass
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {"cpu": cpu, "cores": os.cpu_count(), "memory_gib": round(mem), "machine": platform.machine(), "system": platform.system()}


def save(path: Path, res: dict, engine: str, key: str, phase: str, run_info: dict) -> None:
    """Merge one agent's run into results/<engine>--<model>.json."""
    model = MODELS[key]
    out = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "label": f"{ENGINES[engine]['name']} · {model['title']}",
            "setup": {
                "engine": engine,
                "model": key,
                "title": model["title"],
                "ollama_tag": model["ollama"],
                "gguf": model["gguf"],
                "llama_server_args": model["sampling"] if engine == "llama.cpp" else None,
                "env": ENGINES[engine]["env"],
            },
            "host": host_info(),
            "endpoint": res["endpoint"],
            "agents": {},
        }
    )
    for k, v in res["endpoint"].items():
        out["endpoint"].setdefault(k, v)
    for name, entry in res["agents"].items():
        dst = out["agents"].setdefault(name, {"profile": entry["profile"]})
        info = {**run_info, "rigcheck": __version__, "started": res["started"], "finished": res["finished"]}
        if phase == "probes":
            dst["fit"], dst["probes"], dst["probe_run"] = entry["fit"], entry["probes"], info
        else:
            dst["turns"], dst["timing_run"] = entry["turns"], info
    path.write_text(json.dumps(out, indent=1))


def run_one(args, engine: str, key: str, agent: str, out: Path) -> None:
    model = MODELS[key]
    path = out / f"{engine}--{key}.json"
    discarded: list[dict] = []
    while True:
        lock, waited = acquire_quiet(args.lock, args.max_load, args.max_others) if args.lock else (None, 0)
        try:
            log(f"{args.phase}: {engine} / {model['title']} / {agent}")
            load_start = os.getloadavg()
            proc, url, model_id = start_llama(model, args) if engine == "llama.cpp" else start_ollama(engine, model, args)
            yielder = Yield(args.yield_to, proc) if args.yield_to else None
            monitor = Monitor(proc.pid) if args.phase == "timing" else None
            for t in (yielder, monitor):
                if t:
                    t.start()
            try:
                cfg = runner.RunConfig(
                    base_url=url,
                    model=model_id,
                    agents=[agent],
                    trials=args.trials if args.phase == "probes" else 0,
                    sessions=args.sessions if args.phase == "timing" else 0,
                    concurrency=1,
                    timeout=6 * 3600,
                    label=f"{ENGINES[engine]['name']} · {model['title']}",
                    fit=args.phase == "probes",
                )
                res = runner.run(cfg)
                load_end = os.getloadavg()
                info = {
                    "cpus": args.cpus,
                    "threads": args.threads,
                    "concurrency": 1,
                    "load_start": [round(x, 2) for x in load_start],
                    "load_end": [round(x, 2) for x in load_end],
                }
                if monitor:
                    info.update(monitor.stop())
                    info["discarded_runs"] = discarded
                    info["waited_minutes_for_quiet"] = waited
                if yielder:
                    yielder.done.set()
                    yielder.join()
                    info["paused_seconds"] = yielder.paused_s
            finally:
                if yielder:
                    yielder.done.set()
                stop(proc)
        finally:
            if lock:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
        if monitor and (info["others_cores_mean"] or 0) > args.max_others:
            log(f"other processes used {info['others_cores_mean']:.2f} cores on average; discarding this run")
            discarded.append({k: info[k] for k in ("load_start", "load_end", "others_cores_mean", "others_cores_max")} | {"finished": res["finished"]})
            time.sleep(600)
            continue
        save(path, res, engine, key, args.phase, info)
        log(f"wrote {path} ({agent})")
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["probes", "timing"], required=True)
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--agents", nargs="+", default=["opencode"])
    ap.add_argument("--trials", type=int, default=6, help="per probe (probes phase)")
    ap.add_argument("--sessions", type=int, default=5, help="timed sessions per agent (timing phase)")
    ap.add_argument("--cpus", default="0-15")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--out", default="results")
    ap.add_argument("--lock", help="hold this flock around each timed run")
    ap.add_argument("--max-load", type=float, default=4.0, help="1-minute load average to wait for")
    ap.add_argument("--max-others", type=float, default=1.0, help="cores other processes may use, before and during a timed run")
    ap.add_argument("--yield-to", help="pause while another process holds this lock")
    ap.add_argument("--log", default="/tmp/rigcheck-matrix-server.log")
    args = ap.parse_args()
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)  # before any thread starts, so only reaper() gets them
    threading.Thread(target=reaper, daemon=True).start()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for key in args.models:
        for engine in args.engines:
            for agent in args.agents:
                run_one(args, engine, key, agent, out)
    log(f"done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
