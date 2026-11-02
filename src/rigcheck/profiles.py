"""Agent profiles: the system prompt, context messages and tool schemas a coding agent
actually sends, plus a map from its tools to generic actions (read, shell, edit...)."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

PROJECT = "/home/dev/project"
TASK = "{task}"


@dataclass
class Action:
    kind: str  # read | shell | search | list | edit | write | other
    tool: str
    paths: list[str] = field(default_factory=list)
    command: str = ""
    pattern: str = ""
    old: str = ""
    new: str = ""
    content: str = ""
    patch: str = ""

    def touches(self, name: str) -> bool:
        """True if any path (or the command text) refers to file `name` (a relative path)."""
        base = name.rsplit("/", 1)[-1]
        for p in self.paths:
            if p.rstrip("/").endswith(name) or p.rsplit("/", 1)[-1] == base:
                return True
        return bool(self.command) and base in self.command


@dataclass
class Profile:
    name: str
    title: str
    version: str
    license: str
    source: str
    messages: list[dict]
    tools: list[dict]
    params: dict
    roles: dict  # tool name -> {"kind": ..., "fields": {arg: role}}
    canned: dict  # action kind -> {"tool": name, "args": {arg: template}}
    notes: str = ""

    @property
    def tool_names(self) -> set[str]:
        return {t["function"]["name"] for t in self.tools}

    def schema(self, tool: str) -> dict | None:
        for t in self.tools:
            if t["function"]["name"] == tool:
                return t["function"].get("parameters") or {"type": "object"}
        return None

    def prompt_chars(self) -> int:
        return len(json.dumps(self.messages)) + len(json.dumps(self.tools))

    def build_messages(self, task: str, nonce: str | None = None) -> list[dict]:
        """The agent's preamble with the user's task filled in.

        A nonce at the very start of the first message makes the prompt unique, so no
        engine can serve any of it from cache (a cold turn)."""
        msgs = json.loads(json.dumps(self.messages))
        placed = False
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str) and TASK in c:
                m["content"] = c.replace(TASK, task)
                placed = True
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and TASK in (part.get("text") or ""):
                        part["text"] = part["text"].replace(TASK, task)
                        placed = True
        if not placed:
            msgs.append({"role": "user", "content": task})
        if nonce:
            first = msgs[0]
            if isinstance(first.get("content"), str):
                first["content"] = f"[session {nonce}]\n" + first["content"]
            elif isinstance(first.get("content"), list) and first["content"]:
                first["content"][0]["text"] = f"[session {nonce}]\n" + first["content"][0]["text"]
        return msgs

    def call(self, kind: str, call_id: str, **fields: str) -> dict:
        """An assistant tool call, in this agent's own tool format, for a history."""
        spec = self.canned[kind]
        f = {
            "abspath": f"{PROJECT}/{fields.get('path', '')}",
            "content_plus": "\n".join("+" + line for line in fields.get("content", "").splitlines()),
            "description": "Run command",
            **fields,
        }
        args = {key: _fill(tmpl, f) for key, tmpl in spec["args"].items()}
        return {"id": call_id, "type": "function", "function": {"name": spec["tool"], "arguments": json.dumps(args)}}

    def classify(self, name: str | None, args: dict) -> Action:
        name = name or ""
        role = self.roles.get(name) or _infer_role(name, args)
        kind, fields = role["kind"], role.get("fields", {})
        a = Action(kind=kind, tool=name)
        for key, r in fields.items():
            v = args.get(key)
            if v is None:
                continue
            if r == "path":
                a.paths.append(str(v))
            elif r == "command":
                a.command = _command_text(v)
            elif r in ("pattern", "old", "new", "content", "patch"):
                setattr(a, r, v if isinstance(v, str) else json.dumps(v))
        if kind == "meta":  # a dispatcher tool such as Qwen Code's tool_call
            inner = args.get(fields.get("_name", "name")) or args.get("name")
            inner_args = args.get(fields.get("_arguments", "arguments")) or args.get("arguments") or {}
            if isinstance(inner_args, str):
                try:
                    inner_args = json.loads(inner_args)
                except json.JSONDecodeError:
                    inner_args = {}
            if isinstance(inner, str) and isinstance(inner_args, dict):
                inner_action = self.classify(inner, inner_args)
                inner_action.tool = f"{name}:{inner}"
                return inner_action
            return Action(kind="other", tool=name)
        if kind == "shell":
            _refine_shell(a)
        return a


def _fill(tmpl, fields: dict):
    if isinstance(tmpl, str):
        return re.sub(r"\{(\w+)\}", lambda m: str(fields.get(m.group(1), m.group(0))), tmpl)
    if isinstance(tmpl, list):
        return [_fill(t, fields) for t in tmpl]
    return tmpl


def _command_text(v) -> str:
    if isinstance(v, list):
        parts = [str(x) for x in v]
        if len(parts) >= 3 and parts[0] in ("bash", "sh", "zsh") and parts[1] in ("-c", "-lc"):
            return parts[2]
        return " ".join(parts)
    return str(v)


_READ_CMD = re.compile(r"(^|[\s;&|(])(cat|head|tail|nl|less|more|bat|sed\s+-n|awk)\b")
_SEARCH_CMD = re.compile(r"(^|[\s;&|(])(grep|rg|ag|ack|git\s+grep|find)\b")
_LIST_CMD = re.compile(r"(^|[\s;&|(])(ls|tree|find)\b")
_PATCH_FILE = re.compile(r"\*\*\* (?:Update|Add) File: (\S+)")


def _refine_shell(a: Action) -> None:
    cmd = a.command
    if "*** Begin Patch" in cmd or re.search(r"(^|\s)apply_patch\b", cmd):
        a.kind = "edit"
        a.patch = cmd
        a.paths += _PATCH_FILE.findall(cmd)
        a.old = "\n".join(line[1:] for line in cmd.splitlines() if line.startswith("-") and not line.startswith("---"))
        a.new = "\n".join(line[1:] for line in cmd.splitlines() if line.startswith("+") and not line.startswith("+++"))
        return
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()
    a.paths += [t for t in tokens if "/" in t or re.search(r"\.\w{1,5}$", t)]
    if _READ_CMD.search(cmd):
        a.kind = "read"
    elif _SEARCH_CMD.search(cmd):
        a.kind = "search"
        a.pattern = cmd
    elif _LIST_CMD.search(cmd):
        a.kind = "list"


_PATH_KEYS = ("path", "file", "filepath", "file_path", "absolute_path", "filename", "notebook_path", "target_file")


def _infer_role(name: str, args: dict) -> dict:
    """Best-effort mapping for tools without a declared role."""
    n = name.lower()
    fields = {}
    for k in args:
        lk = k.lower()
        if lk in _PATH_KEYS or lk.endswith("path") or lk.endswith("file"):
            fields[k] = "path"
        elif lk in ("command", "cmd", "script"):
            fields[k] = "command"
        elif lk in ("pattern", "query", "regex"):
            fields[k] = "pattern"
        elif lk in ("old", "old_string", "oldstring", "old_str", "search"):
            fields[k] = "old"
        elif lk in ("new", "new_string", "newstring", "new_str", "replace", "replacement"):
            fields[k] = "new"
        elif lk in ("content", "contents", "text", "file_text"):
            fields[k] = "content"
        elif lk in ("patch", "diff", "input"):
            fields[k] = "patch"
    if any(w in n for w in ("bash", "shell", "exec", "terminal", "run_command", "run")):
        kind = "shell"
    elif any(w in n for w in ("grep", "search", "find")):
        kind = "search"
    elif any(w in n for w in ("glob", "list", "ls")):
        kind = "list"
    elif any(w in n for w in ("edit", "replace", "patch", "str_replace")):
        kind = "edit"
    elif any(w in n for w in ("write", "create")):
        kind = "write"
    elif any(w in n for w in ("read", "view", "cat", "open")):
        kind = "read"
    else:
        kind = "other"
    return {"kind": kind, "fields": fields}


def bundled() -> list[str]:
    root = resources.files("rigcheck") / "agents"
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".json"))


def load(name_or_path: str) -> Profile:
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        data = json.loads(p.read_text())
    else:
        f = resources.files("rigcheck") / "agents" / f"{name_or_path}.json"
        if not f.is_file():
            raise SystemExit(f"unknown agent '{name_or_path}'. Bundled: {', '.join(bundled())}")
        data = json.loads(f.read_text())
    return Profile(
        name=data["name"],
        title=data.get("title", data["name"]),
        version=data.get("version", ""),
        license=data.get("license", ""),
        source=data.get("source", ""),
        messages=data["messages"],
        tools=data["tools"],
        params=data.get("params", {}),
        roles=data.get("roles", {}),
        canned=data["canned"],
        notes=data.get("notes", ""),
    )
