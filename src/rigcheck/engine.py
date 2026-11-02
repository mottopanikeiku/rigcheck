"""Work out what is serving the endpoint, and how much context it actually loaded."""

from __future__ import annotations

from .client import Client


def root_of(base_url: str) -> str:
    b = base_url.rstrip("/")
    return b[: -len("/v1")] if b.endswith("/v1") else b


def first_model(client: Client) -> str | None:
    data = client.get_json(f"{client.base_url}/models")
    if isinstance(data, dict):
        for m in data.get("data") or []:
            if isinstance(m, dict) and m.get("id"):
                return m["id"]
    return None


def describe(client: Client, model: str) -> dict:
    root = root_of(client.base_url)
    info: dict = {"engine": "unknown", "version": None, "model": model, "n_ctx": None}
    props = client.get_json(f"{root}/props")
    if isinstance(props, dict) and ("build_info" in props or "default_generation_settings" in props):
        info["engine"] = "llama.cpp"
        info["version"] = props.get("build_info")
        gen = props.get("default_generation_settings") or {}
        info["n_ctx"] = gen.get("n_ctx") or props.get("n_ctx")
        info["slots"] = props.get("total_slots")
        info["model_path"] = (props.get("model_path") or "").rsplit("/", 1)[-1] or None
        return info
    ver = client.get_json(f"{root}/api/version")
    if isinstance(ver, dict) and "version" in ver:
        info["engine"] = "ollama"
        info["version"] = ver["version"]
        try:
            r = client.http.post(f"{root}/api/show", json={"model": model}, timeout=10.0)
            show = r.json() if r.status_code == 200 else {}
        except Exception:
            show = {}
        details = show.get("details") or {}
        info["quantization"] = details.get("quantization_level")
        info["parameter_size"] = details.get("parameter_size")
        for k, v in (show.get("model_info") or {}).items():
            if k.endswith(".context_length"):
                info["model_max_ctx"] = v
        return info
    return info


def loaded_context(client: Client, info: dict) -> int | None:
    """Context the engine is really running with, once the model is loaded."""
    if info.get("engine") == "ollama":
        ps = client.get_json(f"{root_of(client.base_url)}/api/ps")
        for m in (ps or {}).get("models") or []:
            if m.get("name") == info["model"] or m.get("model") == info["model"]:
                return m.get("context_length")
    return info.get("n_ctx")
