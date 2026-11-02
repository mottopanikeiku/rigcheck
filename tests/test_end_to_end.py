"""The whole pipeline against the mock server: run, summarize, render, compare."""

import json

import pytest

from rigcheck import report, runner
from rigcheck.cli import main

from . import mockserver


@pytest.fixture(scope="module")
def url():
    server, url = mockserver.start()
    yield url
    server.shutdown()


def _run(url, model, agents=("minimal", "opencode"), trials=3, sessions=2, **kw):
    cfg = runner.RunConfig(base_url=url, model=model, agents=list(agents), trials=trials, sessions=sessions, **kw)
    return runner.run(cfg)


def test_good_server_passes_everything(url):
    res = _run(url, "good", agents=("minimal", "opencode", "codex", "qwen-code"))
    assert res["endpoint"]["engine"] == "llama.cpp"
    for row in report.summarize(res):
        assert row["fit"] == "fits"
        assert row["pass"] == row["n"] == 27, row["reasons"]
        assert not row["warnings"]
        assert row["prefix_reuse"][0] > 0.8
        if row["agent"] != "minimal":
            assert row["warm_over_cold"][0] < 0.5


def test_cache_miss_shows_up_in_warm_turns(url):
    res = _run(url, "nocache", agents=("qwen-code",), trials=0, sessions=3)
    (row,) = report.summarize(res)
    assert row["warm_over_cold"][0] > 1.0
    assert row["prefix_reuse"][0] < 0.2
    assert row["engine_cache_share"] == 0


def test_markup_leak_fails_tool_probes(url):
    res = _run(url, "leak", agents=("opencode",), sessions=0)
    (row,) = report.summarize(res)
    assert row["reasons"].get("markup_in_content", 0) >= 15
    # probes that need no tool still pass
    assert row["by_probe"]["no_tool"] == [3, 3]


def test_bad_json_and_index_bug(url):
    (row,) = report.summarize(_run(url, "badjson", agents=("minimal",), sessions=0))
    assert row["reasons"].get("bad_json")
    res = _run(url, "index0", agents=("minimal",), probes=["parallel"], sessions=0)
    (row,) = report.summarize(res)
    assert row["pass"] == 0 and row["reasons"] == {"bad_json": 3}
    assert row["warnings"].get("tool_call index reused by a different call id") == 3


def test_truncated_prompt_is_flagged(url):
    (row,) = report.summarize(_run(url, "truncate", agents=("qwen-code",), trials=0, sessions=0))
    assert row["fit"] == "truncated"


def test_non_streaming_path(url):
    (row,) = report.summarize(_run(url, "good", agents=("minimal",), sessions=0, stream=False))
    assert row["pass"] == row["n"]


def test_cli_run_report_compare(url, tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    main(["run", url, "--model", "good", "--agent", "minimal", "--trials", "2", "--sessions", "2", "-o", str(a), "--label", "good"])
    main(["run", url, "--model", "nocall", "--agent", "minimal", "--trials", "2", "--sessions", "2", "-o", str(b), "--label", "nocall"])
    out = capsys.readouterr().out
    assert "minimal" in out and "100%" in out
    md, html, svg, js = (tmp_path / f"r.{x}" for x in ("md", "html", "svg", "json"))
    main(["report", str(a), str(b), "--md", str(md), "--html", str(html), "--svg", str(svg), "--json", str(js), "--title", "mock"])
    assert "| good | minimal |" in md.read_text()
    assert svg.read_text().startswith("<svg") and "</svg>" in svg.read_text()
    assert "<table>" in html.read_text()
    assert len(json.loads(js.read_text())) == 2
    capsys.readouterr()
    main(["compare", str(a), str(b)])
    out = capsys.readouterr().out
    assert "McNemar" in out and "warm turns first token" in out


def test_summary_rescores_stored_replies(url):
    res = _run(url, "good", agents=("minimal",), sessions=0, trials=1)
    for r in res["agents"]["minimal"]["probes"]:
        r["pass"], r["reason"] = False, "stale"
    (row,) = report.summarize(res)
    assert row["pass"] == row["n"] == 9
