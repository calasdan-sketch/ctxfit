"""Tests for the ctxfit CLI."""

import json

import pytest

from ctxfit.cli import build_parser, main


@pytest.fixture
def convo_file(tmp_path):
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello " * 500},
    ]
    p = tmp_path / "convo.json"
    p.write_text(json.dumps(msgs), encoding="utf-8")
    return str(p)


def test_analyze_table(convo_file, capsys):
    rc = main(["--budget", "128000", "analyze", convo_file])
    assert rc == 0  # under budget
    out = capsys.readouterr().out
    assert "system" in out and "history" in out
    assert "est. tokens" in out


def test_analyze_json(convo_file, capsys):
    rc = main(["analyze", convo_file, "--format", "json"])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["message_count"] == 3
    assert info["over_budget"] is False
    assert "usage_est" in info and "top_consumers" in info


def test_analyze_over_budget_exit_code(convo_file, capsys):
    rc = main(["--budget", "10", "analyze", convo_file, "--format", "json"])
    assert rc == 2
    info = json.loads(capsys.readouterr().out)
    assert info["over_budget"] is True
    assert info["warnings"]


def test_fit_writes_files(convo_file, tmp_path, capsys):
    out = str(tmp_path / "fitted.json")
    rep = str(tmp_path / "report.json")
    rc = main(["--budget", "100", "fit", convo_file, "--out", out, "--report", rep])
    assert rc == 2  # tiny budget: strategies can't shrink prose -> warns
    fitted = json.loads(open(out, encoding="utf-8").read())
    assert isinstance(fitted, list)
    report = json.loads(open(rep, encoding="utf-8").read())
    assert "total_before" in report
    assert "est. tokens" in capsys.readouterr().out


def test_fit_over_budget_exit_code(tmp_path, capsys):
    msgs = [{"role": "user", "content": "x" * 50_000}]
    p = tmp_path / "big.json"
    p.write_text(json.dumps(msgs), encoding="utf-8")
    rc = main(["--budget", "50", "fit", str(p)])
    assert rc == 2  # warns, can't fully fit


def test_estimate_text(capsys):
    rc = main(["estimate", "hello world"])
    assert rc == 0
    assert "est. tokens" in capsys.readouterr().out


def test_estimate_file(tmp_path, capsys):
    p = tmp_path / "t.txt"
    p.write_text("a" * 400, encoding="utf-8")
    rc = main(["estimate", "--file", str(p)])
    assert rc == 0
    assert "100" in capsys.readouterr().out  # 400 chars / 4


def test_missing_file_errors_cleanly(capsys):
    rc = main(["analyze", "/does/not/exist.json"])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_bad_budgets_flag_errors(capsys):
    rc = main(["--budgets", "bogus", "analyze", "/does/not/exist.json"])
    assert rc == 1


def test_parser_has_subcommands():
    p = build_parser()
    for cmd in ("analyze", "fit", "estimate"):
        assert cmd in p._subparsers._group_actions[0].choices
