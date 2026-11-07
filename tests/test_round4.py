"""Round 4: what a stranger gets from `pip install ctxfit` (CLI behaviour, packaging, Windows-ish)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import ctxfit
from ctxfit import cli
from ctxfit.cli import main
from ctxfit.messages import Message, normalize, repair_pairing

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(ctxfit.__file__).parent / "demo"


def run_cli(*args: str, env: dict[str, str] | None = None, stdin: str | None = None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "ctxfit", *args],
        capture_output=True,
        input=stdin.encode() if stdin is not None else None,
        env=e,
        timeout=120,
    )


@pytest.fixture
def convo(tmp_path):
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "go"}]
    for i in range(3):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "line\n" * 4000})
    p = tmp_path / "c.json"
    p.write_text(json.dumps(msgs), encoding="utf-8")
    return str(p)


# --- first minute -------------------------------------------------------------------------


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"ctxfit {ctxfit.__version__}"


def test_python_dash_m_works():
    r = run_cli("--version")
    assert r.returncode == 0 and r.stdout.decode().strip() == f"ctxfit {ctxfit.__version__}"


def test_demo_runs_and_keeps_every_needle(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert out.count("kept:") == 3 and "LOST" not in out and "within budget" in out


def test_demo_json(capsys):
    assert main(["demo", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert all(data["needles_kept"].values()) and data["savings_pct"] > 80


def test_demo_is_deterministic(capsys):
    main(["demo", "--json"])
    a = capsys.readouterr().out
    main(["demo", "--json"])
    assert a == capsys.readouterr().out


# --- plain-English errors, never a traceback ------------------------------------------------


@pytest.mark.parametrize(
    "content,needle",
    [
        (None, "there is no file"),
        (b"", "is empty"),
        (b"   \n", "is empty"),
        (b"[", "not valid JSON"),
        (b'["caf\xe9"]', "not UTF-8"),
        (b'{"a": 1}', "JSON array of messages"),
        (b"[" * 5000 + b"]" * 5000, "ctxfit: error"),
    ],
)
def test_bad_input_files_give_a_sentence(tmp_path, capsys, content, needle):
    p = tmp_path / "x.json"
    if content is not None:
        p.write_bytes(content)
    assert main(["analyze", str(p)]) == 1
    err = capsys.readouterr().err
    assert needle in err and "Traceback" not in err and err.startswith("ctxfit: error:")


def test_folder_as_input(tmp_path, capsys):
    assert main(["analyze", str(tmp_path)]) == 1
    assert "is a folder" in capsys.readouterr().err


def test_filename_escapes_are_printable(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "a\x1b[31m‮b.json")]) == 1
    err = capsys.readouterr().err
    assert "\x1b" not in err and "‮" not in err and "\\u001b" in err


def test_orphan_tool_result_is_explained(tmp_path, capsys):
    p = tmp_path / "o.json"
    p.write_text(json.dumps([{"role": "user", "content": "a"},
                             {"role": "tool", "tool_call_id": "zz", "content": "x"}]))  # fmt: skip
    assert main(["fit", str(p)]) == 1
    err = capsys.readouterr().err
    assert "--repair-pairing" in err and "Traceback" not in err and err.count("\n") == 1


def test_repair_pairing_flag(tmp_path, capsys):
    p = tmp_path / "o.json"
    p.write_text(json.dumps([{"role": "user", "content": "a"},
                             {"role": "tool", "tool_call_id": "zz", "content": "x" * 50}]))  # fmt: skip
    out = tmp_path / "f.json"
    assert main(["fit", str(p), "--repair-pairing", "--out", str(out), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["pairing_repaired"] == 1 and "zz" in rep["pairing_repairs"][0]
    assert json.loads(out.read_text(encoding="utf-8")) == [{"role": "user", "content": "a"}]


def test_malformed_anthropic_block_is_a_clean_error(tmp_path, capsys):
    p = tmp_path / "n.json"
    p.write_text(json.dumps([{"role": "user", "content": [{"type": "text", "text": "a"}, None]}]))
    for flavor in ("anthropic",):
        assert main(["--flavor", flavor, "analyze", str(p)]) == 1
    err = capsys.readouterr().err
    assert "content block #1" in err and "Traceback" not in err


def _force_failure(monkeypatch, convo, exc):
    """Make the chosen subcommand raise ``exc`` after a normal parse."""

    def boom(args):
        raise exc

    ns = cli.build_parser().parse_args(["analyze", convo])
    ns.func = boom
    parser = type("P", (), {"parse_args": lambda self, a: ns})
    monkeypatch.setattr(cli, "build_parser", lambda: parser())


def test_unexpected_exception_is_reported_not_dumped(monkeypatch, capsys, convo):
    _force_failure(monkeypatch, convo, KeyError("oops"))
    monkeypatch.delenv("CTXFIT_DEBUG", raising=False)
    assert main(["analyze", convo]) == 1
    err = capsys.readouterr().err
    assert "bug in ctxfit" in err and "KeyError" in err and "Traceback" not in err


def test_debug_env_shows_the_traceback(monkeypatch, convo):
    _force_failure(monkeypatch, convo, KeyError("oops"))
    monkeypatch.setenv("CTXFIT_DEBUG", "1")
    with pytest.raises(KeyError):
        main(["analyze", convo])


def test_broken_pipe_is_silent(monkeypatch, convo, capsys):
    _force_failure(monkeypatch, convo, BrokenPipeError(32, "Broken pipe"))
    assert main(["analyze", convo]) == 0
    assert "Broken pipe" not in capsys.readouterr().err


def test_real_pipe_closed_early(convo):
    cmd = f"{sys.executable} -m ctxfit analyze {convo} | head -c 10 >/dev/null"
    if sys.platform == "win32":
        pytest.skip("POSIX shell pipe")
    r = subprocess.run(["bash", "-c", cmd + "; exit ${PIPESTATUS[0]}"], capture_output=True)
    assert (
        r.returncode in (0, 2) and b"Broken pipe" not in r.stderr and b"Traceback" not in r.stderr
    )


# --- json / stdout discipline ---------------------------------------------------------------


def test_fit_json_is_only_json_on_stdout(convo, capsys):
    assert main(["--budget", "3000", "fit", convo, "--json"]) in (0, 2)
    cap = capsys.readouterr()
    data = json.loads(cap.out)
    assert "total_before" in data and cap.err == ""


def test_fit_json_with_out_dash_refused(convo, capsys):
    assert main(["fit", convo, "--json", "--out", "-"]) == 1
    assert "standard output" in capsys.readouterr().err


def test_fit_out_dash_keeps_stdout_clean(convo):
    r = run_cli("--budget", "3000", "fit", convo, "--out", "-")
    json.loads(r.stdout.decode())  # stdout is the data, nothing else
    assert b"est. tokens" in r.stderr


def test_estimate_json_and_stdin(capsys):
    assert main(["estimate", "hello world", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["chars"] == 11 and d["tokens"] >= 2
    r = run_cli("estimate", "-", "--json", stdin="hello world")
    assert json.loads(r.stdout.decode())["chars"] == 11


def test_analyze_json_flag(convo, capsys):
    assert main(["analyze", convo, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["message_count"] == 8


# --- odd consoles and files (Windows code pages, LC_ALL=C) ----------------------------------


@pytest.mark.parametrize("enc", ["cp1252", "ascii", "cp437"])
def test_legacy_console_encodings_do_not_crash(convo, enc):
    r = run_cli("--budget", "3000", "fit", convo, env={"PYTHONIOENCODING": enc})
    assert r.returncode in (0, 2), r.stderr
    assert b"Traceback" not in r.stderr and b"est. tokens" in r.stdout


def test_legacy_console_demo(tmp_path):
    r = run_cli("demo", env={"PYTHONIOENCODING": "ascii"})
    assert r.returncode == 0 and b"kept:" in r.stdout and b"Traceback" not in r.stderr


def test_out_dash_on_ascii_console_stays_valid_json(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps([{"role": "user", "content": "café 你好"}]), encoding="utf-8")
    r = run_cli("fit", str(p), "--out", "-", env={"PYTHONIOENCODING": "ascii"})
    assert json.loads(r.stdout.decode("ascii"))[0]["content"] == "café 你好"


def test_path_with_spaces_and_unicode(tmp_path, convo):
    d = tmp_path / "my folder é"
    d.mkdir()
    dst = d / "con vo.json"
    shutil.copy(convo, dst)
    out = d / "out file.json"
    assert main(["--budget", "3000", "fit", str(dst), "--out", str(out)]) in (0, 2)
    assert json.loads(out.read_text(encoding="utf-8"))


def test_crlf_and_bom_input(tmp_path, capsys):
    p = tmp_path / "w.json"
    p.write_bytes(b'\xef\xbb\xbf[\r\n {"role": "user",\r\n "content": "hi"}\r\n]\r\n')
    assert main(["analyze", str(p), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["message_count"] == 1


def test_replace_retries_when_target_is_locked(monkeypatch, tmp_path):
    real = os.replace
    calls = []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(5, "Access is denied")
        return real(a, b)

    monkeypatch.setattr(cli.os, "replace", flaky)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    target = tmp_path / "t.json"
    cli._write(str(target), "[]\n")
    assert target.read_text() == "[]\n" and len(calls) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_replace_gives_up_with_a_sentence(monkeypatch, tmp_path, capsys, convo):
    def locked(a, b):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(cli.os, "replace", locked)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    assert main(["fit", convo, "--out", str(tmp_path / "t.json")]) == 1
    assert "not allowed to write" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a non-root POSIX user",
)
def test_read_only_folder(tmp_path, convo, capsys):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        assert main(["fit", convo, "--out", str(ro / "o.json")]) == 1
        assert "not allowed to write" in capsys.readouterr().err
    finally:
        ro.chmod(0o755)


def test_out_into_missing_folder(tmp_path, convo, capsys):
    assert main(["fit", convo, "--out", str(tmp_path / "nope" / "o.json")]) == 1
    assert "folder does not exist" in capsys.readouterr().err


# --- replay globbing ------------------------------------------------------------------------


def test_replay_bare_glob_in_cwd(tmp_path, monkeypatch):
    from ctxfit.replay import expand_paths

    shutil.copy(DEMO / "session-a.jsonl", tmp_path / "a.jsonl")
    (tmp_path / "sub").mkdir()
    shutil.copy(DEMO / "session-b.jsonl", tmp_path / "sub" / "b.jsonl")
    monkeypatch.chdir(tmp_path)
    assert expand_paths(["*.jsonl"]) == ["a.jsonl"]
    assert expand_paths(["./*.jsonl"]) == ["./a.jsonl"]
    assert sorted(expand_paths(["**/*.jsonl"])) == ["a.jsonl", os.path.join("sub", "b.jsonl")]
    assert expand_paths(["sub/*.jsonl"]) == [os.path.join("sub", "b.jsonl")]


def test_replay_crlf_equals_lf(tmp_path):
    from ctxfit.replay import dumps, replay

    raw = (DEMO / "session-c.jsonl").read_bytes().replace(b"\r\n", b"\n")
    (tmp_path / "lf").mkdir()
    (tmp_path / "crlf").mkdir()
    (tmp_path / "lf" / "s.jsonl").write_bytes(raw)
    (tmp_path / "crlf" / "s.jsonl").write_bytes(raw.replace(b"\n", b"\r\n"))
    a = dumps(replay([str(tmp_path / "lf")], redact=True))
    b = dumps(replay([str(tmp_path / "crlf")], redact=True))
    assert a == b


# --- library: pairing repair, bad blocks ----------------------------------------------------


def _pair_convo():
    return [
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "gone", "content": "orphan result"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "fine"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "pending", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ]  # fmt: skip


def test_pairing_strict_is_still_the_default():
    with pytest.raises(ctxfit.PairingBroken, match="pairing='repair'"):
        ctxfit.fit(_pair_convo())


def test_pairing_repair_drops_only_the_unmatched():
    out, rep = ctxfit.fit(_pair_convo(), pairing="repair")
    ids = [m.get("tool_call_id") or [c["id"] for c in m.get("tool_calls", [])] for m in out]
    assert ids == [[], ["a"], "a"]
    assert rep.pairing_repaired == 2 and "pairing='repair'" in rep.summary()
    assert rep.to_dict()["pairing_repaired"] == 2


def test_pairing_repair_noop_on_clean_input():
    clean = _pair_convo()[2:4]
    out, rep = ctxfit.fit(clean, pairing="repair")
    assert rep.pairing_repaired == 0 and "pairing='repair'" not in rep.summary()


def test_pairing_unknown_mode():
    with pytest.raises(ValueError, match="pairing"):
        ctxfit.fit([{"role": "user", "content": "a"}], pairing="nope")


def test_repair_drops_second_user_of_a_repeated_id():
    msgs = normalize(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "one"},
            {"role": "tool", "tool_call_id": "a", "content": "two"},
        ]  # fmt: skip
    )
    kept, dropped = repair_pairing(msgs)
    assert [m.content for m in kept if m.kind == "tool_result"] == ["one"] and len(dropped) == 1


def test_repair_refuses_to_drop_a_message_with_an_image():
    raw = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "r"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}]},
    ]  # fmt: skip
    with pytest.raises(ValueError, match="non-text parts"):
        ctxfit.fit(raw, pairing="repair")


def test_pairing_repair_report_does_not_lose_savings():
    big = _pair_convo()
    big[1]["content"] = "z " * 5000
    _, rep = ctxfit.fit(big, pairing="repair")
    assert rep.total_before > 4000 > rep.total_after  # the dropped orphan counts as saved


@pytest.mark.parametrize("bad", [None, "text", 5, [1]])
def test_bad_anthropic_block_raises_typeerror(bad):
    with pytest.raises(TypeError, match="content block #1"):
        ctxfit.fit(
            [{"role": "user", "content": [{"type": "text", "text": "a"}, bad]}], flavor="anthropic"
        )


def test_summary_does_not_claim_passthrough_is_pinned():
    raw = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}]}]  # fmt: skip
    _, rep = ctxfit.fit(raw)
    text = rep.summary()
    assert "handed back unchanged" in text and "pinned" not in text


def test_message_objects_still_work_with_repair():
    ms = [Message(role="user", content="a")]
    out, rep = ctxfit.fit(ms, pairing="repair")
    assert rep.pairing_repaired == 0


# --- packaging facts a stranger relies on ---------------------------------------------------


def test_py_typed_ships_in_the_package():
    assert (Path(ctxfit.__file__).parent / "py.typed").is_file()


def test_version_is_single_sourced_and_in_changelog():
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in toml and not re.search(r"^version\s*=\s*\"", toml, re.M)
    assert re.fullmatch(r"\d+\.\d+\.\d+", ctxfit.__version__)
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## .*{re.escape(ctxfit.__version__)}", log, re.M)


def test_license_file_matches_metadata():
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "MIT"' in toml
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "Calas Automations Inc." in text


def test_readme_names_only_places_that_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "@decnutz" not in readme  # that npm name does not exist (404 on 2026-11-07)
    assert "https://github.com/calasdan-sketch/ctxfit" in readme


def test_cli_help_lists_every_subcommand():
    r = run_cli("--help")
    text = r.stdout.decode()
    for name in ("analyze", "fit", "mask", "replay", "eval", "badge", "estimate", "demo"):
        assert name in text


def test_piece_estimator_matches_the_per_character_definition():
    import random

    random.seed(7)
    alphabet = list("abcXYZ019 \t\n\r\x0b\x0c\x1c\x1f.,;:{}[]()_-/\\\"'") + list(
        "éüñ你好\U0001f600→…—‮  ​́ß"
    )
    est = ctxfit.get_estimator("pieces")
    import math

    from ctxfit.estimate import _CJK_RE, _DIGITS_RE, _EMOJI_RE, _LONG_WORD_RE, _WORD_RE

    for _ in range(2000):
        t = "".join(random.choice(alphabet) for _ in range(random.randint(1, 80)))
        cjk = len(_CJK_RE.findall(t))
        emoji = len(_EMOJI_RE.findall(t))
        u = _CJK_RE.sub("", t) if cjk else t
        total = float(len(_WORD_RE.findall(u)))
        total += sum(math.ceil(len(w) / 7) - 1 for w in _LONG_WORD_RE.findall(u))
        total += sum(math.ceil(len(d) / 2) for d in _DIGITS_RE.findall(u))
        punct = sum(1 for ch in u if not (ch.isspace() or ch.isalnum() and ch.isascii()))
        total += 0.7 * (punct - emoji) + 2.0 * emoji + 0.6 * u.count("\n") + cjk * 1.5
        assert est._count(t) == max(1, math.ceil(total)), repr(t)


class _FakeEnc:
    def encode(self, text, allowed_special=(), disallowed_special="all"):
        if disallowed_special == "all" and "<|" in text:
            raise ValueError("Encountered text corresponding to disallowed special token")
        return list(text.split())


def _fake_tiktoken(monkeypatch, *, offline=False):
    import types

    mod = types.ModuleType("tiktoken")

    def get_encoding(name):
        if offline:
            raise ConnectionError("no network")
        return _FakeEnc()

    mod.get_encoding = get_encoding
    monkeypatch.setitem(sys.modules, "tiktoken", mod)


def test_tiktoken_special_token_text_does_not_crash_fit(monkeypatch):
    _fake_tiktoken(monkeypatch)
    out, rep = ctxfit.fit(
        [{"role": "user", "content": "web page says <|endoftext|> and <|fim_prefix|>"}],
        estimator="tiktoken",
    )
    assert rep.estimator_name == "tiktoken"


def test_auto_estimator_falls_back_when_tiktoken_cannot_load(monkeypatch):
    _fake_tiktoken(monkeypatch, offline=True)
    assert ctxfit.get_estimator("auto").name == "pieces"
    with pytest.raises(ImportError, match="TIKTOKEN_CACHE_DIR"):
        ctxfit.get_estimator("tiktoken")


def test_cli_with_offline_tiktoken_installed(monkeypatch, convo, capsys):
    _fake_tiktoken(monkeypatch, offline=True)
    assert main(["--budget", "3000", "fit", convo]) in (0, 2)
    assert "[pieces]" in capsys.readouterr().out
    assert main(["--estimator", "tiktoken", "estimate", "hi"]) == 1
    assert "TIKTOKEN_CACHE_DIR" in capsys.readouterr().err
