import importlib.util
import pathlib

import pytest

from copilot import llm

STAGE_PATH = pathlib.Path(__file__).resolve().parent.parent / "stages" / "01_first_call.py"


def _load_stage():
    spec = importlib.util.spec_from_file_location("stage_01_first_call", STAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stage():
    return _load_stage()


def _fake_llm_response(**overrides):
    defaults = {
        "text": "1. Cause A\n2. Cause B\n3. Cause C",
        "tool_input": None,
        "refused": False,
        "truncated": False,
        "input_tokens": 100,
        "output_tokens": 50,
        "stop_reason": "end_turn",
    }
    return llm.LLMResponse(**{**defaults, **overrides})


def _install_fake_complete(monkeypatch, stage, response=None, exc=None):
    def _complete(*args, **kwargs):
        if exc:
            raise exc
        return response

    monkeypatch.setattr(stage.llm, "complete", _complete)


def test_happy_path(monkeypatch, stage, capsys):
    _install_fake_complete(monkeypatch, stage, response=_fake_llm_response())

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low", provider="anthropic")

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Cause A" in out
    assert "input tokens" in out
    assert "output tokens" in out


def test_refusal(monkeypatch, stage, capsys):
    _install_fake_complete(monkeypatch, stage, response=_fake_llm_response(refused=True, stop_reason="refusal"))

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low", provider="anthropic")

    assert exit_code != 0
    assert "declined" in capsys.readouterr().err


def test_max_tokens_warns_but_still_prints(monkeypatch, stage, capsys):
    _install_fake_complete(
        monkeypatch, stage, response=_fake_llm_response(truncated=True, stop_reason="max_tokens")
    )

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low", provider="anthropic")

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cut off" in captured.err
    assert "Cause A" in captured.out


def test_missing_cli_arg_exits_nonzero(stage, capsys):
    with pytest.raises(SystemExit) as exc_info:
        stage.main([])

    assert exc_info.value.code != 0
    assert "usage" in capsys.readouterr().err.lower()


def test_unknown_model_fails_before_any_api_call(monkeypatch, stage):
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("llm.complete should not be called for an unknown model")

    monkeypatch.setattr(stage.llm, "complete", _should_not_be_called)

    exit_code = stage.run("ERROR boom", model="not-a-real-model", effort="low", provider="anthropic")

    assert exit_code != 0


def test_authentication_error(monkeypatch, stage, capsys):
    _install_fake_complete(monkeypatch, stage, exc=llm.LLMAuthenticationError("boom"))

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low", provider="anthropic")

    assert exit_code != 0
    assert "API key" in capsys.readouterr().err


def test_rate_limit_error(monkeypatch, stage, capsys):
    _install_fake_complete(monkeypatch, stage, exc=llm.LLMRateLimitError("boom"))

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low", provider="anthropic")

    assert exit_code != 0
    assert "rate limited" in capsys.readouterr().err


def test_api_status_error(monkeypatch, stage, capsys):
    _install_fake_complete(monkeypatch, stage, exc=llm.LLMAPIStatusError("boom", 500))

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low", provider="anthropic")

    assert exit_code != 0
    assert "500" in capsys.readouterr().err


def test_connection_error(monkeypatch, stage, capsys):
    _install_fake_complete(monkeypatch, stage, exc=llm.LLMConnectionError("boom"))

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low", provider="anthropic")

    assert exit_code != 0
    assert "network" in capsys.readouterr().err


def test_different_provider_without_model_fails_before_any_call(monkeypatch, stage):
    monkeypatch.setattr(stage.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(stage.settings, "llm_model", "claude-haiku-5-5")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("llm.complete should not be called")

    monkeypatch.setattr(stage.llm, "complete", _should_not_be_called)

    exit_code = stage.run("ERROR boom", model=None, effort="low", provider="openai")

    assert exit_code != 0


def test_main_resolves_effort_from_cli_flag_first(monkeypatch, stage):
    captured = {}
    monkeypatch.setattr(stage, "run", lambda *a, **k: captured.setdefault("effort", a[2]) or 0)
    monkeypatch.setattr(stage.settings, "llm_effort", "high")

    with pytest.raises(SystemExit):
        stage.main(["--effort", "medium", "ERROR boom"])

    assert captured["effort"] == "medium"


def test_main_resolves_effort_from_env_when_no_cli_flag(monkeypatch, stage):
    captured = {}
    monkeypatch.setattr(stage, "run", lambda *a, **k: captured.setdefault("effort", a[2]) or 0)
    monkeypatch.setattr(stage.settings, "llm_effort", "high")

    with pytest.raises(SystemExit):
        stage.main(["ERROR boom"])

    assert captured["effort"] == "high"


def test_main_resolves_effort_to_low_when_nothing_configured(monkeypatch, stage):
    captured = {}
    monkeypatch.setattr(stage, "run", lambda *a, **k: captured.setdefault("effort", a[2]) or 0)
    monkeypatch.setattr(stage.settings, "llm_effort", None)

    with pytest.raises(SystemExit):
        stage.main(["ERROR boom"])

    assert captured["effort"] == "low"


def test_effort_noop_note_printed_for_openai(monkeypatch, stage, capsys):
    # assert_known_model reads from the real, user-editable data/pricing.json -
    # this test isn't about pricing validation, so stub it out rather than
    # depend on that file's current (changeable) content.
    monkeypatch.setattr(stage, "assert_known_model", lambda provider, model: None)
    _install_fake_complete(monkeypatch, stage, response=_fake_llm_response())

    stage.run("ERROR boom", model="gpt-test", effort="medium", provider="openai")

    assert "no OpenAI equivalent" in capsys.readouterr().err
