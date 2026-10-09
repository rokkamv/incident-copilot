import importlib.util
import pathlib
import types

import anthropic
import httpx
import pytest

STAGE_PATH = pathlib.Path(__file__).resolve().parent.parent / "stages" / "01_first_call.py"


def _load_stage():
    spec = importlib.util.spec_from_file_location("stage_01_first_call", STAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stage():
    return _load_stage()


def _fake_response(
    *,
    stop_reason="end_turn",
    text="1. Cause A\n2. Cause B\n3. Cause C",
    input_tokens=100,
    output_tokens=50,
):
    return types.SimpleNamespace(
        stop_reason=stop_reason,
        content=[types.SimpleNamespace(type="text", text=text)],
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _FakeMessages:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return self._response


class _FakeClient:
    def __init__(self, messages):
        self.messages = messages


def _install_fake_client(monkeypatch, stage, response=None, exc=None):
    monkeypatch.setattr(
        stage.anthropic,
        "Anthropic",
        lambda: _FakeClient(_FakeMessages(response=response, exc=exc)),
    )


def _fake_api_error(exc_type, status_code):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request, json={"type": "error"})
    return exc_type("boom", response=response, body=None)


def test_happy_path(monkeypatch, stage, capsys):
    _install_fake_client(monkeypatch, stage, response=_fake_response())

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low")

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Cause A" in out
    assert "input tokens" in out
    assert "output tokens" in out


def test_refusal(monkeypatch, stage, capsys):
    _install_fake_client(monkeypatch, stage, response=_fake_response(stop_reason="refusal"))

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low")

    assert exit_code != 0
    assert "declined" in capsys.readouterr().err


def test_max_tokens_warns_but_still_prints(monkeypatch, stage, capsys):
    _install_fake_client(monkeypatch, stage, response=_fake_response(stop_reason="max_tokens"))

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low")

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
    def _should_not_be_called():
        raise AssertionError("Anthropic client should not be constructed for an unknown model")

    monkeypatch.setattr(stage.anthropic, "Anthropic", _should_not_be_called)

    exit_code = stage.run("ERROR boom", model="not-a-real-model", effort="low")

    assert exit_code != 0


def test_authentication_error(monkeypatch, stage, capsys):
    _install_fake_client(
        monkeypatch, stage, exc=_fake_api_error(anthropic.AuthenticationError, 401)
    )

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low")

    assert exit_code != 0
    assert "API key" in capsys.readouterr().err


def test_rate_limit_error(monkeypatch, stage, capsys):
    _install_fake_client(
        monkeypatch, stage, exc=_fake_api_error(anthropic.RateLimitError, 429)
    )

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low")

    assert exit_code != 0
    assert "rate limited" in capsys.readouterr().err


def test_api_status_error(monkeypatch, stage, capsys):
    _install_fake_client(
        monkeypatch, stage, exc=_fake_api_error(anthropic.InternalServerError, 500)
    )

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low")

    assert exit_code != 0
    assert "500" in capsys.readouterr().err


def test_api_connection_error(monkeypatch, stage, capsys):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    _install_fake_client(
        monkeypatch, stage, exc=anthropic.APIConnectionError(request=request)
    )

    exit_code = stage.run("ERROR boom", model="claude-haiku-5-5", effort="low")

    assert exit_code != 0
    assert "network" in capsys.readouterr().err
