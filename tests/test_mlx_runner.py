from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from cv_rag.answer import mlx_runner
from cv_rag.shared.errors import GenerationError


class _FakeStderr:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


class _FakeProcess:
    def __init__(self, chunks: list[str], returncode: int, stderr_text: str = "") -> None:
        self.stdout = iter(chunks)
        self.stderr = _FakeStderr(stderr_text)
        self.returncode: int | None = None
        self._final_returncode = returncode

    def wait(self) -> int:
        self.returncode = self._final_returncode
        return self._final_returncode


def test_mlx_generate_success_strips_output_and_builds_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="  ok\n", stderr="")

    monkeypatch.setattr(mlx_runner.subprocess, "run", fake_run)

    result = mlx_runner.mlx_generate(
        model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        prompt="hello",
        max_tokens=64,
        temperature=0.2,
        top_p=0.9,
    )

    assert result == "ok"
    command = captured["command"]
    assert command[:4] == [sys.executable, "-m", "mlx_lm", "generate"]
    assert "--model" in command
    assert "--prompt" in command
    assert "--max-tokens" in command
    assert "--temp" in command
    assert "--top-p" in command
    assert "--verbose" in command
    assert "False" in command
    assert captured["kwargs"] == {"check": False, "capture_output": True, "text": True}


def test_mlx_generate_includes_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mlx_runner.subprocess, "run", fake_run)

    mlx_runner.mlx_generate(
        model="m",
        prompt="p",
        max_tokens=10,
        temperature=0.1,
        top_p=0.95,
        seed=123,
    )

    command = captured["command"]
    assert "--seed" in command
    seed_index = command.index("--seed")
    assert command[seed_index + 1] == "123"


def test_mlx_generate_missing_runtime_raises_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        raise FileNotFoundError

    monkeypatch.setattr(mlx_runner.subprocess, "run", fake_run)

    with pytest.raises(GenerationError, match=r"runtime .* not found"):
        mlx_runner.mlx_generate(
            model="m",
            prompt="p",
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
        )


def test_mlx_generate_nonzero_with_detail_raises_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="", stderr="bad")

    monkeypatch.setattr(mlx_runner.subprocess, "run", fake_run)

    with pytest.raises(GenerationError, match="bad"):
        mlx_runner.mlx_generate(
            model="m",
            prompt="p",
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
        )


def test_mlx_generate_nonzero_without_detail_raises_exit_code_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=2, stdout="", stderr="")

    monkeypatch.setattr(mlx_runner.subprocess, "run", fake_run)

    with pytest.raises(GenerationError, match="exit code 2"):
        mlx_runner.mlx_generate(
            model="m",
            prompt="p",
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
        )


def test_mlx_generate_empty_output_raises_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="\n\n", stderr="")

    monkeypatch.setattr(mlx_runner.subprocess, "run", fake_run)

    with pytest.raises(GenerationError, match="empty answer"):
        mlx_runner.mlx_generate(
            model="m",
            prompt="p",
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
        )


def test_mlx_generate_stream_success_yields_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _FakeProcess(chunks=["a", "b"], returncode=0)

    monkeypatch.setattr(mlx_runner.subprocess, "Popen", fake_popen)

    chunks = list(
        mlx_runner.mlx_generate_stream(
            model="m",
            prompt="p",
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
        )
    )

    assert chunks == ["a", "b"]
    command = captured["command"]
    assert command[:4] == [sys.executable, "-m", "mlx_lm", "generate"]
    assert "--model" in command
    assert "--prompt" in command
    assert captured["kwargs"]["stdout"] == mlx_runner.subprocess.PIPE
    assert captured["kwargs"]["stderr"] == mlx_runner.subprocess.PIPE
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["bufsize"] == 1


def test_mlx_generate_stream_missing_runtime_raises_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        raise FileNotFoundError

    monkeypatch.setattr(mlx_runner.subprocess, "Popen", fake_popen)

    with pytest.raises(GenerationError, match=r"runtime .* not found"):
        _ = list(
            mlx_runner.mlx_generate_stream(
                model="m",
                prompt="p",
                max_tokens=10,
                temperature=0.1,
                top_p=0.95,
            )
        )


def test_mlx_generate_stream_nonzero_exit_raises_after_yields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(chunks=["a", "b"], returncode=3, stderr_text="oops")

    monkeypatch.setattr(mlx_runner.subprocess, "Popen", fake_popen)

    stream = mlx_runner.mlx_generate_stream(
        model="m",
        prompt="p",
        max_tokens=10,
        temperature=0.1,
        top_p=0.95,
    )

    assert next(stream) == "a"
    assert next(stream) == "b"
    with pytest.raises(GenerationError, match="oops"):
        next(stream)


def test_sse_event_multiline_string_formatting() -> None:
    event = mlx_runner.sse_event("chunk", "x\ny")
    assert event == "event: chunk\ndata: x\ndata: y\n\n"


def test_sse_event_json_payload_formatting() -> None:
    event = mlx_runner.sse_event("meta", {"k": "v"})
    assert event.startswith("event: meta\n")
    assert "data: {\"k\": \"v\"}\n\n" in event
