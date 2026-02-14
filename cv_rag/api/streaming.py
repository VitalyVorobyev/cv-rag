from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Generator

from cv_rag.exceptions import GenerationError


def mlx_generate_stream(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> Generator[str, None, None]:
    """Yield text chunks from MLX generation subprocess.

    Falls back to returning the full output at once if line-level
    streaming is not available (MLX may buffer stdout).
    """
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
        "--verbose",
        "False",
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        raise GenerationError("Python runtime for MLX generation was not found in PATH.") from None

    assert process.stdout is not None

    yield from process.stdout

    process.wait()
    if process.returncode != 0:
        stderr_text = ""
        if process.stderr:
            stderr_text = process.stderr.read().strip()
        raise GenerationError(f"MLX generation failed: {stderr_text}")


def sse_event(event: str, data: object) -> str:
    """Format a server-sent event."""
    payload = data if isinstance(data, str) else json.dumps(data, default=str)
    lines = payload.split("\n")
    result = f"event: {event}\n"
    for line in lines:
        result += f"data: {line}\n"
    result += "\n"
    return result
