from __future__ import annotations

import logging
import subprocess

from cv_rag.exceptions import GenerationError

logger = logging.getLogger(__name__)


def mlx_generate(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> str:
    """Run mlx_lm.generate subprocess and return the generated text.

    Raises GenerationError on failure (not found, non-zero exit, empty output).
    """
    command = [
        "mlx_lm.generate",
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
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise GenerationError("`mlx_lm.generate` was not found in PATH.") from None

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            raise GenerationError(f"MLX generation failed: {detail}")
        raise GenerationError(f"MLX generation failed with exit code {result.returncode}.")

    answer_text = result.stdout.strip()
    if not answer_text:
        raise GenerationError("MLX generation returned an empty answer.")
    return answer_text
