from __future__ import annotations

import typer
from rich.console import Console


def run_serve_command(*, console: Console, host: str, port: int, reload: bool) -> None:
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Web dependencies not installed. Run: uv sync --extra web[/red]")
        raise typer.Exit(code=1) from None

    uvicorn.run("cv_rag.interfaces.api.app:create_app", host=host, port=port, reload=reload, factory=True)
