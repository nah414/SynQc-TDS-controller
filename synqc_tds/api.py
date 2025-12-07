from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .engine import SynQcEngine
from .hardware_backends import BackendError
from .models import RunRequest, RunResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="SynQc Temporal Dynamics Series",
        version="0.3.0",
        description="SynQc TDS unified controller + engine backend.",
    )

    engine = SynQcEngine()

    # Static HTML / JS controller
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "static"

    if static_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the main SynQc controller UI."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        # Fallback if HTML isn't present yet
        return "<h1>SynQc TDS API</h1><p>UI not found. Put index.html in /static.</p>"

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/run", response_model=RunResponse)
    async def run_experiment(req: RunRequest) -> RunResponse:
        try:
            kpis = engine.run(req.config, req.session, mode=req.mode)
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 – top-level API guard
            raise HTTPException(status_code=500, detail="Internal SynQc error") from exc

        return RunResponse(
            kpis=kpis,
            backend_name=req.config.hardware_target,
            mode=req.mode,
        )

    return app


# Uvicorn entrypoint
app = create_app()
