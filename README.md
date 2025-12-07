# SynQc-TDS-controller

Synchronized Quantum Circuits Temporal Dynamics Series controller – a small, session-based control stack for drive–probe–drive (DPD) experiments, with a local simulator, a pluggable hardware abstraction layer, and a browser control panel.

## Project layout

- `synqc_tds/` – FastAPI app (`api.py`), execution engine (`engine.py`), and hardware abstraction layer (`hardware_backends.py`, plus the `models.py` schema definitions).
- `synqc_tds_super_backend.py` – Uvicorn entrypoint kept for backwards compatibility.
- `static/index.html` – Lightweight HTML controller that exercises the `/run` endpoint.
- `requirements.txt` – Python dependencies for the API service.
- `Dockerfile` – Slim runtime image for containerized deployments.

## Getting started (local)

Create and activate a virtual environment, install dependencies, and launch the API:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn synqc_tds_super_backend:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in a browser to use the embedded HTML controller. The app also serves health information at `/health` and accepts run submissions at `/run`.

### Sample request payload

POST the following JSON to `/run` to exercise the mock backend without the UI:

```json
{
  "config": {
    "hardware_target": "mock-local",
    "sequence_id": "ramsey-demo",
    "cycles": 4,
    "shot_limit": 2048,
    "metadata": {}
  },
  "session": {
    "session_id": "example-session-001",
    "seed": 42,
    "cache": {}
  },
  "mode": "run"
}
```

## Docker usage

Build and run the containerized backend:

```bash
docker build -t synqc-tds .
docker run --rm -p 8000:8000 synqc-tds
```

Then open `http://localhost:8000` and trigger runs from the control panel. The image exposes port 8000 by default and uses the same `uvicorn synqc_tds_super_backend:app` entrypoint.

## Next steps

- Add more realistic hardware backends and move credentials to environment variables.
- Persist session state, run history, and KPI exports to disk or a database.
- Harden authentication (API keys or OAuth) before exposing the service publicly.
- Expand the HTML controller to visualize KPI traces over time and stream updates via WebSockets.
