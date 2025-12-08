# SynQc-TDS-controller

Synchronized Quantum Circuits Temporal Dynamics Series controller – a small, session-based
control stack for drive–probe–drive (DPD) experiments, with a local simulator, a pluggable
hardware abstraction layer, and a browser control panel.

## 1. Layout

- `synqc_tds_super_backend.py` – FastAPI backend, session engine, KPIs, state store.
- `synqc_agent.py` – LLM-based advisory agent for experiment recommendations.
- `adac1680-...html` – original frontend (Interstellar Control Console).
- `synqc_frontend_improved.html` – polished frontend with improved UX.

## 2. Backend quickstart (local simulator)

Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install fastapi uvicorn[standard] pydantic numpy python-dotenv orjson
```

Run the backend:

```bash
uvicorn synqc_tds_super_backend:app --host 127.0.0.1 --port 8000
```

Or use the convenience launcher:

```bash
python synqc_tds_super_backend.py
```

## 3. Smoke test (quick integration check)

A small smoke test script is included to verify the basic API endpoints 
(health, create session, dry-run, logs). This is useful for quick local 
validation and CI sanity checks.

**Available scripts:**
- `scripts/smoke_test.ps1` – PowerShell version (for Windows or pwsh)
- `scripts/smoke_test.sh` – POSIX shell version (for Linux / macOS / WSL)

**Run locally (PowerShell):**

```powershell
# from repository root, ensure backend is running
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1 -ApiBase "http://localhost:8000/api/v1/synqc"
```

**Run locally (bash / Linux / macOS / WSL):**

```bash
# start the backend in the background, then run the smoke test
python -m uvicorn synqc_tds_super_backend:app --host 127.0.0.1 --port 8000 &
chmod +x ./scripts/smoke_test.sh
./scripts/smoke_test.sh -b "http://localhost:8000/api/v1/synqc"
```

**CI/CD integration:**

A GitHub Actions workflow is included under `.github/workflows/smoke-test.yml` 
to run the smoke test on push. The job will:
1. Checkout the repository
2. Install Python dependencies
3. Start the backend with uvicorn
4. Wait for the `/health` endpoint to be healthy
5. Run the smoke test script
6. Upload server logs on failure

Note: The smoke test requires the backend to be reachable at the given API base URL 
and expects JSON responses as implemented by `synqc_tds_super_backend.py`.

## 4. Frontend

Open the HTML file in a browser (or serve it via a static server) and point it to your 
running backend:

```
http://localhost:8080/path/to/adac1680-...html?apiBase=http://localhost:8000/api/v1/synqc
```

Or edit the default API base in the HTML file. The frontend provides:
- Session management
- Configuration controls (hardware, probe parameters, adaptive rules, objectives)
- Real-time KPI display
- Run logging and telemetry
- Data export (JSON, CSV, notebook cell format)
- LLM agent suggestions (if `OPENAI_API_KEY` is set on the backend)

## 5. Environment variables (optional)

Configure the backend via environment variables (or a `.env` file):

```bash
# API configuration
SYNQC_API_PREFIX=/api/v1/synqc
SYNQC_STATE_DIR=./synqc_state
SYNQC_ALLOWED_ORIGINS=*

# Safety limits
SYNQC_MAX_PROBE_STRENGTH=0.5
SYNQC_MAX_PROBE_DURATION_NS=5000
SYNQC_MAX_SHOTS_PER_RUN=200000

# Session persistence
SYNQC_SESSION_FLUSH_INTERVAL_SEC=1.0
SYNQC_ENABLE_BACKGROUND_FLUSH=1

# Server launch
SYNQC_HOST=127.0.0.1
SYNQC_PORT=8000
SYNQC_RELOAD=0

# LLM agent (optional)
OPENAI_API_KEY=sk-...
```

## 6. Performance & optimization

The backend includes several optimizations:
- **Thread-local RNG:** Avoids allocating a new random generator per call.
- **Optional fast JSON:** Uses `orjson` when available; falls back to stdlib `json`.
- **In-memory session updates:** High-frequency operations (telemetry polling) avoid disk I/O.
- **Background flusher:** Batches session writes to disk on a configurable interval.
- **Async run launcher:** CPU-bound simulation runs in a thread pool, keeping the event loop responsive.

See `PERFORMANCE_OPTIMIZATION_SUMMARY.md` for detailed optimization notes.

## 7. Architecture & safety

- **No shell execution or dynamic code eval** — all inputs are validated.
- **Bounded parameters** — probe strength, duration, and shot budgets are clamped and validated.
- **Belt-and-braces validation** — agent suggestions are validated before being returned; clamps are re-enforced on apply.
- **Graceful degradation** — agent and fast JSON are optional; backend continues if they're unavailable.
- **CORS configured** — credentials are disabled when using wildcard origins for safety.

## 8. License & attribution

Developed by **eVision Enterprises**.

This project uses:
- FastAPI & Pydantic for HTTP API & validation
- NumPy for scientific computing
- OpenAI SDK (optional, for LLM agent)
- Optional orjson for performance
