````markdown
SynQc TDS – Integration Snippets and Placement Guide
====================================================

This file collects all the integration snippets we discussed and tells you
EXACTLY where each one belongs in your repository.

Use this as a patch guide for:

- Wiring the hardware abstraction layer (`hardware_backends.get_backend`) into `SynQcEngine`
- Adding minimal API-key authentication and multi‑user ownership
- Verifying the API contract for runs
- Installing vendor SDKs
- Setting safety limits for real hardware


--------------------------------------------------
1. Wire SynQcEngine to the hardware abstraction layer
--------------------------------------------------

**File:** `synqc_tds_super_backend.py`  
**Placement:** near the other imports and in the `SynQcEngine` class.

### 1.1 Import the registry

Add this at the top of `synqc_tds_super_backend.py`, alongside your other imports:

```python
from hardware_backends import get_backend
```

### 1.2 Implement `SynQcEngine.run` using `get_backend`

Inside your `SynQcEngine` class, replace the old `run` implementation
or add this method:

```python
class SynQcEngine:
    ...

    def run(
        self,
        config: RunConfiguration,
        session: SessionState,
        *,
        mode: str = "run",
    ) -> KpiBundle:
        # Select the correct hardware backend based on config.hardware_target
        backend = get_backend(config.hardware_target)

        dry = (mode == "dryrun")
        sim_kpis = backend.run_experiment(config, session, dry_run=dry)

        # Convert SimKpis -> your existing KpiBundle model
        return KpiBundle(
            fidelity=sim_kpis.fidelity,
            latency_us=sim_kpis.latency_us,
            backaction=sim_kpis.backaction,
            shots_used=sim_kpis.shots_used,
            shot_limit=sim_kpis.shot_limit,
            shots_used_fraction=sim_kpis.shots_used_fraction,
        )
```


--------------------------------------------------
2. Minimal API-key authentication & multi-user ownership
--------------------------------------------------

### 2.1 Auth helper (API key dependency)

**File:** `synqc_tds_super_backend.py`  
**Placement:** near the top, after the FastAPI imports.

Add:

```python
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user(x_api_key: str | None = Depends(api_key_header)) -> str:
    raw = os.getenv("SYNQC_API_KEYS", "")
    if not raw:
        # Auth disabled: everything is anonymous.
        return "anonymous"

    allowed = {k.strip() for k in raw.split(",") if k.strip()}
    if not allowed:
        return "anonymous"

    if not x_api_key or x_api_key not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key
```

At runtime you can configure allowed API keys via:

```bash
export SYNQC_API_KEYS="alice-token,bob-token"
```

The frontend can send the key via the `X-API-Key` header
(the HTML control panel already has an “API key” field).

### 2.2 Session model: add `owner` field

**File:** `synqc_tds_super_backend.py`  
**Placement:** in your `SessionState` Pydantic model.

Modify/extend it to include an owner field:

```python
class SessionState(BaseModel):
    session_id: str
    owner: str | None = None
    ...
```

### 2.3 Attach the user to new/updated sessions

**File:** `synqc_tds_super_backend.py`  
**Placement:** in your `/sessions` upsert endpoint.

Modify your `upsert_session` endpoint to depend on `get_current_user`
and store the owner on the session object:

```python
@app.post("/sessions", ...)
def upsert_session(payload: SessionUpsert, user: str = Depends(get_current_user)):
    ...
    session.owner = user
    ...
```

Optionally, you can also filter `GET /sessions` by `owner` so each API key only
sees its own sessions.


--------------------------------------------------
3. Run API response contract
--------------------------------------------------

**File:** `synqc_tds_super_backend.py`  
**Placement:** in the `/sessions/{id}/run` endpoint.

The frontend expects the run endpoint to return this shape:

```json
{
  "session": { ... },
  "run": { ... }
}
```

In Python terms, that means your endpoint should return something like:

```python
@app.post("/sessions/{session_id}/run")
def launch_run(session_id: str, request: RunRequest, user: str = Depends(get_current_user)):
    ...
    return {
        "session": session_state,
        "run": run_record,
    }
```

Where:

- `session_state` is your updated `SessionState` instance.
- `run_record` has a `run_id`, `mode`, timestamps, and a `kpis` field that
  matches what the frontend uses.


--------------------------------------------------
4. Backend commands (install, run, health)
--------------------------------------------------

These are shell commands, not Python. They belong in your README, a
`docs/` file, or your own personal notes.

### 4.1 Install backend dependencies

```bash
pip install fastapi uvicorn[standard] pydantic numpy python-dotenv
```

### 4.2 Run the backend

```bash
uvicorn synqc_tds_super_backend:app --host 0.0.0.0 --port 8000 --reload
```

### 4.3 Health check

```bash
curl http://localhost:8000/api/v1/synqc/health
```


--------------------------------------------------
5. Vendor SDKs: IBM & Braket
--------------------------------------------------

These are also shell commands. They belong in your README or a
“hardware backends” doc, not in a `.py` file.

### 5.1 IBM Quantum (qiskit-ibm-runtime)

Install:

```bash
pip install qiskit-ibm-runtime
```

Then implement `IbmQpuBackend.run_experiment` in `hardware_backends.py`
using Qiskit Runtime and your preferred backend.

### 5.2 AWS Braket

Install:

```bash
pip install amazon-braket-sdk
```

Then implement `AwsBraketBackend.run_experiment` in `hardware_backends.py`
using `braket.aws.AwsDevice` and your device ARN.


--------------------------------------------------
6. Safety / limits for real lab equipment
--------------------------------------------------

These environment variables control the safety limits that the hardware
backends enforce before touching any real hardware.

They should be set in your shell, `.env` file, systemd unit, or Kubernetes
deployment – **not** inside Python code.

```bash
export SYNQC_MAX_PROBE_STRENGTH=0.4
export SYNQC_MAX_PROBE_DURATION_NS=4000
export SYNQC_MAX_SHOTS_PER_RUN=100000
```

In `hardware_backends.py` these are read as:

```python
MAX_PROBE_STRENGTH: float = float(os.getenv("SYNQC_MAX_PROBE_STRENGTH", "0.5"))
MAX_PROBE_DURATION_NS: int = int(os.getenv("SYNQC_MAX_PROBE_DURATION_NS", "5000"))
MAX_SHOTS_PER_RUN: int = int(os.getenv("SYNQC_MAX_SHOTS_PER_RUN", "200_000"))
```

Any attempt to run with values above those limits will cause the backend
to raise an error instead of silently pushing unsafe configs to hardware.


--------------------------------------------------
7. Summary of where things go
--------------------------------------------------

- `from hardware_backends import get_backend`  
  → Top of `synqc_tds_super_backend.py` (imports section).

- `class SynQcEngine: ... def run(...)`  
  → Inside `synqc_tds_super_backend.py` in the `SynQcEngine` class,
    replacing or adding a `run` method.

- Auth helper (`get_current_user`) and `api_key_header`  
  → `synqc_tds_super_backend.py`, near other FastAPI imports and dependencies.

- `owner: str | None = None` in `SessionState`  
  → `synqc_tds_super_backend.py`, inside `SessionState` Pydantic model.

- `upsert_session(..., user: str = Depends(get_current_user))` and `session.owner = user`  
  → `synqc_tds_super_backend.py`, in the `/sessions` upsert endpoint.

- JSON shape `{ "session": {...}, "run": {...} }`  
  → Contract for `/sessions/{id}/run` response body.

- `pip install ...`, `uvicorn ...`, `curl ...`, `pip install qiskit-ibm-runtime`, `pip install amazon-braket-sdk`, `export SYNQC_MAX_...`  
  → README / docs / shell, not in Python files.

Drop this file into your repo as e.g. `docs/synqc_integration_snippets.md` or
`SynQc-TDS-integration-guide.md` and use it as a checklist while you wire
the system together.

````
