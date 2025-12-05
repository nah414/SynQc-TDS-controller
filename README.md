# SynQc-TDS-controller
Synchronized Quantum Circuits Temporal Dynamics Series controller.  
# SynQc-TDS-controller

Synchronized Quantum Circuits Temporal Dynamics Series controller – a small, session-based
control stack for drive–probe–drive (DPD) experiments, with a local simulator, a pluggable
hardware abstraction layer, and a browser control panel.

## 1. Layout

- `synqc_tds_super_backend.py` – FastAPI backend, session engine, KPIs, state store.
- `hardware_backends.py` – hardware abstraction layer (local sim + IBM / Braket / IonQ / FPGA stubs).
- `synqc_control_panel.html` – static frontend that talks to the FastAPI API.
- `adac1680-...html` – older prototype UI (kept for reference).

## 2. Backend quickstart (local simulator)

Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
