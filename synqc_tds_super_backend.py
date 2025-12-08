"""
SynQc Temporal Dynamics Series — Super Backend (safe, single-file)

This backend is designed to pair with the "Interstellar" SynQc TDS controller
HTML UI. It provides:

- Session model that mirrors the front-end controls:
  * hardware target & preset
  * drive envelope
  * probe strength & duration
  * adaptive rule
  * objective
  * iterations / shot limit (as hints)
  * free-form notes

- Simulation engine that produces physically-inspired KPIs:
  * DPD Fidelity (0–1)
  * Loop latency (microseconds)
  * Probe back-action (0–1)
  * Shot budget usage (per session)

- Persistent state:
  * Sessions and runs stored under SYNQC_STATE_DIR (JSON files)
  * Shot budget tracked per session
  * Logs kept per session
  * Lightweight telemetry history per session

- Export endpoint:
  * JSON, CSV, or "notebook" (ready-to-paste Python cell)

Security / safety notes
-----------------------
- No shell execution, no dynamic code eval.
- No automatic hot-reload subprocess (which can trip AV tools).
- Defaults to 127.0.0.1 if you run via `python synqc_tds_super_backend.py`.
- CORS defaults to `*` but credentials are disabled for that case.
- Probe strength, duration, and shot budgets are bounded and validated.

Environment variables (optional)
--------------------------------
- SYNQC_API_PREFIX=/api/v1/synqc
- SYNQC_STATE_DIR=./synqc_state
- SYNQC_ALLOWED_ORIGINS=*          # comma-separated list or '*'

- SYNQC_MAX_PROBE_STRENGTH=0.5     # clamp / validate ε
- SYNQC_MAX_PROBE_DURATION_NS=5000 # clamp / validate τ_p
- SYNQC_MAX_SHOTS_PER_RUN=200000   # upper bound for overrides

- SYNQC_HOST=127.0.0.1             # for __main__ convenience launcher
- SYNQC_PORT=8000
- SYNQC_RELOAD=0                   # set to '1' if you *explicitly* want reload

To run (recommended)
--------------------
Use uvicorn:

    uvicorn synqc_tds_super_backend:app --host 127.0.0.1 --port 8000

Or, for local testing only:

    python synqc_tds_super_backend.py
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
import asyncio
from threading import local as _thread_local_class
import threading

import numpy as np
_thread_local = _thread_local_class()

# Thread-local RNG: each thread gets its own Generator to avoid contention
def _get_thread_rng() -> np.random.Generator:
    g = getattr(_thread_local, "rng", None)
    if g is None:
        g = np.random.default_rng()
        _thread_local.rng = g
    return g
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Import the LLM agent (optional; only used if OPENAI_API_KEY is set)
try:
    from synqc_agent import SynQcAgent, AgentSuggestion
    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False

# Optional fast JSON serializer: use orjson when available for speed
try:
    import orjson

    def _json_dumps(obj: Any, indent: Optional[int] = None) -> bytes:
        # orjson ignores indent; mimic json by pretty-printing when requested
        if indent:
            return json.dumps(obj, indent=indent).encode("utf-8")
        return orjson.dumps(obj)

    def _json_loads(b: bytes) -> Any:
        return orjson.loads(b)

    _JSON_USES_ORJSON = True
except Exception:
    def _json_dumps(obj: Any, indent: Optional[int] = None) -> bytes:
        return json.dumps(obj, indent=indent).encode("utf-8")

    def _json_loads(b: bytes) -> Any:
        if isinstance(b, bytes):
            b = b.decode("utf-8")
        return json.loads(b)

    _JSON_USES_ORJSON = False


# -------------------------------------------------------------------------
# Environment & global bounds
# -------------------------------------------------------------------------

load_dotenv()


def _env(name: str, default: str) -> str:
    """Small helper for environment variables with a clear default."""
    return os.getenv(name, default)


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(lo, min(hi, value))


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, value))


API_PREFIX = _env("SYNQC_API_PREFIX", "/api/v1/synqc").rstrip("/")
STATE_DIR = Path(_env("SYNQC_STATE_DIR", "./synqc_state")).expanduser().resolve()

# Background flusher configuration
FLUSH_INTERVAL_SEC = _env_float("SYNQC_SESSION_FLUSH_INTERVAL_SEC", 1.0, 0.1, 60.0)
ENABLE_BACKGROUND_FLUSH = _env("SYNQC_ENABLE_BACKGROUND_FLUSH", "1") != "0"

ALLOWED_ORIGINS_RAW = _env("SYNQC_ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()] or ["*"]

# Safety limits for hardware-ish parameters
MAX_PROBE_STRENGTH: float = _env_float("SYNQC_MAX_PROBE_STRENGTH", 0.5, 0.0, 1.0)
MAX_PROBE_DURATION_NS: int = _env_int("SYNQC_MAX_PROBE_DURATION_NS", 5000, 1, 1_000_000)
MAX_SHOTS_PER_RUN: int = _env_int("SYNQC_MAX_SHOTS_PER_RUN", 200_000, 1, 10_000_000)

# Defaults used by the UI/session when no override is provided
DEFAULT_SHOT_LIMIT = min(50_000, MAX_SHOTS_PER_RUN)


# -------------------------------------------------------------------------
# Enumerations mirroring the front-end controls
# -------------------------------------------------------------------------


class HardwareTarget(str, Enum):
    SIM_LOCAL = "sim-local"
    IBM_QPU = "ibm-qpu"
    AWS_BRAKET = "aws-braket"
    IONQ = "ionq"
    CLASSICAL_ONLY = "classical-only"


class HardwarePreset(str, Enum):
    TRANSMON_DEFAULT = "transmon-default"
    FLUXONIUM_PILOT = "fluxonium-pilot"
    ION_CHAIN = "ion-chain"
    NEUTRAL_ATOM = "neutral-atom"


class DriveEnvelope(str, Enum):
    GAUSSIAN = "gaussian"
    SQUARE = "square"
    DRAG = "drag"
    COSINE = "cosine"


class AdaptiveRule(str, Enum):
    NONE = "none"
    KALMAN = "kalman"
    BAYES = "bayes"
    RL = "rl"


class Objective(str, Enum):
    MAXIMIZE_FIDELITY = "maximize-fidelity"
    MINIMIZE_LATENCY = "minimize-latency"
    INFO_VS_DAMAGE = "info-vs-damage"
    STABILITY_WINDOW = "stability-window"


class RunMode(str, Enum):
    RUN = "run"
    DRYRUN = "dryrun"


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    TERMINATED = "terminated"
    ERROR = "error"


class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    NOTEBOOK = "notebook"


# -------------------------------------------------------------------------
# Core configuration & KPI models
# -------------------------------------------------------------------------


class RunConfiguration(BaseModel):
    """
    Configuration that corresponds one-to-one with front-end controls.
    Extra fields in the JSON are safely ignored.
    """

    hardware_target: HardwareTarget = Field(
        ..., description="Target provider selected in the sidebar."
    )
    hardware_preset: HardwarePreset = Field(
        ..., description="Hardware profile (e.g. transmon, fluxonium, ion chain)."
    )
    drive_envelope: DriveEnvelope = Field(
        ..., description="Drive envelope shape used in the DPD schedule."
    )
    probe_strength: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Dimensionless probe strength ε (0–1).",
    )
    probe_duration_ns: int = Field(
        120,
        ge=5,
        le=5000,
        description="Probe duration τ_p in nanoseconds.",
    )
    adaptive_rule: AdaptiveRule = Field(
        AdaptiveRule.NONE,
        description="High-level controller class for adaptation.",
    )
    objective: Objective = Field(
        Objective.MAXIMIZE_FIDELITY,
        description="Primary objective for the loop.",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-form annotation from the Notes panel.",
    )

    @field_validator("probe_strength")
    @classmethod
    def _strength_non_nan(cls, v: float) -> float:
        if math.isnan(v):
            return 0.2
        return v

    @field_validator("probe_duration_ns")
    @classmethod
    def _duration_reasonable(cls, v: int) -> int:
        if v <= 0:
            return 120
        return v


class KpiBundle(BaseModel):
    """
    KPIs that drive the 'Live KPIs' cards in the HTML controller.
    """

    fidelity: float = Field(..., description="DPD fidelity proxy (0–1).")
    latency_us: float = Field(..., description="End-to-end DPD loop latency (µs).")
    backaction: float = Field(
        ..., description="Probe back-action (0–1, where lower is better)."
    )
    shots_used: int = Field(..., description="Cumulative shots used in session.")
    shot_limit: int = Field(DEFAULT_SHOT_LIMIT, description="Shot budget limit.")
    shots_used_fraction: float = Field(
        ..., description="shots_used / shot_limit, clipped to [0, 1]."
    )

    @classmethod
    def from_raw(
        cls,
        fidelity: float,
        latency_us: float,
        backaction: float,
        shots_used: int,
        shot_limit: int,
    ) -> "KpiBundle":
        shot_limit = max(0, int(shot_limit))
        shots_used = max(0, int(shots_used))
        if shot_limit <= 0:
            frac = 0.0
        else:
            frac = min(1.0, max(0.0, shots_used / shot_limit))

        return cls(
            fidelity=float(fidelity),
            latency_us=float(latency_us),
            backaction=float(backaction),
            shots_used=shots_used,
            shot_limit=shot_limit,
            shots_used_fraction=frac,
        )


class RunRecord(BaseModel):
    """
    Complete record for one run (real or dry-run).
    """

    run_id: str
    session_id: str
    mode: RunMode
    config_snapshot: RunConfiguration
    created_at: datetime
    kpis: Optional[KpiBundle] = None
    measurements: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """
    State for a control-panel session. This is what we persist between runs.
    """

    session_id: str
    created_at: datetime
    last_updated_at: datetime
    status: SessionStatus = SessionStatus.IDLE
    status_text: str = "Idle · no active run"
    mode_label: str = "Local Simulation"
    config: RunConfiguration
    last_run_id: Optional[str] = None
    shot_limit: int = DEFAULT_SHOT_LIMIT
    shots_used: int = 0
    logs: List[str] = Field(default_factory=list)
    last_kpis: Optional[KpiBundle] = None
    telemetry: List[Dict[str, Any]] = Field(default_factory=list)

    def add_log(self, message: str) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        entry = f"[{now}] {message}"
        self.logs.append(entry)
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]


# -------------------------------------------------------------------------
# Simulation engine — compact but SynQc-flavored
# -------------------------------------------------------------------------


@dataclass
class EngineConfig:
    base_latency_sim_local: float = 10.0
    base_latency_classical: float = 25.0
    base_latency_quantum: float = 80.0

    base_fidelity_sim_local: float = 0.99
    base_fidelity_classical: float = 0.985
    base_fidelity_quantum: float = 0.97

    shot_cost_baseline: int = 800
    shot_cost_per_ns: float = 0.3
    random_snr_db: float = 1.5  # jitter scale in dB


class SynQcEngine:
    """
    Simple engine that maps (config, session state, run hints) → KPIs.

    This is a synthetic but discipline-respecting model; it does NOT talk
    to actual QPUs here. Real hardware adapters can be added later.
    """

    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()

    def run(
        self,
        session: SessionState,
        mode: RunMode,
        *,
        num_iterations: Optional[int] = None,
    ) -> RunRecord:
        """
        Execute one run (or dry-run). For dry-run we still synthesize KPIs,
        but we do NOT increase the shot budget.
        """
        run_id = self._new_run_id(session.session_id)
        created_at = datetime.utcnow()
        cfg = session.config

        # run simulation — CPU work; caller may choose to run this in a thread
        kpis = self._simulate_kpis(
            cfg,
            session,
            count_shots=(mode == RunMode.RUN),
            num_iterations=num_iterations,
        )
        events = self._explain_kpis(cfg, kpis, mode)

        # synthesize per-qubit measurement payload for visualizer
        measurements = self._synthesize_measurements(cfg, kpis.fidelity)

        session.last_run_id = run_id
        session.last_updated_at = created_at
        session.last_kpis = kpis

        if mode == RunMode.RUN:
            session.shots_used = kpis.shots_used
            session.status = SessionStatus.IDLE
            session.status_text = "Idle · last run completed"
        else:
            session.status = SessionStatus.IDLE
            session.status_text = "Idle · last dry-run completed"

        label = "SynQc run" if mode == RunMode.RUN else "SynQc dry-run"
        session.add_log(
            f"{label} finished – fidelity={kpis.fidelity:.3f}, "
            f"latency={kpis.latency_us:.1f}µs, back-action={kpis.backaction:.3f}, "
            f"shots_used={kpis.shots_used}/{kpis.shot_limit}"
        )

        return RunRecord(
            run_id=run_id,
            session_id=session.session_id,
            mode=mode,
            config_snapshot=cfg.copy(deep=True),
            created_at=created_at,
            kpis=kpis,
            measurements=measurements,
            events=events,
        )

    @staticmethod
    def _new_run_id(session_id: str) -> str:
        safe = session_id.replace(":", "_").replace("/", "_")
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"{safe}-{stamp}"

    def _simulate_kpis(
        self,
        cfg: RunConfiguration,
        session: SessionState,
        count_shots: bool,
        num_iterations: Optional[int],
    ) -> KpiBundle:
        ecfg = self.cfg

        # Hardware class baseline
        if cfg.hardware_target == HardwareTarget.SIM_LOCAL:
            base_latency = ecfg.base_latency_sim_local
            base_fid = ecfg.base_fidelity_sim_local
        elif cfg.hardware_target == HardwareTarget.CLASSICAL_ONLY:
            base_latency = ecfg.base_latency_classical
            base_fid = ecfg.base_fidelity_classical
        else:
            base_latency = ecfg.base_latency_quantum
            base_fid = ecfg.base_fidelity_quantum

        # Hardware preset adjustments
        if cfg.hardware_preset == HardwarePreset.TRANSMON_DEFAULT:
            pass
        elif cfg.hardware_preset == HardwarePreset.FLUXONIUM_PILOT:
            base_fid -= 0.005
            base_latency += 15.0
        elif cfg.hardware_preset == HardwarePreset.ION_CHAIN:
            base_fid += 0.005
            base_latency += 30.0
        elif cfg.hardware_preset == HardwarePreset.NEUTRAL_ATOM:
            base_fid -= 0.01
            base_latency += 45.0

        latency = base_latency
        fid = base_fid

        # Objective tweaks
        if cfg.objective == Objective.MAXIMIZE_FIDELITY:
            fid += 0.005
            latency += 10.0
        elif cfg.objective == Objective.MINIMIZE_LATENCY:
            fid -= 0.007
            latency -= 12.0
        elif cfg.objective == Objective.INFO_VS_DAMAGE:
            fid += 0.002
            latency += 3.0
        elif cfg.objective == Objective.STABILITY_WINDOW:
            fid += 0.001
            latency += 5.0

        # Adaptive rule tweaks
        if cfg.adaptive_rule == AdaptiveRule.RL:
            latency += 8.0
        elif cfg.adaptive_rule == AdaptiveRule.KALMAN:
            fid += 0.003
        elif cfg.adaptive_rule == AdaptiveRule.BAYES:
            latency += 5.0

        # Safety bounds for probe parameters
        eps_raw = float(cfg.probe_strength)
        if eps_raw > MAX_PROBE_STRENGTH:
            # Fail fast with clear message instead of silently running
            raise ValueError(
                f"probe_strength={eps_raw} exceeds safe limit "
                f"{MAX_PROBE_STRENGTH}. Refusing to run."
            )
        eps = max(0.0, min(MAX_PROBE_STRENGTH, eps_raw))

        tau_raw = int(cfg.probe_duration_ns)
        if tau_raw > MAX_PROBE_DURATION_NS:
            raise ValueError(
                f"probe_duration_ns={tau_raw} exceeds safe limit "
                f"{MAX_PROBE_DURATION_NS}. Refusing to run."
            )
        tau_ns = max(1, min(MAX_PROBE_DURATION_NS, tau_raw))

        # Probe trade-offs
        deviation = (eps - 0.2) / 0.2  # 0 at "optimal" ε ≈ 0.2
        fid -= 0.015 * deviation * deviation

        backaction = 0.08 + 0.4 * (eps ** 1.1) + 0.00005 * tau_ns
        backaction = max(0.0, min(1.0, backaction))

        latency += 0.01 * (tau_ns / 10.0)

        # Envelope type influence
        if cfg.drive_envelope == DriveEnvelope.GAUSSIAN:
            pass
        elif cfg.drive_envelope == DriveEnvelope.SQUARE:
            latency -= 3.0
            fid -= 0.003
        elif cfg.drive_envelope == DriveEnvelope.DRAG:
            latency += 3.0
            fid += 0.004
        elif cfg.drive_envelope == DriveEnvelope.COSINE:
            latency += 1.0

        # Noise via effective SNR
        # Use a thread-local RNG to avoid constructing a new Generator each call
        rng = _get_thread_rng()
        snr_jitter_db = float(rng.normal(loc=0.0, scale=self.cfg.random_snr_db))
        snr_factor = math.exp(snr_jitter_db / 20.0)  # approximate

        fid *= min(1.02, max(0.95, snr_factor))
        latency /= min(1.05, max(0.95, snr_factor))

        # Shot accounting
        iters = max(1, int(num_iterations or 1))
        base_shots = self.cfg.shot_cost_baseline
        extra = int(self.cfg.shot_cost_per_ns * tau_ns)
        shots_this_run = max(100, (base_shots + extra) * iters)

        prior_shots = max(0, int(session.shots_used))
        shot_limit = max(1, min(session.shot_limit, MAX_SHOTS_PER_RUN))

        if count_shots:
            total_shots = min(shot_limit, prior_shots + shots_this_run)
        else:
            total_shots = prior_shots

        # Final clipping
        fid = max(0.0, min(0.9999, fid))
        latency = max(1.0, float(latency))

        return KpiBundle.from_raw(
            fidelity=fid,
            latency_us=latency,
            backaction=backaction,
            shots_used=total_shots,
            shot_limit=shot_limit,
        )

    def _explain_kpis(
        self,
        cfg: RunConfiguration,
        kpis: KpiBundle,
        mode: RunMode,
    ) -> List[str]:
        events: List[str] = []
        label = "RUN" if mode == RunMode.RUN else "DRY-RUN"
        events.append(
            f"[{label}] objective={cfg.objective.value}, "
            f"adaptive={cfg.adaptive_rule.value}, envelope={cfg.drive_envelope.value}"
        )

        if kpis.fidelity >= 0.98:
            events.append(
                f"[KPIs] High fidelity regime (f={kpis.fidelity:.4f}); calibration stable."
            )
        elif kpis.fidelity >= 0.97:
            events.append(
                f"[KPIs] Good fidelity (f={kpis.fidelity:.4f}); small drifts acceptable."
            )
        else:
            events.append(
                f"[KPIs] Fidelity at {kpis.fidelity:.4f}; consider recalibrating or "
                f"reducing probe strength."
            )

        if kpis.latency_us <= 20.0:
            events.append(
                f"[KPIs] Low loop latency ({kpis.latency_us:.1f}µs); suitable for fast feedback."
            )
        elif kpis.latency_us <= 60.0:
            events.append(
                f"[KPIs] Moderate latency ({kpis.latency_us:.1f}µs); within nominal budget."
            )
        else:
            events.append(
                f"[KPIs] High latency ({kpis.latency_us:.1f}µs); bottleneck detected."
            )

        if kpis.backaction <= 0.15:
            events.append(
                f"[KPIs] Back-action low ({kpis.backaction:.3f}); probes are gentle."
            )
        elif kpis.backaction <= 0.3:
            events.append(
                f"[KPIs] Back-action moderate ({kpis.backaction:.3f}); acceptable trade-off."
            )
        else:
            events.append(
                f"[KPIs] Back-action elevated ({kpis.backaction:.3f}); consider weaker probes."
            )

        frac = kpis.shots_used_fraction
        used_pct = 100.0 * frac
        events.append(
            f"[BUDGET] Shot usage: {kpis.shots_used}/{kpis.shot_limit} "
            f"({used_pct:.1f}% of budget)."
        )

        if frac >= 0.95:
            events.append("[BUDGET] WARNING: Near shot budget exhaustion.")
        elif frac >= 0.75:
            events.append("[BUDGET] CAUTION: Shot usage high; plan refills.")

        return events

    def _synthesize_measurements(self, cfg: RunConfiguration, fidelity: float, num_qubits: int = 4) -> List[Dict[str, Any]]:
        """Create a simple measurement list compatible with the frontend visualizer.

        Each measurement dict contains {qubit, p0, p1, last}.
        """
        rng = _get_thread_rng()
        out: List[Dict[str, Any]] = []
        for q in range(num_qubits):
            # create p1 biased around fidelity with per-qubit jitter
            jitter = float(rng.normal(0.0, 0.06))
            p1 = max(0.0, min(1.0, 0.5 * fidelity + 0.5 * float(rng.uniform(0.0, 1.0)) + jitter))
            p0 = 1.0 - p1
            last = int(rng.random() < p1)
            out.append({"qubit": q, "p0": p0, "p1": p1, "last": last})
        return out


# -------------------------------------------------------------------------
# Persistent storage
# -------------------------------------------------------------------------


class StateStore:
    """
    File-backed store for sessions and runs.

    Layout under STATE_DIR:
      - sessions.json : list of SessionState objects
      - runs/         : one JSON per RunRecord, named {run_id}.json
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, SessionState] = {}
        # IO lock to serialize file writes/reads and avoid concurrent renames
        self._io_lock = threading.Lock()
        self._load_sessions()

    def _sessions_path(self) -> Path:
        return self.root / "sessions.json"

    def _load_sessions(self) -> None:
        path = self._sessions_path()
        if not path.exists():
            self.sessions = {}
            return
        try:
            with self._io_lock:
                raw_bytes = path.read_bytes()
            raw = _json_loads(raw_bytes)
            if not isinstance(raw, list):
                self.sessions = {}
                return
            loaded: Dict[str, SessionState] = {}
            for item in raw:
                try:
                    # item is expected to be a dict serialized from model_dump()
                    st = SessionState.model_validate(item)
                    loaded[st.session_id] = st
                except Exception:
                    continue
            self.sessions = loaded
        except Exception:
            # Fail safe: start from empty if file is corrupt
            self.sessions = {}

    def _save_sessions(self) -> None:
        # Use dicts from model_dump() and write bytes via preferred JSON backend
        data = [s.model_dump() for s in self.sessions.values()]
        tmp_path = self._sessions_path().with_suffix(".tmp")
        with self._io_lock:
            tmp_path.write_bytes(_json_dumps(data, indent=2))
            tmp_path.replace(self._sessions_path())

    def _run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"{run_id}.json"

    def save_run(self, run: RunRecord) -> None:
        path = self._run_path(run.run_id)
        # Persist run record using fast JSON path
        with self._io_lock:
            path.write_bytes(_json_dumps(run.model_dump(), indent=2))

    def load_run(self, run_id: str) -> RunRecord:
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(run_id)
        with self._io_lock:
            raw_bytes = path.read_bytes()
        raw = _json_loads(raw_bytes)
        return RunRecord.model_validate(raw)

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def upsert_session(self, session: SessionState) -> SessionState:
        self.sessions[session.session_id] = session
        # By default, persist to disk. Callers that are high-frequency (e.g.
        # telemetry polling) may set `persist=False` by calling directly and
        # then not calling _save_sessions; to keep the surface simple we keep
        # this method as the canonical in-memory updater and allow callers to
        # call _save_sessions() separately when needed.
        self._save_sessions()
        return session

    def all_sessions(self) -> List[SessionState]:
        return list(self.sessions.values())

    def update_in_memory(self, session: SessionState) -> None:
        """Update the in-memory session without forcing a disk write.

        Use this for high-frequency operations (e.g. telemetry polling)
        to avoid excessive disk I/O.
        """
        self.sessions[session.session_id] = session


# -------------------------------------------------------------------------
# FastAPI app setup
# -------------------------------------------------------------------------


app = FastAPI(
    title="SynQc Temporal Dynamics Series — Backend",
    description=(
        "Backend API that matches the SynQc Temporal Dynamics Series Control "
        "Panel. Provides session management, synthetic DPD KPIs, logging, "
        "telemetry, and export."
    ),
    version="0.3.0",
)

# If you set explicit origins, we let credentials through; if wildcard,
# credentials are disabled for safety.
allow_credentials_flag = ALLOWED_ORIGINS != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=allow_credentials_flag,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve optional static assets for the UI at /ui/static/
# Directory `ui_static` is expected next to this file. It's harmless if empty.
try:
    static_dir = Path(__file__).parent / "ui_static"
    # Ensure the path exists when running in production; if not, StaticFiles will
    # still mount but return 404 for missing files. Creating the directory is
    # non-destructive and helps local development.
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/ui/static", StaticFiles(directory=str(static_dir)), name="ui-static")
except Exception:
    # Do not raise on mount failure; the app should still start.
    pass

store = StateStore(STATE_DIR)
engine = SynQcEngine()
agent: Optional[SynQcAgent] = None  # LLM-based advisory agent
_flusher_task: Optional[asyncio.Task] = None
_flusher_stop: Optional[asyncio.Event] = None

# Initialize the LLM agent if available and API key is set
if _AGENT_AVAILABLE and os.getenv("OPENAI_API_KEY"):
    try:
        agent = SynQcAgent()
    except Exception:
        # If agent init fails, continue without it (graceful degradation)
        pass


async def _session_flusher_loop(stop_event: asyncio.Event, interval: float) -> None:
    """Periodically flush in-memory sessions to disk.

    Runs _save_sessions in a thread to avoid blocking the event loop.
    """
    try:
        while not stop_event.is_set():
            await asyncio.sleep(interval)
            # run the blocking save in a thread
            try:
                await asyncio.to_thread(store._save_sessions)
            except Exception:
                # flusher must be resilient; log in-memory and continue
                # Can't call st.add_log here without a session; rely on file system
                continue
    except asyncio.CancelledError:
        return


@app.on_event("startup")
async def _start_background_flusher() -> None:
    global _flusher_task, _flusher_stop
    if not ENABLE_BACKGROUND_FLUSH:
        return
    if _flusher_task is not None:
        return
    _flusher_stop = asyncio.Event()
    _flusher_task = asyncio.create_task(_session_flusher_loop(_flusher_stop, FLUSH_INTERVAL_SEC))


@app.on_event("shutdown")
async def _stop_background_flusher() -> None:
    global _flusher_task, _flusher_stop
    if _flusher_task is None:
        return
    # signal stop and wait for the task to finish
    _flusher_stop.set()
    try:
        await asyncio.wait_for(_flusher_task, timeout=5.0)
    except Exception:
        _flusher_task.cancel()
    _flusher_task = None
    _flusher_stop = None


# -------------------------------------------------------------------------
# Serve simple static frontend
# -------------------------------------------------------------------------

@app.get("/ui")
def _serve_ui() -> FileResponse:
    """Serve the bundled control panel HTML for convenience.

    This is a convenience route for local use. In production you may serve
    static assets separately (CDN / web server). The file is expected to be
    next to this module: `synqc_control_panel.html`.
    """
    try:
        p = Path(__file__).parent / "synqc_control_panel.html"
        if not p.exists():
            raise HTTPException(status_code=404, detail="UI not found")
        return FileResponse(str(p), media_type="text/html")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read UI file")


# -------------------------------------------------------------------------
# API schemas (requests / responses)
# -------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description=(
            "Optional client-provided session ID. If omitted, the backend "
            "generates a new one of the form 'synqc-XXXXXX'."
        ),
    )
    config: RunConfiguration


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    last_updated_at: datetime
    status: SessionStatus
    status_text: str
    mode_label: str
    shot_limit: int
    shots_used: int
    shots_used_fraction: float
    last_run_id: Optional[str] = None
    last_kpis: Optional[KpiBundle] = None
    config: RunConfiguration

    @classmethod
    def from_state(cls, st: SessionState) -> "SessionSummary":
        shot_limit = max(1, st.shot_limit)
        frac = min(1.0, max(0.0, st.shots_used / shot_limit))
        return cls(
            session_id=st.session_id,
            created_at=st.created_at,
            last_updated_at=st.last_updated_at,
            status=st.status,
            status_text=st.status_text,
            mode_label=st.mode_label,
            shot_limit=shot_limit,
            shots_used=st.shots_used,
            shots_used_fraction=frac,
            last_run_id=st.last_run_id,
            last_kpis=st.last_kpis,
            config=st.config,
        )


class RunRequest(BaseModel):
    mode: RunMode = RunMode.RUN
    num_iterations: Optional[int] = Field(
        None,
        ge=1,
        le=10_000,
        description="Optional iterations per run (used in the simulator).",
    )
    shot_limit: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_SHOTS_PER_RUN,
        description="Optional per-session shot budget override.",
    )


class RunResponse(BaseModel):
    run: RunRecord
    session: SessionSummary


class AgentRequest(BaseModel):
    goal: str = Field(
        ...,
        description=(
            "High-level experimental goal, e.g. 'maximize fidelity under 20k shots' "
            "or 'minimize latency while maintaining fidelity above 0.97'."
        ),
    )
    max_retries: int = Field(3, ge=1, le=10, description="Max LLM retries on parse errors.")


class AgentSuggestionResponse(BaseModel):
    recommended_config: RunConfiguration = Field(
        ..., description="Full validated suggested configuration."
    )
    rationale: str = Field(..., description="Why the agent made this suggestion.")
    warnings: List[str] = Field(default_factory=list, description="Potential risks or caveats.")
    changes_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Diff of changes from current config."
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


def _derive_mode_label(target: HardwareTarget) -> str:
    """Derive a label for the target hardware from its name."""
    parts = target.name.lower().split("_")
    return " ".join(p.capitalize() for p in parts)


def _session_snapshot_for_agent(st: SessionState) -> Dict[str, Any]:
    """Build a compact snapshot for LLM agent analysis."""
    k = st.last_kpis or KpiBundle.from_raw(
        fidelity=0.0,
        latency_us=0.0,
        backaction=0.0,
        shots_used=st.shots_used,
        shot_limit=st.shot_limit,
    )
    return {
        "session_id": st.session_id,
        "created_at": st.created_at.isoformat(),
        "status": st.status.value,
        "mode_label": st.mode_label,
        "current_config": st.config.model_dump(),
        "shots_used": st.shots_used,
        "shot_limit": st.shot_limit,
        "last_kpis": {
            "fidelity": k.fidelity,
            "latency_us": k.latency_us,
            "backaction": k.backaction,
            "shots_used_fraction": k.shots_used_fraction,
        },
    }

    if target == HardwareTarget.SIM_LOCAL:
        return "Local Simulation"
    if target == HardwareTarget.CLASSICAL_ONLY:
        return "Classical Hardware"
    return "Quantum Backend"


def _new_session_id() -> str:
    import secrets

    suffix = secrets.token_hex(3)
    return f"synqc-{suffix}"


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------


@app.get(f"{API_PREFIX}/health")
def health() -> Dict[str, Any]:
    """Lightweight health check plus basic configuration info."""
    return {
        "status": "ok",
        "version": app.version,
        "api_prefix": API_PREFIX,
        "state_dir": str(STATE_DIR),
        "allowed_origins": ALLOWED_ORIGINS,
        "max_probe_strength": MAX_PROBE_STRENGTH,
        "max_probe_duration_ns": MAX_PROBE_DURATION_NS,
        "max_shots_per_run": MAX_SHOTS_PER_RUN,
    }


# --- Session CRUD -----------------------------------------------------


@app.get(f"{API_PREFIX}/sessions", response_model=List[SessionSummary])
def list_sessions() -> List[SessionSummary]:
    """List all known sessions."""
    return [SessionSummary.from_state(s) for s in store.all_sessions()]


@app.post(f"{API_PREFIX}/sessions", response_model=SessionSummary)
def create_or_update_session(req: SessionCreateRequest) -> SessionSummary:
    """
    Create a new session or update an existing one.

    The front-end should call this when configuration changes.
    """
    now = datetime.utcnow()
    existing = store.get_session(req.session_id) if req.session_id else None

    if existing is None:
        session_id = req.session_id or _new_session_id()
        st = SessionState(
            session_id=session_id,
            created_at=now,
            last_updated_at=now,
            status=SessionStatus.IDLE,
            status_text="Idle · session created",
            mode_label=_derive_mode_label(req.config.hardware_target),
            config=req.config,
            shot_limit=DEFAULT_SHOT_LIMIT,
            shots_used=0,
        )
        st.add_log("Session created.")
    else:
        st = existing
        st.config = req.config
        st.mode_label = _derive_mode_label(req.config.hardware_target)
        st.last_updated_at = now
        st.add_log("Configuration updated from front-end.")

    store.upsert_session(st)
    return SessionSummary.from_state(st)


@app.get(f"{API_PREFIX}/sessions/{{session_id}}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    """Get full state for a session."""
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return SessionSummary.from_state(st)


# --- Logs -------------------------------------------------------------


@app.get(f"{API_PREFIX}/sessions/{{session_id}}/logs")
def get_logs(
    session_id: str,
    limit: int = Query(200, ge=1, le=1000, description="Maximum lines to return."),
) -> Dict[str, Any]:
    """Return recent log lines for the session."""
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    logs = st.logs[-limit:]
    return {"session_id": session_id, "lines": logs}


@app.delete(f"{API_PREFIX}/sessions/{{session_id}}/logs")
def clear_logs(session_id: str) -> Dict[str, Any]:
    """Clear all logs for the session."""
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    st.logs = []
    st.last_updated_at = datetime.utcnow()
    st.add_log("Log cleared.")
    store.upsert_session(st)
    return {"status": "ok", "session_id": session_id}


# --- Run control ------------------------------------------------------


@app.post(f"{API_PREFIX}/sessions/{{session_id}}/run", response_model=RunResponse)
async def launch_run(session_id: str, req: RunRequest) -> RunResponse:
    """
    Launch a SynQc run or dry-run for the given session.

    Front-end mappings:
      - Launch run: mode='run'
      - Dry-run:    mode='dryrun'
    """
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    if st.status == SessionStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Session is already running.")

    # Optional shot_limit override (per session)
    if req.shot_limit is not None:
        safe_limit = min(MAX_SHOTS_PER_RUN, max(1, req.shot_limit))
        st.shot_limit = safe_limit
        st.add_log(f"Shot limit overridden from front-end: {safe_limit}.")

    # Hardware safety checks (before touching the engine)
    cfg = st.config
    if cfg.probe_strength > MAX_PROBE_STRENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"probe_strength={cfg.probe_strength} exceeds safe limit "
                f"{MAX_PROBE_STRENGTH}. Adjust slider and retry."
            ),
        )
    if cfg.probe_duration_ns > MAX_PROBE_DURATION_NS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"probe_duration_ns={cfg.probe_duration_ns} exceeds safe limit "
                f"{MAX_PROBE_DURATION_NS}. Adjust slider and retry."
            ),
        )

    st.status = SessionStatus.RUNNING
    st.status_text = "Running · SynQc DPD sequence in progress"
    st.last_updated_at = datetime.utcnow()
    st.add_log(f"Run requested with mode={req.mode.value}.")
    # persist the state quickly before heavy compute; write in thread
    await asyncio.to_thread(store._save_sessions)

    try:
        # Run the CPU-bound simulation in a thread to avoid blocking the event loop
        run = await asyncio.to_thread(engine.run, st, req.mode, num_iterations=req.num_iterations)
    except ValueError as exc:
        # Domain (safety) errors become 400 for the client
        st.status = SessionStatus.ERROR
        st.status_text = f"Run aborted: {exc}"
        st.add_log(f"Run aborted: {exc}")
        await asyncio.to_thread(store._save_sessions)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # defensive catch-all
        st.status = SessionStatus.ERROR
        st.status_text = "Error during run."
        st.add_log(f"Run failed with error: {exc}")
        await asyncio.to_thread(store._save_sessions)
        raise HTTPException(status_code=500, detail="Internal SynQc error.") from exc

    # Persist run and final session state without blocking the loop
    await asyncio.to_thread(store.save_run, run)
    await asyncio.to_thread(store._save_sessions)
    return RunResponse(run=run, session=SessionSummary.from_state(st))


@app.post(f"{API_PREFIX}/sessions/{{session_id}}/kill")
def kill_run(session_id: str) -> Dict[str, Any]:
    """
    Kill-switch endpoint.

    In this synchronous demo engine, this marks the session's status and logs
    the event; it does not interrupt an actual background worker.
    """
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    if st.status != SessionStatus.RUNNING:
        st.add_log("Kill switch pressed with no active run.")
        store.upsert_session(st)
        return {
            "status": "no-active-run",
            "message": "Kill switch pressed with no active run.",
        }

    st.status = SessionStatus.TERMINATED
    st.status_text = "Idle · last run terminated via kill switch"
    st.last_updated_at = datetime.utcnow()
    st.add_log("Kill switch activated. Run terminated.")
    store.upsert_session(st)
    return {"status": "terminated", "session_id": session_id}


# --- Run retrieval ----------------------------------------------------


@app.get(f"{API_PREFIX}/runs/{{run_id}}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    """Retrieve a stored RunRecord."""
    try:
        return store.load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")


# --- Telemetry for real-time monitoring -------------------------------


@app.get(f"{API_PREFIX}/sessions/{{session_id}}/telemetry")
def get_telemetry(
    session_id: str,
    limit: int = Query(64, ge=1, le=512, description="Maximum telemetry rows."),
) -> Dict[str, Any]:
    """
    Lightweight telemetry feed.

    The front-end polls this endpoint regularly. We append a synthetic row
    based on the latest KPIs (plus a bit of jitter) and return up to `limit`
    most recent entries.
    """
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    k = st.last_kpis
    if k is None:
        return {"session_id": session_id, "rows": []}

    now = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    # Use a thread-local RNG to avoid allocating per-call and to avoid
    # cross-thread contention.
    rng = _get_thread_rng()
    fid_jitter = float(rng.normal(0.0, 0.0005))
    lat_jitter = float(rng.normal(0.0, 1.5))

    row = {
        "timestamp": now,
        "provider": st.config.hardware_target.value,
        "fidelity": max(0.0, min(0.9999, k.fidelity + fid_jitter)),
        "latency_us": max(1.0, k.latency_us + lat_jitter),
        "shots_used": int(st.shots_used),
        "shot_rate": None,  # could be filled from real hardware timing
    }

    # Update telemetry in-memory only — avoid heavy disk writes on frequent polls
    st.telemetry.append(row)
    if len(st.telemetry) > 512:
        st.telemetry = st.telemetry[-512:]
    store.update_in_memory(st)

    rows = st.telemetry[-limit:]
    return {"session_id": session_id, "rows": rows}


# --- Agent suggestions (LLM-powered advisory) -------------------------


@app.post(f"{API_PREFIX}/sessions/{{session_id}}/agent-suggestion", response_model=AgentSuggestionResponse)
async def get_agent_suggestion(
    session_id: str,
    req: AgentRequest,
) -> AgentSuggestionResponse:
    """
    Ask the LLM agent for a configuration suggestion for the next run.

    This endpoint requires OPENAI_API_KEY to be set and the agent to be initialized.
    It is purely advisory — the suggestion is never applied automatically.

    Returns a full validated RunConfiguration, rationale, warnings, and diff of changes.
    """
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Agent not available. Ensure OPENAI_API_KEY is set and synqc_agent "
                "module is installed."
            ),
        )

    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    # Build compact snapshot for agent
    snapshot = _session_snapshot_for_agent(st)

    try:
        # Run agent in a thread to avoid blocking event loop
        suggestion = await asyncio.to_thread(
            agent.suggest,
            session_snapshot=snapshot,
            goal=req.goal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agent error: {str(exc)[:200]}",
        ) from exc

    # Parse suggested config dict into a RunConfiguration
    suggested_dict = suggestion.recommended_config
    try:
        suggested_cfg = RunConfiguration(**suggested_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agent response failed validation: {str(exc)[:200]}",
        ) from exc

    # Apply the suggestion as an in-memory patch (don't persist)
    current_dict = st.config.model_dump()
    changes = {}
    for key, value in suggested_dict.items():
        if key in current_dict and current_dict[key] != value:
            changes[key] = {"old": current_dict[key], "new": value}

    # Re-enforce hard limits as belt-and-braces validation
    clamped_cfg = RunConfiguration(
        **{
            **suggested_cfg.model_dump(),
            "shot_limit": min(suggested_cfg.shot_limit, MAX_SHOTS_PER_RUN),
        }
    )

    # Log the suggestion for auditing
    print(
        f"Agent suggested config for session {session_id}: "
        f"goal={req.goal!r}, changes={changes}"
    )

    return AgentSuggestionResponse(
        recommended_config=clamped_cfg,
        rationale=suggestion.rationale,
        warnings=suggestion.warnings,
        changes_applied=changes,
    )


# ---------- Chat endpoint (simple echo / passthrough to agent) ----------


@app.post(f"{API_PREFIX}/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest) -> ChatResponse:
    """Minimal chat endpoint. If a SynQcAgent is available, defer to it; otherwise echo."""
    session_part = f" (session {body.session_id})" if body.session_id else ""

    # If a richer agent is configured, try to use it (run in thread)
    if agent is not None and hasattr(agent, "chat"):
        try:
            # run agent.chat in thread if it's blocking
            reply = await asyncio.to_thread(agent.chat, body.message, body.history)
            if isinstance(reply, dict) and "reply" in reply:
                text = reply.get("reply")
            else:
                text = str(reply)
            return ChatResponse(reply=text, session_id=body.session_id)
        except Exception:
            # Fall through to fallback reply
            pass

    # Fallback echo-style pseudo-reply (safe, deterministic)
    last = (body.message or "").strip()
    reply = (
        f"Pseudo-agent{session_part}: you said '{last}'. "
        "Wire this endpoint to a real model to get richer guidance."
    )
    return ChatResponse(reply=reply, session_id=body.session_id)


# --- Export snapshot --------------------------------------------------


@app.get(f"{API_PREFIX}/sessions/{{session_id}}/export")
def export_snapshot(
    session_id: str,
    format: ExportFormat = Query(
        ExportFormat.JSON,
        description="Export format: json, csv, or notebook.",
    ),
):
    """
    Export a snapshot that mirrors the front-end's export payload.
    """
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    kpis = st.last_kpis or KpiBundle.from_raw(
        fidelity=0.0,
        latency_us=0.0,
        backaction=0.0,
        shots_used=st.shots_used,
        shot_limit=st.shot_limit,
    )

    data = {
        "sessionId": st.session_id,
        "mode": st.mode_label,
        "hardwareTarget": st.config.hardware_target.value,
        "hardwarePreset": st.config.hardware_preset.value,
        "driveEnvelope": st.config.drive_envelope.value,
        "probeStrength": st.config.probe_strength,
        "probeDurationNs": st.config.probe_duration_ns,
        "adaptiveRule": st.config.adaptive_rule.value,
        "objective": st.config.objective.value,
        "kpis": {
            "fidelity": kpis.fidelity,
            "latencyUs": kpis.latency_us,
            "backAction": kpis.backaction,
            "shotUsage": f"{kpis.shots_used} / {kpis.shot_limit}",
        },
        "notes": st.config.notes,
        "exportedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if format == ExportFormat.CSV:
        flat = {
            "sessionId": data["sessionId"],
            "mode": data["mode"],
            "hardwareTarget": data["hardwareTarget"],
            "hardwarePreset": data["hardwarePreset"],
            "driveEnvelope": data["driveEnvelope"],
            "probeStrength": data["probeStrength"],
            "probeDurationNs": data["probeDurationNs"],
            "adaptiveRule": data["adaptiveRule"],
            "objective": data["objective"],
            "kpi_fidelity": data["kpis"]["fidelity"],
            "kpi_latencyUs": data["kpis"]["latencyUs"],
            "kpi_backAction": data["kpis"]["backAction"],
            "kpi_shotUsage": data["kpis"]["shotUsage"],
            "notes": data["notes"],
            "exportedAt": data["exportedAt"],
        }

        def _csv_escape(value: Any) -> str:
            s = "" if value is None else str(value)
            if any(c in s for c in [",", '"', "\n", "\r"]):
                s = '"' + s.replace('"', '""') + '"'
            return s

        header = list(flat.keys())
        header_line = ",".join(header)
        row_line = ",".join(_csv_escape(flat[k]) for k in header)
        csv_text = header_line + "\n" + row_line + "\n"
        return {"format": "csv", "payload": csv_text}

    if format == ExportFormat.NOTEBOOK:
        cell = (
            "# SynQc TDS snapshot\n"
            "snapshot = "
            + json.dumps(data, indent=2)
            + "\n\n"
            "# Use `snapshot` inside your Jupyter pipeline.\n"
        )
        return {"format": "notebook", "payload": cell}

    return {"format": "json", "payload": data}


# -------------------------------------------------------------------------
# Convenience launcher (optional, safer defaults)
# -------------------------------------------------------------------------


if __name__ == "__main__":
    # This is purely for convenience when running:
    #   python synqc_tds_super_backend.py
    # For production or serious use, prefer:
    #   uvicorn synqc_tds_super_backend:app --host 127.0.0.1 --port 8000
    import uvicorn

    host = _env("SYNQC_HOST", "127.0.0.1")
    port = _env_int("SYNQC_PORT", 8000, 1, 65535)
    reload_flag = _env("SYNQC_RELOAD", "0") == "1"

    uvicorn.run(app, host=host, port=port, reload=reload_flag)
