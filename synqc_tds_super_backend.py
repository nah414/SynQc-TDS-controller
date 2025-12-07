"""
hardware_backends.py

Hardware abstraction layer for SynQc TDS.

This isolates:
- Local simulation (for the UI and development)
- Real quantum providers (IBM, AWS Braket, IonQ)
- Classical-only lab hardware (FPGA/DAQ)
from the FastAPI app and session engine.

You plug it into your existing SynQcEngine so that
SynQcEngine only knows "call backend.run_experiment(config, session)".
"""

from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Shared limits (safety guards)
# ---------------------------------------------------------------------------

MAX_PROBE_STRENGTH: float = float(os.getenv("SYNQC_MAX_PROBE_STRENGTH", "0.5"))
MAX_PROBE_DURATION_NS: int = int(os.getenv("SYNQC_MAX_PROBE_DURATION_NS", "5000"))
MAX_SHOTS_PER_RUN: int = int(os.getenv("SYNQC_MAX_SHOTS_PER_RUN", "200_000"))


@dataclass
class SimKpis:
  """Simple dataclass so we don't tie to a particular Pydantic model here."""

  fidelity: float
  latency_us: float
  backaction: float
  shots_used: int
  shot_limit: int

  @property
  def shots_used_fraction(self) -> float:
    return min(1.0, self.shots_used / self.shot_limit) if self.shot_limit > 0 else 0.0


class HardwareBackend(ABC):
  """
  Contract for all hardware backends.

  cfg:  your existing RunConfiguration (Pydantic model on the backend).
  session: your existing SessionState instance.
  Returns: something that can be converted into your KpiBundle.
  """

  target_id: str  # e.g. "sim-local", "ibm-qpu", ...

  def __init__(self, target_id: str, name: str):
    self.target_id = target_id
    self.name = name

  @abstractmethod
  def run_experiment(self, cfg: Any, session: Any, *, dry_run: bool = False) -> SimKpis:
    """
    Execute (or simulate) a DPD experiment for the given configuration.

    This is where:
      - you call Qiskit / Braket / IonQ / DAQ drivers
      - OR you call a local simulator
      - you enforce hardware safety limits
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Local simulation backend
# ---------------------------------------------------------------------------


class LocalSimBackend(HardwareBackend):
  """
  High-level "sane" simulator that mimics tradeoffs:

  - Higher probe_strength -> more information but more back-action.
  - Longer probe_duration_ns -> higher latency and back-action, some fidelity changes.
  - Different objectives tweak effective scaling.

  This is where the previous SynQcEngine._simulate_kpis logic lives.
  """

  def __init__(self) -> None:
    super().__init__(target_id="sim-local", name="Local simulator")
    self._rng = np.random.default_rng()

  def run_experiment(self, cfg: Any, session: Any, *, dry_run: bool = False) -> SimKpis:
    # Safety clamps
    eps = float(cfg.probe_strength)
    tau_ns = int(cfg.probe_duration_ns)

    if eps < 0:
      eps = 0.0
    if eps > MAX_PROBE_STRENGTH:
      # Hard fail instead of silently clamping, safer for real hardware.
      raise ValueError(
        f"probe_strength={cfg.probe_strength} exceeds limit "
        f"{MAX_PROBE_STRENGTH}. Refusing to run."
      )
    if tau_ns < 1:
      tau_ns = 1
    if tau_ns > MAX_PROBE_DURATION_NS:
      raise ValueError(
        f"probe_duration_ns={cfg.probe_duration_ns} exceeds limit "
        f"{MAX_PROBE_DURATION_NS}. Refusing to run."
      )

    # Basic "difficulty" factor by objective
    objective = getattr(cfg, "objective", "maximize-fidelity")
    obj_weight = {
      "maximize-fidelity": 1.0,
      "minimize-latency": 0.9,
      "info-vs-damage": 0.95,
      "stability-window": 0.97,
    }.get(objective, 1.0)

    # "Hardware" scaling: transmon vs trapped-ion vs neutral atoms etc.
    preset = getattr(cfg, "hardware_preset", "transmon-default")
    hw_factor = {
      "transmon-default": 1.0,
      "fluxonium-pilot": 0.96,
      "ion-chain": 0.93,
      "neutral-atom": 0.9,
    }.get(preset, 1.0)

    # Latency in microseconds (toy model)
    base_latency_us = 15.0
    latency_us = base_latency_us + (tau_ns / 500.0) + 20.0 * eps
    latency_us *= hw_factor

    # Back-action grows with eps and tau (but saturates)
    backaction = 1.0 - math.exp(-eps * tau_ns / 1000.0)
    backaction = min(backaction, 1.0)

    # Fidelity: high at small eps and moderate tau; falls if too aggressive
    sweet_eps = 0.18
    sweet_tau = 300.0
    eps_term = math.exp(-((eps - sweet_eps) ** 2) / (2 * 0.06**2))
    tau_term = math.exp(-((tau_ns - sweet_tau) ** 2) / (2 * 280.0**2))
    fidelity_mean = 0.80 + 0.15 * eps_term * tau_term
    fidelity_mean *= obj_weight
    fidelity_mean *= hw_factor

    # Some stochasticity
    jitter = float(self._rng.normal(0.0, 0.004))
    fidelity = max(0.0, min(0.9999, fidelity_mean + jitter))

    # Shot accounting
    default_shots = 20_000
    shot_limit = min(MAX_SHOTS_PER_RUN, int(getattr(session, "shot_limit", default_shots)))
    shots_used = min(shot_limit, int(default_shots * (0.6 + 0.8 * eps)))

    # Optionally be nicer for dry-run (short latency, no consumption)
    if dry_run:
      latency_us *= 0.3
      shots_used = 0

    return SimKpis(
      fidelity=fidelity,
      latency_us=latency_us,
      backaction=backaction,
      shots_used=shots_used,
      shot_limit=shot_limit,
    )


# ---------------------------------------------------------------------------
# IBM Quantum QPU backend (skeleton)
# ---------------------------------------------------------------------------


class IbmQpuBackend(HardwareBackend):
  """
  Skeleton integration for IBM QPU.

  You configure:
    - IBM_QUANTUM_CHANNEL (optional)
    - IBM_QUANTUM_BACKEND_NAME
    - IBM_QUANTUM_TOKEN (or use qiskit-ibm-runtime's default account config)

  This class intentionally imports qiskit lazily so the file can be imported
  even if qiskit is not installed (e.g. purely-sim local dev).
  """

  def __init__(self) -> None:
    super().__init__(target_id="ibm-qpu", name="IBM Quantum")

  def run_experiment(self, cfg: Any, session: Any, *, dry_run: bool = False) -> SimKpis:
    if dry_run:
      # Delegate to local sim for UI-only dry-runs.
      return LocalSimBackend().run_experiment(cfg, session, dry_run=True)

    try:
      from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
      raise RuntimeError(
        "qiskit-ibm-runtime is not installed. "
        "Install with `pip install qiskit-ibm-runtime`."
      ) from exc

    backend_name = os.getenv("IBM_QUANTUM_BACKEND_NAME", "").strip()
    if not backend_name:
      raise RuntimeError("IBM_QUANTUM_BACKEND_NAME not set")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    # TODO: translate cfg into a circuit and a schedule.
    # This is where you call assemble/run/etc. For now we refuse to run
    # to avoid giving you fake QPU results.
    raise NotImplementedError(
      "IbmQpuBackend.run_experiment: implement circuit compilation + execution "
      "for your specific DPD experiment."
    )


# ---------------------------------------------------------------------------
# AWS Braket backend (skeleton)
# ---------------------------------------------------------------------------


class AwsBraketBackend(HardwareBackend):
  """
  Skeleton integration for AWS Braket.

  Configure AWS creds and set:
    - BRAKET_DEVICE_ARN
  """

  def __init__(self) -> None:
    super().__init__(target_id="aws-braket", name="AWS Braket")

  def run_experiment(self, cfg: Any, session: Any, *, dry_run: bool = False) -> SimKpis:
    if dry_run:
      return LocalSimBackend().run_experiment(cfg, session, dry_run=True)

    try:
      from braket.aws import AwsDevice
    except ImportError as exc:
      raise RuntimeError(
        "amazon-braket-sdk is not installed. "
        "Install with `pip install amazon-braket-sdk`."
      ) from exc

    device_arn = os.getenv("BRAKET_DEVICE_ARN", "").strip()
    if not device_arn:
      raise RuntimeError("BRAKET_DEVICE_ARN not set")

    device = AwsDevice(device_arn)

    # TODO: translate cfg into a Braket circuit/task.
    raise NotImplementedError(
      "AwsBraketBackend.run_experiment: implement task creation + result handling."
    )


# ---------------------------------------------------------------------------
# IonQ backend (skeleton)
# ---------------------------------------------------------------------------


class IonqBackend(HardwareBackend):
  """
  Skeleton integration for IonQ.

  Depending on whether you use IonQ directly or via a provider,
  you might use:
    - ionq python SDK
    - qiskit-ionq
  """

  def __init__(self) -> None:
    super().__init__(target_id="ionq", name="IonQ")

  def run_experiment(self, cfg: Any, session: Any, *, dry_run: bool = False) -> SimKpis:
    if dry_run:
      return LocalSimBackend().run_experiment(cfg, session, dry_run=True)

    # Don’t pretend to know your IonQ integration – keep this as an explicit TODO.
    raise NotImplementedError(
      "IonqBackend.run_experiment: wire this up to your IonQ integration "
      "(native SDK or qiskit-ionq)."
    )


# ---------------------------------------------------------------------------
# Classical-only backend (FPGA / DAQ)
# ---------------------------------------------------------------------------


class ClassicalFpgaBackend(HardwareBackend):
  """
  Skeleton integration for classical-only lab hardware
  (FPGA controllers, AWGs, digitizers, DAQs).

  Expected pattern:
    - you have a local driver library (pyvisa / custom TCP / vendor API)
    - you translate cfg into pulse programs
    - you stream those to hardware, read back metrics, produce KPIs
  """

  def __init__(self) -> None:
    super().__init__(target_id="classical-only", name="Classical FPGA/DAQ")

  def run_experiment(self, cfg: Any, session: Any, *, dry_run: bool = False) -> SimKpis:
    if dry_run:
      # For purely classical rigs, dry-run usually means "compile only".
      return LocalSimBackend().run_experiment(cfg, session, dry_run=True)

    # This is intentionally left as an explicit integration point.
    raise NotImplementedError(
      "ClassicalFpgaBackend.run_experiment: integrate with your FPGA/DAQ driver."
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, HardwareBackend] = {
  "sim-local": LocalSimBackend(),
  "ibm-qpu": IbmQpuBackend(),
  "aws-braket": AwsBraketBackend(),
  "ionq": IonqBackend(),
  "classical-only": ClassicalFpgaBackend(),
}


def get_backend(target_id: str) -> HardwareBackend:
  """
  Resolve hardware_target from RunConfiguration -> backend instance.
  """
  if target_id not in _BACKENDS:
    raise KeyError(f"Unknown hardware_target={target_id!r}")
  return _BACKENDS[target_id]

"""
SynQc Temporal Dynamics Series — Super Backend

Single-file backend that matches the "SynQc Temporal Dynamics Series —
Control Panel v0.2" HTML controller.

Features
--------
- Session model that mirrors the front-end controls:
  * hardware target & preset
  * drive envelope
  * probe strength & duration
  * adaptive rule
  * objective
  * free-form notes

- Simulation engine that produces physically-inspired KPIs:
  * DPD Fidelity (0–1)
  * Loop latency (microseconds)
  * Probe back-action (0–1 scale)
  * Shot budget usage (out of a configurable limit)

- Persistent state:
  * Sessions and runs stored under SYNQC_STATE_DIR (JSON files)
  * Shot budget tracked per session
  * Logs kept per session

- Export endpoint that replicates the front-end snapshot payload:
  * JSON, CSV, or "notebook" (a ready-to-paste Python cell)

Tech stack
----------
- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic v2+
- python-dotenv
- NumPy

To run
------
1. Install dependencies (example):

   pip install fastapi uvicorn[standard] pydantic>=2.7.0 python-dotenv numpy

2. Optional: create a .env next to this file with:

   SYNQC_API_PREFIX=/api/v1/synqc
   SYNQC_STATE_DIR=./synqc_state
   SYNQC_ALLOWED_ORIGINS=*

   SYNQC_HOST=0.0.0.0
   SYNQC_PORT=8000

3. Start server:

   uvicorn synqc_tds_super_backend:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


# -------------------------------------------------------------------------
# Environment & configuration
# -------------------------------------------------------------------------


load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


API_PREFIX = _env("SYNQC_API_PREFIX", "/api/v1/synqc").rstrip("/")
STATE_DIR = Path(_env("SYNQC_STATE_DIR", "./synqc_state")).resolve()
ALLOWED_ORIGINS_RAW = _env("SYNQC_ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()] or ["*"]

# Defaults used by the UI
DEFAULT_SHOT_LIMIT = 50_000


# -------------------------------------------------------------------------
# Enumerations that mirror the HTML controller
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


# -------------------------------------------------------------------------
# Core configuration & KPI models
# -------------------------------------------------------------------------


class RunConfiguration(BaseModel):
    """
    Configuration that corresponds one-to-one with front-end controls.
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
    # Server-computed convenience fields
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
        frac = 0.0 if shot_limit <= 0 else min(1.0, max(0.0, shots_used / shot_limit))
        return cls(
            fidelity=float(fidelity),
            latency_us=float(latency_us),
            backaction=float(backaction),
            shots_used=int(shots_used),
            shot_limit=int(shot_limit),
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
    # log messages attached only to this run
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
    # cached KPIs from the last non-dry run, if available
    last_kpis: Optional[KpiBundle] = None

    def add_log(self, message: str) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        entry = f"[{now}] {message}"
        self.logs.append(entry)
        # avoid unbounded growth
        if len(self.logs) > 1000:
            # keep the most recent 1000 lines
            self.logs = self.logs[-1000:]


# -------------------------------------------------------------------------
# Simulation engine — SynQc-style, but compact
# -------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """
    Parameters that govern the synthetic behavior.

    The goal is to emulate reasonable qualitative behavior for:
    - different hardware targets
    - different objectives
    - the trade-off between probe strength and back-action
    """

    base_latency_sim_local: float = 10.0
    base_latency_classical: float = 25.0
    base_latency_quantum: float = 80.0

    base_fidelity_sim_local: float = 0.99
    base_fidelity_classical: float = 0.985
    base_fidelity_quantum: float = 0.97

    shot_cost_baseline: int = 800
    shot_cost_per_ns: float = 0.3  # extra shots per ns of probe duration
    random_snr_db: float = 1.5  # jitter in effective SNR in dB units


class SynQcEngine:
    """
    Simple engine that maps (config, session state) → KPIs.

    Everything here is deterministic math + small noise, not full quantum
    simulation. It encodes our "research sense" of how these quantities move.
    """

    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()

    # Public API -------------------------------------------------------

    def run(self, session: SessionState, mode: RunMode) -> RunRecord:
        """
        Execute one run (or dry-run).

        For dry-run we still synthesize KPIs, but we do NOT charge shot
        budget.
        """
        run_id = self._new_run_id(session.session_id)
        created_at = datetime.utcnow()

        cfg = session.config

        kpis = self._simulate_kpis(cfg, session, count_shots=(mode == RunMode.RUN))
        events = self._explain_kpis(cfg, kpis, mode)

        # Update session state
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

        # Add a top-level log line
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
            events=events,
        )

    # Internals --------------------------------------------------------

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
    ) -> KpiBundle:
        # --- Baseline by hardware target --------------------------------
        ecfg = self.cfg

        if cfg.hardware_target == HardwareTarget.SIM_LOCAL:
            base_latency = ecfg.base_latency_sim_local
            base_fid = ecfg.base_fidelity_sim_local
        elif cfg.hardware_target == HardwareTarget.CLASSICAL_ONLY:
            base_latency = ecfg.base_latency_classical
            base_fid = ecfg.base_fidelity_classical
        else:
            base_latency = ecfg.base_latency_quantum
            base_fid = ecfg.base_fidelity_quantum

        # --- Adjustments by hardware preset -----------------------------
        if cfg.hardware_preset == HardwarePreset.TRANSMON_DEFAULT:
            base_fid += 0.0
            base_latency += 0.0
        elif cfg.hardware_preset == HardwarePreset.FLUXONIUM_PILOT:
            base_fid -= 0.005
            base_latency += 15.0
        elif cfg.hardware_preset == HardwarePreset.ION_CHAIN:
            base_fid += 0.005
            base_latency += 30.0
        elif cfg.hardware_preset == HardwarePreset.NEUTRAL_ATOM:
            base_fid -= 0.01
            base_latency += 45.0

        # --- Objective / adaptive rule tweaks ---------------------------
        latency = base_latency
        fid = base_fid

        if cfg.objective == Objective.MAXIMIZE_FIDELITY:
            fid += 0.005
            latency += 10.0
        elif cfg.objective == Objective.MINIMIZE_LATENCY:
            fid -= 0.007
            latency -= 12.0
        elif cfg.objective == Objective.INFO_VS_DAMAGE:
            # Balanced; small nudges
            fid += 0.002
            latency += 3.0
        elif cfg.objective == Objective.STABILITY_WINDOW:
            fid += 0.001
            latency += 5.0

        if cfg.adaptive_rule == AdaptiveRule.RL:
            # RL tends to be heavier compute; slightly slower loop
            latency += 8.0
        elif cfg.adaptive_rule == AdaptiveRule.KALMAN:
            # Filtered estimates help fidelity a bit
            fid += 0.003
        elif cfg.adaptive_rule == AdaptiveRule.BAYES:
            # More branching, slightly slower
            latency += 5.0

        # --- Probe strength & duration trade-offs -----------------------
        # There is an "optimal" probe_strength ~ 0.2; too low or too high hurts.
        eps = max(0.0, min(1.0, cfg.probe_strength))
        deviation = (eps - 0.2) / 0.2  # 0 at optimum

        # Quadratic penalty on fidelity
        fid -= 0.015 * deviation * deviation

        # Back-action grows with probe strength; mild penalty from long windows
        backaction = 0.08 + 0.4 * (eps ** 1.1) + 0.00005 * cfg.probe_duration_ns
        backaction = max(0.0, min(1.0, backaction))

        # Latency gets a small bump from probe window length
        latency += 0.01 * (cfg.probe_duration_ns / 10.0)

        # Envelope type influences timing slightly
        if cfg.drive_envelope == DriveEnvelope.GAUSSIAN:
            latency += 0.0
        elif cfg.drive_envelope == DriveEnvelope.SQUARE:
            latency -= 3.0
            fid -= 0.003
        elif cfg.drive_envelope == DriveEnvelope.DRAG:
            latency += 3.0
            fid += 0.004
        elif cfg.drive_envelope == DriveEnvelope.COSINE:
            latency += 1.0

        # --- Noise model via "effective SNR" ----------------------------
        # Treat SNR as an internal latent variable influencing both fidelity and
        # latency. We don't need to expose it directly; we only need consistent
        # jitter.
        rng = np.random.default_rng()
        snr_jitter_db = rng.normal(loc=0.0, scale=self.cfg.random_snr_db)
        snr_factor = math.exp(snr_jitter_db / 20.0)  # convert dB to multiplicative

        # Higher SNR → better fidelity, more stable latency
        fid *= min(1.02, max(0.95, snr_factor))
        latency /= min(1.05, max(0.95, snr_factor))

        # --- Shots used for this run ------------------------------------
        # Cost model: baseline + proportional to probe duration.
        base_shots = self.cfg.shot_cost_baseline
        extra = int(self.cfg.shot_cost_per_ns * cfg.probe_duration_ns)
        shots_this_run = max(100, base_shots + extra)

        # Dry-run doesn't consume shots
        prior_shots = session.shots_used
        if count_shots:
            total_shots = prior_shots + shots_this_run
        else:
            total_shots = prior_shots

        # Clip + sanitize
        fid = max(0.0, min(0.9999, fid))
        latency = max(1.0, float(latency))

        return KpiBundle.from_raw(
            fidelity=fid,
            latency_us=latency,
            backaction=backaction,
            shots_used=total_shots,
            shot_limit=session.shot_limit,
        )

    def _explain_kpis(
        self,
        cfg: RunConfiguration,
        kpis: KpiBundle,
        mode: RunMode,
    ) -> List[str]:
        """
        Generate human-readable events that describe the resulting KPIs.
        """

        events: List[str] = []
        label = "RUN" if mode == RunMode.RUN else "DRY-RUN"
        events.append(f"[{label}] Config objective={cfg.objective.value}, "
                      f"adaptive={cfg.adaptive_rule.value}, envelope={cfg.drive_envelope.value}")

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


# -------------------------------------------------------------------------
# Persistent storage
# -------------------------------------------------------------------------


class StateStore:
    """
    File-backed store for sessions and runs.

    Layout under STATE_DIR:
    - sessions.json : list of SessionState objects (serialized)
    - runs/         : one JSON per RunRecord, named {run_id}.json
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

        self.sessions: Dict[str, SessionState] = {}
        self._load_sessions()

    # --- session persistence -------------------------------------------

    def _sessions_path(self) -> Path:
        return self.root / "sessions.json"

    def _load_sessions(self) -> None:
        path = self._sessions_path()
        if not path.exists():
            self.sessions = {}
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                self.sessions = {}
                return
            loaded: Dict[str, SessionState] = {}
            for item in raw:
                try:
                    st = SessionState.model_validate(item)
                    loaded[st.session_id] = st
                except Exception:
                    continue
            self.sessions = loaded
        except Exception:
            self.sessions = {}

    def _save_sessions(self) -> None:
        data = [s.model_dump(mode="json") for s in self.sessions.values()]
        self._sessions_path().write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    # --- runs persistence ----------------------------------------------

    def _run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"{run_id}.json"

    def save_run(self, run: RunRecord) -> None:
        path = self._run_path(run.run_id)
        path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")

    def load_run(self, run_id: str) -> RunRecord:
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(run_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RunRecord.model_validate(raw)

    # --- sessions API --------------------------------------------------

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def upsert_session(self, session: SessionState) -> SessionState:
        self.sessions[session.session_id] = session
        self._save_sessions()
        return session

    def all_sessions(self) -> List[SessionState]:
        return list(self.sessions.values())


# -------------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------------


app = FastAPI(
    title="SynQc Temporal Dynamics Series — Backend",
    description=(
        "Backend API that matches the SynQc Temporal Dynamics Series "
        "Control Panel v0.2 HTML controller. Provides session management, "
        "synthetic DPD KPIs, logging, and export."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = StateStore(STATE_DIR)
engine = SynQcEngine()


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
        frac = 0.0 if st.shot_limit <= 0 else min(1.0, st.shots_used / st.shot_limit)
        return cls(
            session_id=st.session_id,
            created_at=st.created_at,
            last_updated_at=st.last_updated_at,
            status=st.status,
            status_text=st.status_text,
            mode_label=st.mode_label,
            shot_limit=st.shot_limit,
            shots_used=st.shots_used,
            shots_used_fraction=frac,
            last_run_id=st.last_run_id,
            last_kpis=st.last_kpis,
            config=st.config,
        )


class RunRequest(BaseModel):
    mode: RunMode = RunMode.RUN


class RunResponse(BaseModel):
    run: RunRecord
    session: SessionSummary


class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    NOTEBOOK = "notebook"


# -------------------------------------------------------------------------
# Helper functions (session creation, mode label)
# -------------------------------------------------------------------------


def _derive_mode_label(target: HardwareTarget) -> str:
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
    """
    Lightweight health check plus basic configuration info.
    """
    return {
        "status": "ok",
        "version": app.version,
        "api_prefix": API_PREFIX,
        "state_dir": str(STATE_DIR),
        "allowed_origins": ALLOWED_ORIGINS,
    }


# --- Session CRUD -----------------------------------------------------


@app.get(f"{API_PREFIX}/sessions", response_model=List[SessionSummary])
def list_sessions() -> List[SessionSummary]:
    """
    List all known sessions. Useful for debugging and multi-session workflows.
    """
    return [SessionSummary.from_state(s) for s in store.all_sessions()]


@app.post(f"{API_PREFIX}/sessions", response_model=SessionSummary)
def create_or_update_session(req: SessionCreateRequest) -> SessionSummary:
    """
    Create a new session or update an existing one.

    The front-end should call this whenever the user changes a major
    configuration (hardware target/preset, envelopes, objectives, etc).
    """
    now = datetime.utcnow()

    if req.session_id:
        existing = store.get_session(req.session_id)
    else:
        existing = None

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
        # Update configuration in-place
        st = existing
        st.config = req.config
        st.mode_label = _derive_mode_label(req.config.hardware_target)
        st.last_updated_at = now
        st.add_log("Configuration updated from front-end.")

    store.upsert_session(st)
    return SessionSummary.from_state(st)


@app.get(f"{API_PREFIX}/sessions/{{session_id}}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    """
    Get full state for a session.
    """
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
    """
    Return recent log lines for the session. This drives the 'Run Log & Events'
    panel in the HTML controller.
    """
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    logs = st.logs[-limit:]
    return {"session_id": session_id, "lines": logs}


@app.delete(f"{API_PREFIX}/sessions/{{session_id}}/logs")
def clear_logs(session_id: str) -> Dict[str, Any]:
    """
    Clear all logs for the session (used by 'Clear Log' button).
    """
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
def launch_run(session_id: str, req: RunRequest) -> RunResponse:
    """
    Launch a SynQc run or dry-run for the given session.

    The HTML buttons map to:
    - Launch SynQc Run  → mode='run'
    - Dry-Run (no hardware) → mode='dryrun'
    """
    st = store.get_session(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    if st.status == SessionStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Session is already running.")

    st.status = SessionStatus.RUNNING
    st.status_text = "Running · SynQc DPD sequence in progress"
    st.last_updated_at = datetime.utcnow()
    st.add_log(f"Run requested with mode={req.mode.value}.")
    store.upsert_session(st)

    # For now, we execute synchronously. If later you want to offload to a
    # background worker, this is the hook.
    try:
        run = engine.run(st, mode=req.mode)
    except Exception as exc:  # pragma: no cover - defensive
        st.status = SessionStatus.ERROR
        st.status_text = f"Error during run: {exc}"
        st.add_log(f"Run failed with error: {exc}")
        store.upsert_session(st)
        raise

    # Persist results
    store.save_run(run)
    store.upsert_session(st)
    return RunResponse(run=run, session=SessionSummary.from_state(st))


@app.post(f"{API_PREFIX}/sessions/{{session_id}}/kill")
def kill_run(session_id: str) -> Dict[str, Any]:
    """
    Kill-switch endpoint. In this synchronous demo engine, this mainly
    records the user's intention and updates status/logs.
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
    """
    Retrieve a stored RunRecord.
    """
    try:
        return store.load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")


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
    Export a snapshot that mirrors the front-end's `buildExportPayload`:

    {
      "sessionId": ...,
      "mode": ...,
      "hardwareTarget": ...,
      "hardwarePreset": ...,
      "driveEnvelope": ...,
      "probeStrength": ...,
      "probeDurationNs": ...,
      "adaptiveRule": ...,
      "objective": ...,
      "kpis": { ... },
      "notes": ...,
      "exportedAt": ...
    }
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
        # Flatten the structure into a single CSV row
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
        import io as _io
        buf = _io.StringIO()
        writer = csv_writer = None
        # Write header + row manually to avoid quoting headaches.
        header = list(flat.keys())
        writer = _io.StringIO()
        # Simple CSV construction: quote strings with commas or quotes
        def _csv_escape(value: Any) -> str:
            s = "" if value is None else str(value)
            if any(c in s for c in [",", "\"", "\n", "\r"]):
                s = '"' + s.replace('"', '""') + '"'
            return s

        header_line = ",".join(header)
        row_line = ",".join(_csv_escape(flat[k]) for k in header)
        csv_text = header_line + "\n" + row_line + "\n"
        return {
            "format": "csv",
            "payload": csv_text,
        }

    if format == ExportFormat.NOTEBOOK:
        # Provide a ready-to-paste Python cell
        cell = (
            "# SynQc TDS snapshot\n"
            "snapshot = "
            + json.dumps(data, indent=2)
            + "\n\n"
            "# Use `snapshot` inside your Jupyter pipeline.\n"
        )
        return {
            "format": "notebook",
            "payload": cell,
        }

    # Default JSON
    return {
        "format": "json",
        "payload": data,
    }


# -------------------------------------------------------------------------
# Main entry-point
# -------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    host = _env("SYNQC_HOST", "0.0.0.0")
    try:
        port = int(_env("SYNQC_PORT", "8000"))
    except ValueError:
        port = 8000

    uvicorn.run("synqc_tds_super_backend:app", host=host, port=port, reload=True)
