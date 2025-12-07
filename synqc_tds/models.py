from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunConfiguration(BaseModel):
    """User-facing SynQc run configuration."""

    hardware_target: str
    sequence_id: str = "default"
    cycles: int = 1
    shot_limit: int = 1024
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Session / run history context."""

    session_id: str
    seed: int | None = None
    cache: dict[str, Any] = Field(default_factory=dict)


class KpiBundle(BaseModel):
    """Key performance indicators for a SynQc experiment."""

    fidelity: float
    latency_us: float
    backaction: float
    shots_used: int
    shot_limit: int
    shots_used_fraction: float


class RunRequest(BaseModel):
    """JSON payload from the HTML controller → backend."""

    config: RunConfiguration
    session: SessionState
    mode: Literal["run", "dryrun"] = "run"


class RunResponse(BaseModel):
    """Returned to the HTML controller."""

    kpis: KpiBundle
    backend_name: str
    mode: Literal["run", "dryrun"]
