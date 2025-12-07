from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict

from .models import KpiBundle, RunConfiguration, SessionState


class BackendError(Exception):
    """Raised when a backend cannot fulfill a request."""


class Backend:
    """Abstract backend interface."""

    name: str

    def run_experiment(
        self,
        config: RunConfiguration,
        session: SessionState,
        *,
        dry_run: bool = False,
    ) -> KpiBundle:
        raise NotImplementedError


@dataclass
class MockBackend(Backend):
    """Synthetic backend for local testing and UI development."""

    name: str = "mock-local"

    def run_experiment(
        self,
        config: RunConfiguration,
        session: SessionState,
        *,
        dry_run: bool = False,
    ) -> KpiBundle:
        # deterministic-ish seed per session to keep runs stable
        rnd = random.Random(session.seed or hash(session.session_id))

        base_fidelity = 0.995
        # penalize long sequences and low shot budgets
        complexity_penalty = 0.0005 * max(config.cycles - 1, 0)
        shot_bonus = 0.0002 * math.log10(max(config.shot_limit, 10))

        fidelity = max(
            0.0,
            min(
                1.0,
                base_fidelity - complexity_penalty + shot_bonus + rnd.uniform(-0.002, 0.0),
            ),
        )

        # mock latency: 5 µs baseline + per-cycle cost
        latency_us = 5.0 + config.cycles * 2.0 + rnd.uniform(0.0, 1.0)

        # backaction: small value that grows with cycles
        backaction = max(0.0, min(1.0, 0.01 * config.cycles + rnd.uniform(0.0, 0.01)))

        # pretend each cycle uses 256 shots up to shot_limit
        estimated_shots = config.cycles * 256
        shots_used = min(config.shot_limit, estimated_shots)
        shot_limit = config.shot_limit
        shots_used_fraction = shots_used / shot_limit if shot_limit > 0 else 0.0

        # in dry-run mode, simulate a bit of delay but don't talk to real hardware
        if dry_run:
            time.sleep(0.01)

        return KpiBundle(
            fidelity=fidelity,
            latency_us=latency_us,
            backaction=backaction,
            shots_used=shots_used,
            shot_limit=shot_limit,
            shots_used_fraction=shots_used_fraction,
        )


# registry

_BACKENDS: Dict[str, Backend] = {
    "mock-local": MockBackend(),
    # later: "ibm-q": IbmBackend(...),
    #        "aws-braket": BraketBackend(...),
    #        "lab-rig-1": LabRigBackend(...),
}


def get_backend(target: str) -> Backend:
    try:
        return _BACKENDS[target]
    except KeyError as exc:
        raise BackendError(f"Unknown hardware_target: {target!r}") from exc
