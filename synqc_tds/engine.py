from __future__ import annotations

from .hardware_backends import BackendError, get_backend
from .models import KpiBundle, RunConfiguration, SessionState


class SynQcEngine:
    """High-level interface between API, scheduler, and hardware backends."""

    def run(
        self,
        config: RunConfiguration,
        session: SessionState,
        *,
        mode: str = "run",
    ) -> KpiBundle:
        if mode not in {"run", "dryrun"}:
            raise ValueError(f"Unsupported mode: {mode!r}")

        backend = get_backend(config.hardware_target)
        dry = mode == "dryrun"

        # In the future: plug in scheduler, probes, demod, adapt, etc.
        kpis = backend.run_experiment(config, session, dry_run=dry)
        return kpis
