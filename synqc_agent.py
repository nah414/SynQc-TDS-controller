# synqc_agent.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class AgentSuggestion:
    """
    Result of a single agent suggestion.

    recommended_config:
        A partial or full RunConfiguration, expressed as a plain dict using
        the backend's field names (snake_case):
        {
            "probe_strength": 0.18,
            "probe_duration_ns": 140,
            "adaptive_rule": "kalman",
            "objective": "maximize-fidelity",
            ...
        }

    rationale:
        Human-readable explanation of why the agent suggested this patch.

    warnings:
        Free-form strings, e.g. about shot usage, aggressive trade-offs, etc.
    """
    recommended_config: Dict[str, Any]
    rationale: str
    warnings: List[str]


class SynQcAgent:
    """
    LLM-powered experiment copilot for the SynQc Temporal Dynamics Series.

    - Reads a session snapshot (config + KPIs + shot budget).
    - Receives a high-level goal ("maximize fidelity under shot budget 20k").
    - Proposes a *patch* to the RunConfiguration and explains why.
    - Never directly touches hardware; it's advisory only.

    The caller is responsible for:
    - Applying the patch to an existing RunConfiguration.
    - Enforcing hard safety limits (which your backend already does).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        self.default_model = model or os.getenv(
            "SYNQC_AGENT_MODEL",
            # Adjust this to whatever you actually use; this is just a sane default.
            "gpt-4-mini",
        )

        env_temp = os.getenv("SYNQC_AGENT_TEMPERATURE")
        if env_temp is not None:
            try:
                temperature = float(env_temp)
            except ValueError:
                pass

        self.default_temperature = max(0.0, min(1.0, float(temperature)))
        # Uses OPENAI_API_KEY from env by default
        self.client = OpenAI()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def suggest(
        self,
        *,
        session_snapshot: Dict[str, Any],
        goal: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AgentSuggestion:
        """
        Ask the agent for a configuration patch given the current session state.

        session_snapshot:
            A JSON-serializable dict built from SessionState + KPIs. The backend
            decides what to put here.

        goal:
            Free-text goal from the user ("minimize latency while staying above
            fidelity 0.97" etc.).

        Returns AgentSuggestion (config patch + rationale + warnings).
        """
        if not goal or not goal.strip():
            raise ValueError("Agent goal must be non-empty.")

        model_name = model or self.default_model
        temp = self._resolve_temperature(temperature)

        system_msg = self._build_system_message()
        user_msg = self._build_user_message(session_snapshot, goal)

        completion = self.client.chat.completions.create(
            model=model_name,
            temperature=temp,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = completion.choices[0].message.content
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            snippet = content[:200] if content else "<empty>"
            raise RuntimeError(f"Agent returned non-JSON content: {snippet}") from exc

        cfg_patch = raw.get("recommended_config") or raw.get("config_patch") or {}
        rationale = raw.get("rationale") or raw.get("explanation") or ""
        warnings = raw.get("warnings") or []

        if not isinstance(cfg_patch, dict):
            raise RuntimeError("Agent result missing 'recommended_config' object.")
        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        return AgentSuggestion(
            recommended_config=cfg_patch,
            rationale=str(rationale),
            warnings=[str(w) for w in warnings],
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_temperature(self, override: Optional[float]) -> float:
        if override is None:
            return self.default_temperature
        try:
            t = float(override)
        except (TypeError, ValueError):
            return self.default_temperature
        return max(0.0, min(1.0, t))

    def _build_system_message(self) -> str:
        return (
            "You are an expert quantum control assistant for the SynQc Temporal "
            "Dynamics Series (TDS) drive–probe–drive controller.\n\n"
            "You are given:\n"
            "- The current SynQc session snapshot (JSON).\n"
            "- A high-level experimental goal from the user.\n\n"
            "You must propose *small, safe, incremental* adjustments to the "
            "RunConfiguration for the NEXT run.\n\n"
            "Constraints:\n"
            "- Never exceed any stated parameter limits.\n"
            "- Prefer tuning probe_strength and probe_duration_ns gently, not huge jumps.\n"
            "- You may change: hardware_target, hardware_preset, drive_envelope, "
            "  probe_strength, probe_duration_ns, adaptive_rule, objective, notes.\n"
            "- Use the backend's snake_case field names exactly.\n"
            "- Do NOT attempt to schedule multiple runs; you only choose the next one.\n\n"
            "Output MUST be a JSON object with this structure:\n"
            "{\n"
            '  "recommended_config": {\n'
            "    // partial RunConfiguration patch, using snake_case keys\n"
            "  },\n"
            '  "rationale": "short explanation",\n'
            "  \"warnings\": [\"optional\", \"warnings\"]\n"
            "}\n"
            "Do not wrap the JSON in backticks or add extra commentary."
        )

    def _build_user_message(
        self,
        session_snapshot: Dict[str, Any],
        goal: str,
    ) -> str:
        snapshot_json = json.dumps(session_snapshot, indent=2)
        return (
            "High-level goal from experimenter:\n"
            f"{goal.strip()}\n\n"
            "Current SynQc TDS session snapshot (JSON):\n"
            f"{snapshot_json}\n\n"
            "Decide on a config patch for the NEXT run only. Be conservative and "
            "respect shot budgets."
        )
