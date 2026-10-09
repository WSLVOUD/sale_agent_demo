"""Snapshot and restore conversation state for superseded turns."""
from __future__ import annotations

import copy
import logging

from src.memory.store import memory

logger = logging.getLogger(__name__)


def snapshot_turn_state(session_id: str) -> dict:
    """Capture the state that a superseded turn may mutate."""
    from .dialogue import get_conversation_state

    snapshot: dict = {"profile": None, "requirements": None, "conversation": None}
    try:
        stored = memory.get_requirement_profile(session_id)
        if stored is not None:
            snapshot["profile"] = (
                stored.model_dump() if hasattr(stored, "model_dump") else stored
            )
    except Exception as exc:  # pragma: no cover - defensive recovery
        logger.warning("snapshot profile failed: %s", exc)
    try:
        snapshot["requirements"] = copy.deepcopy(memory.get_requirements(session_id))
    except Exception as exc:  # pragma: no cover - defensive recovery
        logger.warning("snapshot requirements failed: %s", exc)
    try:
        snapshot["conversation"] = copy.deepcopy(
            get_conversation_state(session_id).__dict__
        )
    except Exception as exc:  # pragma: no cover - defensive recovery
        logger.warning("snapshot conversation failed: %s", exc)
    return snapshot


def restore_turn_state(session_id: str, snapshot: dict) -> None:
    """Restore the state captured before a superseded turn began."""
    if not snapshot:
        return

    profile_data = snapshot.get("profile")
    if profile_data is not None:
        try:
            from .models.requirement import RequirementProfile

            memory.set_requirement_profile(
                session_id, RequirementProfile.model_validate(profile_data)
            )
        except Exception as exc:  # pragma: no cover - defensive recovery
            logger.warning("restore profile failed: %s", exc)

    requirements = snapshot.get("requirements")
    if requirements is not None:
        try:
            memory.set_requirements(session_id, copy.deepcopy(requirements))
        except Exception as exc:  # pragma: no cover - defensive recovery
            logger.warning("restore requirements failed: %s", exc)

    conversation = snapshot.get("conversation")
    if conversation is not None:
        try:
            from .dialogue import get_conversation_state

            get_conversation_state(session_id).__dict__.update(
                copy.deepcopy(conversation)
            )
        except Exception as exc:  # pragma: no cover - defensive recovery
            logger.warning("restore conversation failed: %s", exc)
