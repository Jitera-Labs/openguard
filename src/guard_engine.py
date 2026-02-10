"""Guard application engine - applies guards to requests based on matching rules."""

import importlib
from typing import List, Tuple

from src import log
from src.chat import Chat
from src.guards import GuardBlockedError, GuardRule
from src.llm import LLM
from src.selection import match_filter

logger = log.setup_logger(__name__)


def apply_guards(chat: Chat, llm: LLM, guards: List[GuardRule]) -> Tuple[Chat, List[str]]:
    """
    Apply guards to a chat session in order.

    Args:
        chat: Chat object to process
        llm: LLM object containing parameters
        guards: List of guard rules to apply

    Returns:
        Tuple of (modified_chat, audit_logs)
    """
    audit_logs = []

    # Construct context for matching
    # This allows guards to match against model, params, messages, etc.
    match_context = {"model": llm.model, "messages": chat.history(), **llm.params}

    for guard_idx, guard in enumerate(guards):
        # Check if guard matches the context
        try:
            matches = match_filter(match_context, guard.match)
        except Exception as e:
            logger.error(f"Error matching guard {guard_idx}: {e}")
            matches = False

        if not matches:
            continue

        logger.debug(f"Guard {guard_idx} matched, applying {len(guard.apply)} action(s)")

        # Apply each action in the guard
        for action in guard.apply:
            action_type = action.type
            action_config = action.config

            try:
                # Dynamically import the guard type module
                module_name = f"src.guard_types.{action_type}"
                guard_module = importlib.import_module(module_name)

                # Call the apply function
                # Signature: apply(chat, llm, config) -> List[str]
                action_logs = guard_module.apply(chat, llm, action_config)

                if action_logs:
                    audit_logs.extend(action_logs)

                logger.debug(f"Applied guard action '{action_type}'")

            except GuardBlockedError:
                raise
            except ModuleNotFoundError:
                logger.error(f"Guard type '{action_type}' not found")
            except AttributeError:
                logger.error(f"Guard type '{action_type}' does not have an 'apply' function")
            except Exception as e:
                logger.error(f"Error applying guard action '{action_type}': {e}")

    return chat, audit_logs


def log_audit(audit_logs: List[str]):
    """
    Log audit entries.

    Args:
        audit_logs: List of audit log entries
    """
    for entry in audit_logs:
        logger.info(f"[AUDIT] {entry}")
