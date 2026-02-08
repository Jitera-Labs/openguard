"""Guard application engine - applies guards to requests based on matching rules."""
from typing import Dict, List, Tuple, Any
import importlib

from src.selection import match_filter
from src import log
from src.guards import GuardRule

logger = log.setup_logger(__name__)


def apply_guards(payload: dict, guards: List[GuardRule]) -> Tuple[dict, List[str]]:
    """
    Apply guards to a payload in order.

    Args:
        payload: Request payload to process
        guards: List of guard rules to apply

    Returns:
        Tuple of (modified_payload, audit_logs)
    """
    audit_logs = []
    current_payload = payload

    for guard_idx, guard in enumerate(guards):
        # Check if guard matches the payload
        try:
            matches = match_filter(current_payload, guard.match)
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
                module_name = f'src.guard_types.{action_type}'
                guard_module = importlib.import_module(module_name)

                # Call the apply function
                current_payload, action_logs = guard_module.apply(current_payload, action_config)
                audit_logs.extend(action_logs)

                logger.debug(f"Applied guard action '{action_type}' with {len(action_logs)} audit log(s)")

            except ModuleNotFoundError:
                logger.error(f"Guard type '{action_type}' not found")
            except AttributeError:
                logger.error(f"Guard type '{action_type}' does not have an 'apply' function")
            except Exception as e:
                logger.error(f"Error applying guard action '{action_type}': {e}")

    return current_payload, audit_logs


def log_audit(audit_logs: List[str]):
    """
    Log audit entries.

    Args:
        audit_logs: List of audit log entries
    """
    for entry in audit_logs:
        logger.info(f"[AUDIT] {entry}")
