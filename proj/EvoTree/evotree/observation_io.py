"""Bounded retry of read-only VM observation queries; never replay actions."""
from collections.abc import Mapping
import time


def read_python_output(controller, command: str, attempts: int = 3, delay: float = 1.0) -> str:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    reason = "no response"
    for index in range(attempts):
        try:
            result = controller.execute_python_command(command)
        except Exception as exc:
            # A transient controller/HTTP failure is still a failed read; retry
            # the same read-only query without replaying the user action.
            reason = type(exc).__name__ + ': ' + str(exc)[:160]
            result = None
        if isinstance(result, Mapping):
            output = result.get("output")
            if (isinstance(output, str) and result.get("status") != "error"
                    and result.get("returncode", 0) == 0):
                return output.strip()
            reason = "missing output or unsuccessful observation command"
        if index + 1 < attempts:
            time.sleep(delay)
    raise RuntimeError(f"Read-only VM observation query failed after {attempts} attempts: {reason}")
