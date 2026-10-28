"""Observed preconditions for Calc copy advice, without binding test answers.

The tool contract copies FROM the active sheet. Parameter-free action stages
alone cannot distinguish switching to the source from switching to the target.
Unknown roles fail closed for dependent advice, while task workflows survive.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import re


COPY_TOOL = "libreoffice_calc2.copy_cells_between_sheets"
SWITCH_TOOL = "libreoffice_calc.switch_active_sheet"
_COPY_TASK = re.compile(
    r'^Copy the ["“](?P<column>[^"”]+)["”] column\b.*?'
    r'\bto a new sheet named ["“](?P<target>[^"”]+)["”]\.?$', re.IGNORECASE)


def parse_app_info(value) -> dict:
    """Accept a mapping or identical repeated JSON stdout, never partial data."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    decoder, values, remaining = json.JSONDecoder(), [], value.strip()
    try:
        while remaining:
            item, end = decoder.raw_decode(remaining)
            values.append(item)
            remaining = remaining[end:].strip()
    except ValueError:
        return {}
    if values and isinstance(values[0], dict) and all(v == values[0] for v in values):
        return values[0]
    return {}


def current_conditions(instruction: str, observation: dict) -> dict:
    """Bind source/target roles using current task + current structured state.

    `app_info.name` is Calc's automatic CURRENT active-sheet observation.
    An arbitrary env_info(sheet_name=...) tool response is not an active-sheet
    observation and must not be passed here as one. No previous action argument
    is treated as proof that a switch really took effect.
    """
    conditions = {}
    progress = observation.get("state_progress", {})
    if isinstance(progress, Mapping) and type(progress.get("copy_call_observed")) is bool:
        conditions["copy_call_observed"] = progress["copy_call_observed"]
    task = _COPY_TASK.fullmatch(instruction.strip())
    if task is None:
        return conditions
    if observation.get('observation_provenance', {}).get('status') in {'unstable', 'unavailable'}:
        return conditions
    info = parse_app_info(observation.get("app_info"))
    active = info.get("active_sheet", info.get("name"))
    if not isinstance(active, str) or not active:
        return conditions
    if active == task["target"]:
        conditions["active_sheet_role"] = "target"
        return conditions
    data = info.get("data")
    if not isinstance(data, str):
        return conditions
    # Exact column-header evidence, not a substring occurring in a data cell.
    headers = []
    for line in data.splitlines():
        if re.match(r'^\s*1\s*\|', line):
            headers = [cell.strip() for cell in line.split('|')[1:]]
            break
    if task["column"] in headers:
        conditions["active_sheet_role"] = "source"
    return conditions


def historical_conditions(history) -> list[dict]:
    """Derive role guards from historical parameters and measured ordering.

    Multiple source/destination relationships are intentionally left unresolved.
    Even then, a copy candidate always requires observed source-role evidence.
    This derived index works with old frozen banks; their files stay untouched.
    """
    targets = {str(step.action.args.get("target_sheet_name", "")) for step in history.steps
               if step.action.name == COPY_TOOL}
    if len(targets) != 1 or not next(iter(targets), ""):
        return [{} for _ in history.steps]
    target = next(iter(targets))
    sources, last_sheet = set(), None
    for step in history.steps:
        action = step.action
        if step.note:
            last_sheet = None
        elif action.name == SWITCH_TOOL:
            last_sheet = str(action.args.get("sheet_name", "")) or None
        elif action.name == COPY_TOOL:
            if last_sheet and last_sheet != target:
                sources.add(last_sheet)
            last_sheet = None
        elif action.kind.value == "gui":
            last_sheet = None
    if len(sources) != 1:
        return [{} for _ in history.steps]
    source = next(iter(sources))
    role, copied, result = None, False, []
    for step in history.steps:
        conditions = {"copy_call_observed": copied}
        if role:
            conditions["active_sheet_role"] = role
        result.append(conditions)
        action = step.action
        if step.note:
            role = None
            if action.name == COPY_TOOL:
                copied = False
        elif action.name == SWITCH_TOOL:
            name = str(action.args.get("sheet_name", ""))
            role = "target" if name == target else "source" if name == source else None
        elif action.name == COPY_TOOL:
            copied = True
            # The copy tool's post-action active sheet is not guaranteed here.
            role = None
        elif action.name not in {"libreoffice_calc.env_info", "libreoffice_calc.save"}:
            role = None
    return result


def unmet_conditions(required: dict, observed: dict) -> list[str]:
    return [f"{key}: required={value}, observed={observed.get(key, 'unknown')}"
            for key, value in required.items() if observed.get(key) != value]
