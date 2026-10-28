"""Optional visual grounding of an executed GUI action, with audit provenance.

This annotates an operation's target, never its task success. Coordinates stay
in the source trajectory; a fresh screenshot must be used to locate the target
when a remembered action is suggested again.
"""
from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any


def action_digest(action: Any) -> str:
    return hashlib.sha256(json.dumps(action, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def grounded_target(raw: Mapping[str, Any]) -> str | None:
    annotation = raw.get("gui_semantics")
    if not isinstance(annotation, Mapping):
        return None
    if annotation.get("source") != "visual_transition" or annotation.get("supported") is not True:
        return None
    if annotation.get("action_sha256") != action_digest(raw.get("action")):
        return None
    if not all(isinstance(annotation.get(k), str) and annotation[k].strip()
               for k in ("target", "evidence", "before_sha256", "after_sha256")):
        return None
    return annotation["target"].strip()


class VisionGUIAnnotator:
    """Use a supplied vision request function; importing this makes no calls.

    The callback accepts chat messages and returns a JSON string. The caller
    owns model choice, credentials, request limits, and annotation cost logs.
    Unsupported or visually ambiguous actions retain their original evidence
    without inventing a target.
    """

    def __init__(self, request: Callable[[list[dict]], str]) -> None:
        self.request = request

    def __call__(self, raw: Mapping[str, Any], before: Mapping[str, Any],
                 after: Mapping[str, Any]) -> dict | None:
        call = raw.get("tool_call") or {}
        arguments = call.get("arguments") or {}
        if call.get("name") != "computer_use" or arguments.get("action") not in {
            "click", "double_click", "right_click", "middle_click", "drag", "move", "scroll",
        }:
            return None
        first, second = before.get("screenshot"), after.get("screenshot")
        if not isinstance(first, bytes) or not isinstance(second, bytes):
            return None
        instruction = (
            "Identify the semantic target of this one executed GUI action from the BEFORE "
            "and AFTER screenshots and actual executed code. Return only JSON with "
            "supported (boolean), target (short string), evidence (short string). "
            "For example, describe focusing a terminal input area or selecting a column header. "
            "Use supported=false and empty target if ambiguous. Describe the attempted operation, "
            "not the whole task. Do not infer task completion, invent controls, or retain numeric "
            "screen coordinates in target. Evidence must describe visible screenshot facts. "
            "The actual code, not a model's claimed conclusion, determines what executed.\n"
            + json.dumps({"call": call, "executed": raw.get("action")}, ensure_ascii=False)
        )
        content = [{"type": "text", "text": instruction}]
        for label, data in (("BEFORE", first), ("AFTER", second)):
            content.extend([
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(data).decode(),
                }},
            ])
        text = self.request([{"role": "user", "content": content}]).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get("supported"), bool):
            raise ValueError("Visual annotation must supply a boolean supported field")
        for name in ("target", "evidence"):
            if not isinstance(value.get(name), str):
                raise ValueError(f"Visual annotation must supply {name}")
        if value["supported"] and not (value["target"].strip() and value["evidence"].strip()):
            raise ValueError("Supported visual annotation requires target and evidence")
        return {
            "source": "visual_transition", "supported": value["supported"],
            "target": value["target"].strip(), "evidence": value["evidence"].strip(),
            "action_sha256": action_digest(raw.get("action")),
            "before_sha256": hashlib.sha256(first).hexdigest(),
            "after_sha256": hashlib.sha256(second).hexdigest(),
        }
