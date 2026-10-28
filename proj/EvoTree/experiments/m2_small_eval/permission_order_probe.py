"""Same saved observation, alternate advisory wording, first-decision audit.

No VM actions are executed. This diagnoses suggested action order only and
must never be reported as a task success experiment.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import time

from evotree.distill_osworld import parse_tool_calls


VARIANTS = ("original", "target_only", "parameters_only", "combined", "ordered")


def advice_variants(original, combined):
    target_lines = [line for line in combined.splitlines() if "target=" in line]
    assert target_lines, "Observed improved advice must contain a grounded GUI target"
    command = combined.split("；曾用：", 1)[1].split("\n", 1)[0]
    target = target_lines[0].split("target=", 1)[1].split("（", 1)[0]
    return {
        "original": original,
        "target_only": original + "\n" + "\n".join(target_lines),
        "parameters_only": "\n".join(line for line in combined.splitlines() if "target=" not in line),
        "combined": combined,
        "ordered": (
            "[经验] One successful prior execution; low confidence.\n"
            f"CURRENT ACTION: computer_use.click — {target}. "
            "Locate it in the CURRENT screenshot; old coordinates are not reusable.\n"
            "DEPENDENT NEXT ACTION: computer_use.type. "
            "Use this only AFTER completing the current action and observing its result.\n"
            f"Previously successful text: {command}\n"
            "This is advisory evidence. Choose actions from the current observation; "
            "do not skip the preparation or assume the dependent action has executed."
        ),
    }


def replace_advice(messages, old, new, image_bytes):
    result = copy.deepcopy(messages)
    occurrences = 0
    for message in result:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            text = item.get("text")
            if isinstance(text, str) and old in text:
                occurrences += 1
                item["text"] = text.replace(old, new, 1)
    assert occurrences == 1, "Expected exactly one current advisory section"
    assert not any(item.get("type") == "image_url" for message in result
                   for item in message.get("content", []) if isinstance(item, dict))
    result[-1]["content"].append({"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode(), "detail": "high"}})
    return result


def preparation_order(prediction):
    calls = parse_tool_calls(prediction)
    typed = next((i for i, call in enumerate(calls) if call["name"] == "computer_use"
                  and call["arguments"].get("action") == "type"), None)
    preparations = []
    for i, call in enumerate(calls):
        args = call["arguments"]
        if call["name"] == "computer_use" and args.get("action") in {"click", "double_click"}:
            preparations.append(i)
        if call["name"] == "computer_use" and args.get("action") == "key":
            keys = [str(k).lower() for k in args.get("keys", [])]
            if all(key in keys for key in ("ctrl", "alt", "t")):
                preparations.append(i)
    first_terminal = next((i for i, call in enumerate(calls) if call["name"] == "computer_use"
                           and call["arguments"].get("action") == "terminate"), None)
    return dict(calls=calls, typing_without_preparation=typed is not None and not any(i < typed for i in preparations),
                preparation_first=bool(preparations) and (typed is None or preparations[0] < typed),
                dependent_action_deferred=bool(preparations) and typed is None and first_terminal is None,
                first_action=calls[0] if calls else None,
                note="Click order is a protocol proxy; target correctness and task success require VM verification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    import os
    from filelock import FileLock
    from openai import OpenAI
    from experiments.m2_small_eval.diagnostic_pilot import write_json, append
    args.output_root.mkdir(parents=True, exist_ok=False)
    client = OpenAI(base_url=args.api_url, api_key=os.environ["OSWORLD_API_KEY"], timeout=90, max_retries=1)
    results = []
    conditions = None
    expected = 3 * args.repeats * len(VARIANTS)
    write_json(args.output_root / "plan.json", dict(setting="saved observation decision probe; no actions executed",
        variants=VARIANTS, screenshots=3, repeats=args.repeats, expected_requests=expected,
        temperature=1.0, top_p=0.9, source_root=str(args.source_root)))
    for repetition in range(args.repeats):
        for screenshot in range(1, 4):
            directory = args.source_root / f"task-0/rep-{screenshot}/improved"
            rows = [json.loads(line) for line in (directory / "traj.jsonl").read_text().splitlines() if line.strip()]
            first = next(row for row in rows if "messages" in row)
            original = json.loads((args.source_root / f"task-0/rep-{screenshot}/current/decisions.jsonl").read_text().splitlines()[0])["text"]
            combined = json.loads((directory / "decisions.jsonl").read_text().splitlines()[0])["text"]
            variants = advice_variants(original, combined)
            if conditions is None:
                conditions = variants
                write_json(args.output_root / "advice.json", variants)
            image_bytes = (directory / "reset.png").read_bytes()
            # Rotate variant order across observations and repetitions to avoid
            # always requesting the same wording first from the API service.
            offset = (screenshot + repetition - 1) % len(VARIANTS)
            order = VARIANTS[offset:] + VARIANTS[:offset]
            for variant in order:
                messages = replace_advice(first["messages"], combined, variants[variant], image_bytes)
                write_json(args.output_root / "status.json", dict(state="running", completed=len(results),
                    planned=expected, screenshot=screenshot, repetition=repetition+1, variant=variant))
                started = time.monotonic()
                with FileLock(os.environ["OSWORLD_MODEL_REQUEST_LOCK"], timeout=600):
                    reply = client.chat.completions.create(model=args.model, messages=messages,
                        temperature=1.0, top_p=0.9, max_tokens=2048, reasoning_effort="none")
                prediction = reply.choices[0].message.content or ""
                row = dict(variant=variant, screenshot=screenshot, repetition=repetition+1,
                    screenshot_sha256=hashlib.sha256(image_bytes).hexdigest(), prediction=prediction,
                    seconds=time.monotonic()-started, advice_chars=len(variants[variant]),
                    usage=reply.usage.model_dump() if reply.usage else None,
                    **preparation_order(prediction))
                results.append(row)
                append(args.output_root / "responses.jsonl", row)
                summary = {name: dict(completed=sum(r["variant"] == name for r in results),
                    preparation_first=sum(r["variant"] == name and r["preparation_first"] for r in results),
                    typing_without_preparation=sum(r["variant"] == name and r["typing_without_preparation"] for r in results),
                    dependent_action_deferred=sum(r["variant"] == name and r["dependent_action_deferred"] for r in results))
                    for name in VARIANTS}
                write_json(args.output_root / "summary.json", dict(setting="decision order only", variants=summary))
    write_json(args.output_root / "status.json", dict(state="complete", completed=len(results), planned=expected))


if __name__ == "__main__":
    main()
