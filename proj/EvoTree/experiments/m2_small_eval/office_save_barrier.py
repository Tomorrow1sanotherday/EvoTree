"""Wait for the evaluator's existing Office save to reach the target file.

No save is added, no document is edited, and no expected answer is inspected.
The original metric runs once after the existing postconfig has completed.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
import time
from urllib.request import Request, urlopen


def wait_for_save(before, observe, *, timeout=45, stable_seconds=2, clock=time.monotonic, sleep=time.sleep):
    """Confirm the requested save produced a stable disk file, even if UNO reports clean.

    UNO text-cursor formatting can leave isModified() false while the changed
    content is only in memory. Clean status alone cannot acknowledge Ctrl+S.
    """
    started = clock()
    previous = None
    stable_since = None
    observations = []
    while clock() - started < timeout:
        current = observe()
        observations.append(current)
        fingerprint = current.get("file_sha256")
        changed = (fingerprint != before.get("file_sha256")
                   or (current.get("file_mtime_ns") is not None
                       and before.get("file_mtime_ns") is not None
                       and current["file_mtime_ns"] != before["file_mtime_ns"]))
        ready = current.get("modified") is False and fingerprint and changed
        signature = (fingerprint, current.get("file_mtime_ns"))
        if ready and signature == previous and stable_since is not None:
            if clock() - stable_since >= stable_seconds:
                return dict(outcome="saved", seconds=clock() - started, observations=observations)
        else:
            stable_since = clock() if ready else None
        previous = signature
        sleep(1)
    # Preserve a genuine failure as a failure. Waiting never invokes a rescue
    # save, dismisses a dialog, or retries until the metric happens to pass.
    return dict(outcome="save_not_confirmed", seconds=clock() - started, observations=observations)


def _office_paths(env):
    result = env.evaluator.get("result", {})
    results = result if isinstance(result, list) else [result]
    paths = []
    for item in results:
        if not isinstance(item, dict) or item.get("type") != "vm_file":
            continue
        path = item.get("path")
        if (isinstance(path, str)
                and PurePosixPath(path).suffix.lower() in {".xlsx", ".ods", ".pptx", ".odp", ".docx", ".odt"}
                and path not in paths):
            paths.append(path)
    if not paths:
        return []
    # Apply only where the evaluator itself already requests keyboard save.
    commands = []
    for step in env.evaluator.get("postconfig", []):
        if step.get("type") != "execute":
            continue
        command = step.get("parameters", {}).get("command", [])
        commands.extend([command] if isinstance(command, str) else command)
    save = re.compile(r"pyautogui\.hotkey\(\s*['\"]ctrl['\"]\s*,\s*['\"]s['\"]\s*\)", re.I)
    if not any(isinstance(text, str) and save.search(text) for text in commands):
        return []
    return paths


def observe_document(env, path):
    script = '''import uno,json,hashlib,os
from urllib.parse import unquote,urlparse
local=uno.getComponentContext()
resolver=local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',local)
ctx=resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')
desktop=ctx.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop',ctx)
components=desktop.getComponents().createEnumeration()
target=TARGET_PATH
result={'matched':False}
while components.hasMoreElements():
 doc=components.nextElement()
 if hasattr(doc,'getURL') and unquote(urlparse(doc.getURL()).path)==target:
  result={'matched':True,'url':doc.getURL(),'modified':doc.isModified()}
  break
if os.path.isfile(target):
 with open(target,'rb') as stream: result['file_sha256']=hashlib.sha256(stream.read()).hexdigest()
 result['file_size']=os.path.getsize(target)
 result['file_mtime_ns']=os.stat(target).st_mtime_ns
print(json.dumps(result))
'''.replace("TARGET_PATH", repr(path))
    body = json.dumps({"command": ["/usr/bin/python3", "-c", script], "shell": False}).encode()
    request = Request(env.controller.http_server + "/execute", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        value = json.load(response)
    if value.get("returncode") != 0:
        raise RuntimeError("Read-only Office save observation failed")
    return json.loads(value["output"].strip())


def evaluate_with_save_barrier(env, evaluate, *, log=None, observe=observe_document):
    """Wrap the original postconfig so waiting precedes metric file retrieval."""
    paths = _office_paths(env)
    if not paths:
        return evaluate()
    documents = []
    for path in paths:
        before = observe(env, path)
        if before.get("matched"):
            documents.append((path, before))
        elif log:
            log(dict(path=path, outcome="no_matching_office_document", before=before))
    if not documents:
        return evaluate()
    original_setup = env.setup_controller.setup
    postconfig = env.evaluator.get("postconfig", [])

    def setup_then_wait(config, *args, **kwargs):
        value = original_setup(config, *args, **kwargs)
        if config is postconfig:
            for path, before in documents:
                result = wait_for_save(before, lambda: observe(env, path))
                if log:
                    log(dict(path=path, before=before, **result))
        return value

    env.setup_controller.setup = setup_then_wait
    try:
        return evaluate()
    finally:
        env.setup_controller.setup = original_setup
