from types import SimpleNamespace

from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier, wait_for_save


class Clock:
    now = 0

    def read(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_modified_flag_alone_is_insufficient_for_async_save():
    clock = Clock()

    def observe():
        # Reproduces clean UNO document while disk still has old contents.
        return {"modified": False, "file_sha256": "old" if clock.now < 10 else "new"}

    result = wait_for_save({"modified": True, "file_sha256": "old"}, observe,
                          clock=clock.read, sleep=clock.sleep)
    assert result["outcome"] == "saved"
    assert result["seconds"] >= 12


def test_unsaved_or_unchanged_content_is_not_fabricated_as_saved():
    clock = Clock()
    result = wait_for_save({"modified": True, "file_sha256": "old"},
        lambda: {"modified": True, "file_sha256": "old"}, timeout=5,
        clock=clock.read, sleep=clock.sleep)
    assert result["outcome"] == "save_not_confirmed"


def test_clean_uno_flag_does_not_skip_pending_cursor_formatting_save():
    clock = Clock()
    result = wait_for_save({"modified": False, "file_sha256": "old"},
        lambda: {"modified": False, "file_sha256": "old" if clock.now < 10 else "new"},
        clock=clock.read, sleep=clock.sleep)
    assert result["outcome"] == "saved"
    assert result["seconds"] >= 12


def test_resave_with_identical_content_is_confirmed_by_file_write():
    clock = Clock()
    result = wait_for_save({"modified": False, "file_sha256": "same", "file_mtime_ns": 1},
        lambda: {"modified": False, "file_sha256": "same", "file_mtime_ns": 1 if clock.now < 5 else 2},
        clock=clock.read, sleep=clock.sleep)
    assert result["outcome"] == "saved"
    assert result["seconds"] >= 7


def test_clean_unchanged_document_is_not_a_save_acknowledgement():
    clock = Clock()
    result = wait_for_save({"modified": False, "file_sha256": "old"},
        lambda: {"modified": False, "file_sha256": "old"}, timeout=5,
        clock=clock.read, sleep=clock.sleep)
    assert result["outcome"] == "save_not_confirmed"


def test_barrier_waits_before_metric_and_restores_setup(monkeypatch):
    events = []
    postconfig = [{"type": "execute", "parameters": {"command": ["python", "-c", "import pyautogui; pyautogui.hotkey('ctrl','s')"]}}]

    def setup(config, proxy):
        events.append("postconfig")
        return True

    env = SimpleNamespace(evaluator={"result": {"type": "vm_file", "path": "/home/user/a.xlsx"},
                                     "postconfig": postconfig},
                          setup_controller=SimpleNamespace(setup=setup))

    def evaluate():
        env.setup_controller.setup(postconfig, False)
        events.append("metric")
        return 0.25

    def wait(before, observe):
        events.append("wait_for_disk")
        return {"outcome": "saved"}

    monkeypatch.setattr("experiments.m2_small_eval.office_save_barrier.wait_for_save", wait)
    value = evaluate_with_save_barrier(env, evaluate,
        observe=lambda *_: {"matched": True, "modified": True, "file_sha256": "old"})
    assert value == 0.25  # Original metric value is returned without revision.
    assert events == ["postconfig", "wait_for_disk", "metric"]
    assert env.setup_controller.setup is setup


def test_non_office_evaluator_is_unchanged():
    env = SimpleNamespace(evaluator={"result": {"type": "vm_command_line"}})
    assert evaluate_with_save_barrier(env, lambda: 0.0,
        observe=lambda *_: (_ for _ in ()).throw(AssertionError("Must not query UNO"))) == 0.0


def test_office_postconfig_without_save_is_unchanged():
    env = SimpleNamespace(evaluator={"result": {"type": "vm_file", "path": "/home/user/a.xlsx"},
        "postconfig": [{"type": "execute", "parameters": {"command": ["python", "-c", "pyautogui.hotkey('ctrl','c')"]}}]})
    assert evaluate_with_save_barrier(env, lambda: 0.0,
        observe=lambda *_: (_ for _ in ()).throw(AssertionError("Must not query UNO"))) == 0.0


def test_multi_metric_or_waits_once_for_shared_office_file(monkeypatch):
    """Official Impress OR metrics read the same file against two gold files."""
    events = []
    postconfig = [{"type": "execute", "parameters": {"command": ["python", "-c", "pyautogui.hotkey('ctrl','s')"]}}]
    result = {"type": "vm_file", "path": "/home/user/slide.pptx"}
    setup = lambda *_: events.append("postconfig")
    env = SimpleNamespace(evaluator={"result": [result.copy(), result.copy()],
                                     "postconfig": postconfig, "conj": "or"},
                          setup_controller=SimpleNamespace(setup=setup))

    def observe(env, path):
        events.append(("observe", path))
        return {"matched": True, "modified": True, "file_sha256": "original"}

    def wait(before, observe):
        events.append("wait")
        return {"outcome": "saved"}

    def evaluate():
        env.setup_controller.setup(postconfig, False)
        events.extend(["metric-one", "metric-two"])
        return 1.0

    monkeypatch.setattr("experiments.m2_small_eval.office_save_barrier.wait_for_save", wait)
    logs = []
    assert evaluate_with_save_barrier(env, evaluate, observe=observe, log=logs.append) == 1.0
    assert events == [("observe", result["path"]), "postconfig", "wait", "metric-one", "metric-two"]
    assert len(logs) == 1 and logs[0]["path"] == result["path"]
    assert env.setup_controller.setup is setup


def test_distinct_office_results_are_each_checked_before_scoring(monkeypatch):
    paths = ["/home/user/a.pptx", "/home/user/b.pptx"]
    postconfig = [{"type": "execute", "parameters": {"command": "pyautogui.hotkey('ctrl', 's')"}}]
    env = SimpleNamespace(evaluator={"result": [{"type": "vm_file", "path": p} for p in paths],
                                     "postconfig": postconfig},
                          setup_controller=SimpleNamespace(setup=lambda *_: None))
    observed = []
    def observe(env, path):
        observed.append(path)
        return {"matched": True, "modified": False, "file_sha256": path}
    def wait(before, observe):
        observe()
        return {"outcome": "saved"}
    def evaluate():
        env.setup_controller.setup(postconfig, False)
        assert observed == paths + paths
        return 0.0
    monkeypatch.setattr("experiments.m2_small_eval.office_save_barrier.wait_for_save", wait)
    assert evaluate_with_save_barrier(env, evaluate, observe=observe) == 0.0


def test_multi_metric_non_office_results_do_not_query_uno():
    env = SimpleNamespace(evaluator={"result": [{"type": "vm_command_line"}, {"type": "vm_file", "path": "/tmp/x.txt"}]})
    assert evaluate_with_save_barrier(env, lambda: 0.0,
        observe=lambda *_: (_ for _ in ()).throw(AssertionError("Must not query UNO"))) == 0.0
