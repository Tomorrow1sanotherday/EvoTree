import pytest

from evotree.observation_io import read_python_output


class Controller:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.commands = []

    def execute_python_command(self, command):
        self.commands.append(command)
        return next(self.replies)


def test_transient_none_is_retried_for_read_only_observation():
    controller = Controller([None, {"output": "  actual document info \n"}])
    assert read_python_output(controller, "get_env_info()", delay=0) == "actual document info"
    assert controller.commands == ["get_env_info()"] * 2


def test_transient_controller_exception_is_retried():
    class Flaky(Controller):
        def execute_python_command(self, command):
            self.commands.append(command)
            if len(self.commands) == 1:
                raise ConnectionError('controller restarting')
            return {'output': 'recovered'}
    controller = Flaky([])
    assert read_python_output(controller, 'get_env_info()', delay=0) == 'recovered'
    assert len(controller.commands) == 2


@pytest.mark.parametrize("reply", [None, {"output": None}, {"output": "", "status": "error"},
                                  {"output": "partial", "returncode": 1}])
def test_persistent_failure_is_explicit_not_empty_success(reply):
    controller = Controller([reply] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        read_python_output(controller, "get_env_info()", delay=0)
    assert len(controller.commands) == 3
