"""OSWorldEpisodeAdapter 的测试。

测试重点不是"能否读 JSON"，而是确认语义边界：reset/terminate 不进入动作，
中间 reward=0 不被误判为失败，最终成败必须显式提供或来自客观结果。
"""

import json
from pathlib import Path

import pytest

from evotree.osworld_adapter import OSWorldEpisodeAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "rename_dir_gui_6steps.jsonl"


def records_from_fixture() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


class TestFromRecords:
    def test_skips_reset_and_terminal(self):
        episode = OSWorldEpisodeAdapter().from_records(records_from_fixture(), success=True)
        assert episode.success is True
        assert episode.num_steps == 4
        assert all(step.raw_action["action"] != "reset" for step in episode.steps)

    def test_intermediate_reward_zero_is_not_step_failure(self):
        episode = OSWorldEpisodeAdapter().from_records(records_from_fixture(), success=True)
        assert episode.success is True
        assert all(step.reward == 0 for step in episode.steps)

    def test_preserves_raw_action_and_screenshot_metadata(self):
        episode = OSWorldEpisodeAdapter().from_records(records_from_fixture(), success=True)
        first = episode.steps[0]
        assert "prediction" in first.raw_action
        assert first.screenshot_file == "step_1_20260618@151303.png"
        assert first.next_observation is not None

    def test_default_observation_is_explicitly_placeholder(self):
        episode = OSWorldEpisodeAdapter().from_records(records_from_fixture(), success=True)
        assert episode.steps[0].observation["app"] == "unknown"
        assert episode.steps[0].observation["window"].startswith("trajectory-step-")

    def test_custom_observation_factory_is_used(self):
        def factory(record, index):
            return {"app": "files", "window": "<file> — Files", "facts": {"index": index}}

        episode = OSWorldEpisodeAdapter().from_records(
            records_from_fixture(), success=True, observation_factory=factory
        )
        assert episode.steps[0].observation["app"] == "files"
        assert episode.steps[1].observation["facts"] == {"index": 1}

    def test_failure_episode_is_supported(self):
        episode = OSWorldEpisodeAdapter().from_records(records_from_fixture(), success=False)
        assert episode.success is False
        assert episode.num_steps == 4

    def test_unknown_success_requires_explicit_label(self):
        records = records_from_fixture()
        records[-1]["info"] = {"done": True}
        with pytest.raises(ValueError, match="success is unknown"):
            OSWorldEpisodeAdapter().from_records(records)

    def test_nested_tool_arguments_are_parsed(self):
        from evotree.distill_osworld import OSWorldDistiller

        prediction = (
            '<tool_call>{"name":"os.example","arguments":'
            '{"config":{"a":1,"b":[1,2]},"path":"a"}}</tool_call>'
        )
        action = OSWorldDistiller().distill_action({"prediction": prediction})
        assert action.name == "os.example"
        assert action.arg_slots == ("config", "path")

    def test_success_and_score_conflict_is_rejected(self):
        with pytest.raises(ValueError, match="disagree"):
            OSWorldEpisodeAdapter().from_records(
                records_from_fixture(), success=True, score=0.0
            )

    def test_client_error_is_not_a_recorded_action(self):
        records = records_from_fixture()[:1] + [
            {
                "step_num": 1,
                "prediction": "client error",
                "action": "not-an-action",
                "reward": 0,
                "done": False,
                "info": {},
            }
        ]
        episode = OSWorldEpisodeAdapter().from_records(records, success=False)
        assert episode.is_empty

    def test_invalid_json_is_reported(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text('{"step_num": 0}\nnot-json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            OSWorldEpisodeAdapter().from_jsonl(path, success=False)

    def test_result_directory_uses_result_txt(self, tmp_path):
        (tmp_path / "traj.jsonl").write_text(
            FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (tmp_path / "result.txt").write_text("1.0\n", encoding="utf-8")
        episode = OSWorldEpisodeAdapter().from_result_directory(tmp_path)
        assert episode.success is True
        assert episode.metadata["score"] == 1.0

    def test_missing_result_file_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="Missing OSWorld result"):
            OSWorldEpisodeAdapter().from_result_directory(tmp_path)

    def test_source_is_preserved(self):
        episode = OSWorldEpisodeAdapter().from_records(
            records_from_fixture(), success=True, source="fixture"
        )
        assert episode.source == "fixture"

    def test_score_zero_preserves_actions(self):
        episode = OSWorldEpisodeAdapter().from_records(records_from_fixture(), score=0.0)
        assert episode.success is False
        assert episode.num_steps == 4

    def test_factory_receives_original_record_and_compact_index(self):
        seen = []

        def factory(record, index):
            seen.append((record["step_num"], index))
            return {"app": "files", "window": "Files"}

        OSWorldEpisodeAdapter().from_records(
            records_from_fixture(), success=True, observation_factory=factory
        )
        assert seen == [(1, 0), (2, 1), (3, 2), (4, 3)]

    def test_raw_action_keeps_info_mapping(self):
        episode = OSWorldEpisodeAdapter().from_records(records_from_fixture(), success=True)
        assert episode.steps[0].raw_action["info"] == {}

    def test_from_jsonl_source_defaults_to_path(self):
        episode = OSWorldEpisodeAdapter().from_jsonl(FIXTURE, success=True)
        assert episode.source == str(FIXTURE)

    def test_missing_instruction_is_reported(self):
        records = [{"step_num": 0, "action": "reset"}]
        with pytest.raises(ValueError, match="no instruction"):
            OSWorldEpisodeAdapter().from_records(records, success=False)

    def test_empty_records_are_reported(self):
        with pytest.raises(ValueError, match="must not be empty"):
            OSWorldEpisodeAdapter().from_records([], success=False)

    def test_non_object_json_is_reported(self, tmp_path):
        path = tmp_path / "array.jsonl"
        path.write_text("[]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected object"):
            OSWorldEpisodeAdapter().from_jsonl(path, success=False)

    def test_invalid_result_score_is_reported(self, tmp_path):
        (tmp_path / "result.txt").write_text("not-a-score\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid score"):
            OSWorldEpisodeAdapter().from_result_directory(tmp_path)

    def test_nested_tool_arguments_are_parsed(self):
        from evotree.distill_osworld import OSWorldDistiller

        prediction = (
            '<tool_call>{"name":"os.example","arguments":'
            '{"config":{"a":1,"b":[1,2]},"path":"a"}}</tool_call>'
        )
        action = OSWorldDistiller().distill_action({"prediction": prediction})
        assert action.name == "os.example"
        assert action.arg_slots == ("config", "path")
