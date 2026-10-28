"""OSWorld-MCP traj.jsonl 适配器。

职责：把 OSWorld 的原始轨迹转换成 EvoTree 的统一 Episode。

OSWorld 原始记录中：
    - step 0 通常是 reset，不是 agent 动作；
    - 中间 reward 常为 0，不能当作单步失败；
    - prediction 是 agent 的原始 function-calling 输出；
    - 最后一条可能是 DONE/terminate，不是普通动作；
    - 真实任务成败需要从 result.txt（或调用方提供的 score）确定。

适配器只负责格式与语义边界，不负责猜测任务成败。score 必须由调用方
明确传入，避免把"没有 score"错误地当成成功或失败。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from .distill_osworld import OSWorldDistiller, extract_error_note, parse_tool_calls
from .episode import Episode, EpisodeStep

ObservationFactory = Callable[[Mapping[str, Any], int], Mapping[str, Any]]


class OSWorldEpisodeAdapter:
    """把一个 OSWorld ``traj.jsonl`` 转成 EvoTree ``Episode``。

    ``OSWorldDistiller`` 负责 action 的解析与 terminal 判断；本类负责：
        1. 跳过 reset；
        2. 跳过 DONE/terminate；
        3. 保留原始 prediction/action/reward/info；
        4. 用 episode-level score 标记整条轨迹的最终结果。
    """

    def __init__(self, distiller: Optional[OSWorldDistiller] = None) -> None:
        self.distiller = distiller or OSWorldDistiller()

    def from_jsonl(
        self,
        path: str | Path,
        *,
        success: Optional[bool] = None,
        score: Optional[float] = None,
        source: Optional[str] = None,
        observation_factory: Optional[ObservationFactory] = None,
    ) -> Episode:
        """读取一个 ``traj.jsonl`` 并构造 Episode。

        Args:
            path: traj.jsonl 文件路径。
            success: 整轮最终是否成功。建议直接传入；若不传则尝试从最后
                     记录的 ``info`` 推断，但不会把中间 reward=0 当失败。
            score: 可选的最终分数。若给出，``score > 0`` 视为成功。
                   ``success`` 与 score 同时给出且冲突时抛 ValueError。
            source: 轨迹来源标识。
            observation_factory: 可选的 ``record, index -> dict`` 函数，用于
                                 从真实记录生成动作前 observation。没有时使用
                                 最小 metadata observation（不声称是真实 φ）。

        Raises:
            ValueError: instruction 缺失、成败标签缺失/冲突、JSON 无效等。
        """
        records = self._read_records(path)
        return self.from_records(
            records,
            success=success,
            score=score,
            source=source or str(path),
            observation_factory=observation_factory,
        )

    def from_result_directory(
        self,
        directory: str | Path,
        *,
        observation_factory: Optional[ObservationFactory] = None,
    ) -> Episode:
        """从 OSWorld 结果目录读取 ``traj.jsonl`` 与 ``result.txt``。

        ``result.txt`` 是 OSWorld 客观 evaluator 的最终结果；使用它比从
        traj 中的中间 reward 推断更可靠。
        """
        directory = Path(directory)
        result_path = directory / "result.txt"
        try:
            score = float(result_path.read_text(encoding="utf-8").strip())
        except FileNotFoundError as exc:
            raise ValueError(f"Missing OSWorld result file: {result_path}") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid score in {result_path}") from exc
        return self.from_jsonl(
            directory / "traj.jsonl",
            score=score,
            source=str(directory),
            observation_factory=observation_factory,
        )

    def from_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        success: Optional[bool] = None,
        score: Optional[float] = None,
        source: Optional[str] = None,
        observation_factory: Optional[ObservationFactory] = None,
    ) -> Episode:
        """从已加载的记录构造 Episode，便于在线接入和单元测试。"""
        normalized = [dict(record) for record in records]
        if not normalized:
            raise ValueError("records must not be empty")
        # Old runner logs repeat the entire reply on every executed action.
        # Align by turn before dropping terminal records, so GUI replay cannot
        # silently attribute every action to the first call in the reply.
        groups = {}
        for record in normalized:
            prediction = record.get("prediction")
            if isinstance(prediction, str) and prediction != "client error":
                groups.setdefault((record.get("step_num"), prediction), []).append(record)
        for (_, prediction), rows in groups.items():
            calls = parse_tool_calls(prediction)
            if len(calls) > 1 and len(calls) != len(rows):
                raise ValueError("Incomplete multi-action turn cannot be replayed into memory")
            if len(calls) == len(rows):
                for call, record in zip(calls, rows):
                    # The runner writes an explicit null when it has no
                    # per-action annotation. Treat that as missing, too.
                    if record.get("tool_call") is None:
                        record["tool_call"] = call
        instruction = self._find_instruction(normalized)
        episode_success = self._resolve_success(normalized, success=success, score=score)
        steps = self._build_steps(normalized, observation_factory)
        return Episode(
            instruction=instruction,
            steps=tuple(steps),
            success=episode_success,
            source=source,
            metadata={"score": score, "num_raw_records": len(normalized), "adapter": "osworld"},
        )

    @staticmethod
    def _read_records(path: str | Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        path = Path(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
        return records

    @staticmethod
    def _find_instruction(records: list[dict[str, Any]]) -> str:
        for record in records:
            instruction = record.get("instruction")
            if isinstance(instruction, str) and instruction.strip():
                return instruction.strip()
        raise ValueError("OSWorld trajectory has no instruction")

    @staticmethod
    def _resolve_success(
        records: list[dict[str, Any]],
        *,
        success: Optional[bool],
        score: Optional[float],
    ) -> bool:
        if score is not None:
            score_success = score > 0
            if success is not None and success != score_success:
                raise ValueError("success and score disagree")
            return score_success
        if success is not None:
            return success

        # 只使用明确的 episode-level 信息；不读取中间 reward=0 作为失败标签。
        for record in reversed(records):
            info = record.get("info")
            if isinstance(info, dict):
                value = info.get("success")
                if isinstance(value, bool):
                    return value
                value = info.get("status")
                if isinstance(value, str) and value.lower() in {"success", "failure"}:
                    return value.lower() == "success"
        raise ValueError("Episode success is unknown; pass success= or score=")

    def _build_steps(
        self,
        records: list[dict[str, Any]],
        observation_factory: Optional[ObservationFactory],
    ) -> list[EpisodeStep]:
        action_records = []
        for record in records:
            raw_action = record.get("action")
            if isinstance(raw_action, str) and raw_action.strip().lower() == "reset":
                continue
            if self.distiller.is_terminal(record):
                continue
            # "client error" 是模型调用/解析失败，不是 agent 在环境中执行的动作。
            if self._is_client_error(record):
                continue
            action_records.append(record)

        steps: list[EpisodeStep] = []
        for index, record in enumerate(action_records):
            if observation_factory is not None:
                observation = dict(observation_factory(record, index))
            else:
                observation = self._fallback_observation(record, index)
            note = self._extract_note(record)
            steps.append(
                EpisodeStep(
                    step_index=int(record.get("step_num", index)),
                    observation=observation,
                    raw_action=self._raw_action(record),
                    note=note,
                    action_success=note is None,
                    reward=float(record.get("reward", 0.0) or 0.0),
                    done=bool(record.get("done", False)),
                    screenshot_file=record.get("screenshot_file"),
                    metadata={
                        "prediction": record.get("prediction"),
                        "info": record.get("info", {}),
                        "action_timestamp": record.get("action_timestamp"),
                    },
                )
            )

        # 下一条动作的动作前观测就是当前动作的默认落点观测。
        # 最后一步若没有可用的后继观测，保持 None。
        for index in range(len(steps) - 1):
            steps[index] = replace(
                steps[index], next_observation=steps[index + 1].observation
            )
        return steps

    @staticmethod
    def _is_client_error(record: Mapping[str, Any]) -> bool:
        """判断记录是否表示模型调用失败，而不是环境中的真实动作。"""
        prediction = record.get("prediction")
        return isinstance(prediction, str) and prediction.strip().lower() == "client error"

    @staticmethod
    def _raw_action(record: Mapping[str, Any]) -> dict[str, Any]:
        """保留 prediction 与原始 action，给真实 distiller 自己解析。"""
        return {
            "prediction": record.get("prediction"),
            "action": record.get("action"),
            "info": record.get("info", {}),
            "tool_call": record.get("tool_call"),
            "gui_semantics": record.get("gui_semantics"),
        }

    @staticmethod
    def _extract_note(record: Mapping[str, Any]) -> Optional[str]:
        """提取明确的执行错误，不把 reward=0 当成错误。

        与在线 wrapper 共用 ``extract_error_note``，保证同一条轨迹无论走离线
        重放还是在线 hook，都得到同一个边级成败标签。

        （client error 记录在 ``_build_steps`` 里已被整条剔除——那是模型调用
        失败，不是 agent 在环境里执行的动作，不该进经验路径。）
        """
        return extract_error_note(
            record.get("info"), record.get("exe_result"),
            tool_name=(record.get("tool_call") or {}).get("name", ""),
        )

    @staticmethod
    def _fallback_observation(record: dict[str, Any], index: int) -> dict[str, Any]:
        """没有 screenshot/a11y 提取器时的最小 observation。

        这里不伪造真实 app/window：把轨迹元数据作为占位，方便格式测试；
        接入真实 agent 时必须由 observation_factory 提供真实 φ 输入。
        """
        return {
            "app": "unknown",
            "window": f"trajectory-step-{record.get('step_num', index)}",
            "dialog": None,
            "facts": {},
        }


def load_osworld_episode(
    path: str | Path,
    *,
    success: Optional[bool] = None,
    score: Optional[float] = None,
) -> Episode:
    """便捷函数：使用默认 OSWorldEpisodeAdapter 读取单条轨迹。"""
    return OSWorldEpisodeAdapter().from_jsonl(path, success=success, score=score)
