# EvoTree

Self-Evolving Hybrid Agents via Tree-Structured Experience Memory.

## Current status

The project contains the training-free memory core and the first integration boundary:

- `evotree.types`: typed task, state, action, node, and edge records;
- `evotree.scoring`: empirical edge value and UCB exploration score;
- `evotree.forest`: per-task experience forest with persistence and lookahead links;
- `evotree.distill`: replaceable placeholder distiller;
- `evotree.distill_osworld`: parser for OSWorld-MCP `prediction` records;
- `evotree.normalize`: deterministic window-title normalization;
- `evotree.notes`: task-independent caution notes;
- `evotree.retrieval`: recommendation, lookahead, caution, and tool re-ranking;
- `evotree.dual_retrieval`: independent task workflows and cross-task state advice;
- `evotree.memory`: top-level read/write API;
- `evotree.episode`: canonical episode-level data model;
- `evotree.osworld_adapter`: `traj.jsonl`/`result.txt` adapter;
- `evotree.integration`: framework-neutral online agent hooks.

Run the local test suite with:

```bash
python -m pytest tests/ -q
```

The core is framework-neutral. `wrap_osworld()` connects read/write hooks to the
OSWorld-MCP agent loop. Select `EvoTreeMemory(..., retrieval_mode="dual")` to inject
both complete historical task workflows and horizontal state advice; `"task"` and
`"state"` support ablations. The default `"legacy"` retains the original retrieval
configuration for historical comparisons. Workflows are persisted from successful
`write_episode()` calls; older banks need replay of their original seed episodes.

The frozen four-condition diagnostic is in
`experiments/m2_small_eval/dual_eval.py`. Its fixed per-route character budgets and
actual injection lengths are recorded; it does not claim equal token counts.

`retrieval_mode="dual_summary"` is an opt-in diagnostic prototype. It summarizes
pre-action observations and compares them without passing candidate actions or
outcomes to the comparison call. New histories retain bounded structured
observations; older banks without those observations cannot recover the missing
situation. The real-model probe did not reliably distinguish Calc source and target
roles from the current evidence, and each retrieval adds model calls. Do not treat
this mode as a validated general retrieval method or as evidence of improved task
success. The Calc role guard is disabled in this mode to keep the summary probe
separate from that specialized rule.

```python
from evotree.situation_summary import JsonChatTransport, ModelSituationBackend

backend = ModelSituationBackend(JsonChatTransport(api_url, model, api_key))
memory = EvoTreeMemory(retrieval_mode="dual_summary", situation_backend=backend)
```
