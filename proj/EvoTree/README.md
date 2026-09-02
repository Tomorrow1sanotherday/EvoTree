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
- `evotree.memory`: top-level read/write API;
- `evotree.episode`: canonical episode-level data model;
- `evotree.osworld_adapter`: `traj.jsonl`/`result.txt` adapter;
- `evotree.integration`: framework-neutral online agent hooks.

Run the local test suite with:

```bash
python -m pytest tests/ -q
```

The current implementation is deliberately framework-neutral. It does not yet modify
OSWorld-MCP's runner; the next integration step must connect the runtime hooks to the
actual agent loop and provide real state observations.
