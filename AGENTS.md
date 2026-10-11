# Working on the consolidated Skat engine

The stable core implements official digital Skat (ISkO November 2022). The
`vibe-coding` branch preserves experiments and historical models/corpora.

## Commands

- `uv sync`: engine and development tools; Torch is optional.
- `uv sync --extra ai`: additionally install policy training/inference dependencies.
- `uv sync --extra sgf`: additionally install raw ISS SGF parsing support.
- `uv run pytest`: core tests; AI tests skip when Torch is absent.
- `uv run --extra ai pytest`: all tests.
- `uv run ruff check --select F,E9,I .`: errors, unused imports and import order.
- `uv run ruff format --check .`: formatting.
- `uv run python -m skat play --rounds 33`
- `play --policy PATH` shares one checkpoint; `--policy0/1/2 PATH` overrides it
  per seat. Models are egocentric and can play either party from every seat.
- `uv run python -m skat collect --rounds 1000 --out data/run-1`: all available CPU workers.
- `play`, `collect` and `evaluate` use all available CPUs by default; add
  `--workers 1` for serial execution or `--workers N` to limit CPU use.
- `uv run --extra ai python -m skat train --data data/run-1 --out trained_models/policy.pt`
- `uv run --extra ai python -m skat evaluate --policy trained_models/policy.pt --out benchmark_runs/run-1.json`
- `evaluate --reference-policy PATH` fixes the model on the two reference seats
  while rotating the candidate. `--opponent-policy` remains a compatible alias.

Tests must stay short: only tiny training smoke checks are allowed. Real corpus
collection, long training and playing-strength benchmarks run separately.
`configs/teacher-2000.toml` is the opt-in 2,000-simulation / 100-epoch profile;
baseline defaults stay unchanged. Training writes an atomic `.training.json`
after each epoch; cross-entropy and KL metrics use the same visit targets.

## Boundaries

- `rules.py` owns declarations and official scoring. Engine and search call the
  same evaluator. Final overbidding depends on earned grades, not the initial value.
- Agents receive detached `PlayerView` snapshots, never `Round` or `Player`.
  Engine actions validate ownership, turn order and follow-suit before mutation.
- Card selection and pressing return decisions without removing cards.
- MCTS samples uniformly from feasible full worlds; never discard known voids.
- All seats use the same conservative bidding/declaration heuristics.
- Policy observation is egocentric and includes contract, exposed cards, known
  skat and trick counts. Changing its meaning requires a schema bump and new data.
- Distillation imports arrays, Card and model metadata, never Agent/Player/Round.
- Collection workers own separate games and recorders; seeds derive from global
  round indices and seats, so worker count does not change searches.
- Play/evaluate share one persistent search pool across MCTS seats. Independent
  world RNG streams and ordered aggregation keep tallies identical across worker
  counts. Collection uses serial searches inside its parallel game workers.
- Python callers can pass a context-managed `SearchPool` to `MCTSAgent`;
  without one, the agent runs serially and owns no worker processes.
- Groups remain round groups across shard boundaries; never split samples by row.
- Checkpoints/shards require exact rules, observation and objective versions.
- One transformer, one visit target, one win/loss reward. Archive experiments in
  separate branches rather than expanding the common CLI.
- Randomness belongs to individual agents/deals. Stable derived seeds use SHA-256.

Physical-table errors, referee procedures and informal variants are outside this
engine. Invalid digital actions are rejected without changing the round.
