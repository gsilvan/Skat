# Working on the consolidated Skat engine

The stable core implements official digital Skat (ISkO November 2022). The
`vibe-coding` branch preserves experiments and historical models/corpora.

## Commit messages

- Use Conventional Commits: `type(scope): short description` (scope optional).
- Be precise and to the point; describe the concrete change.
- Keep messages concise. No long narratives or unnecessary detail.

## README agent ratings

- Keep a measured agent-strength table in README.md. List card-policy checkpoints
  and bidding strategies separately, including the conservative Random baseline.
- The fixed protocol is 1,000 dealt rounds, seed 42, candidate at seat 0, two
  Random agents with conservative bidding at seats 1 and 2, and rotating dealer.
  Include passes in the 1,000 rounds; report party wins, solo/defence results,
  passes and official points per dealt round. Never mix this with seat-rotated
  evaluation panels or matches against different opponents.
- Refresh ratings when adding a model/agent or changing strategy/bidding. Use
  `python -m scripts.rate_agents --out NEW_DIRECTORY --device cpu`; preserve the
  checkpoint hashes and results in docs/agent-ratings.json. Add new variants to
  the script's PROFILES while retaining the fixed Random baseline. Long rating runs are
  standalone benchmarks, not test-suite training.

## Commands

- `uv sync`: engine and development tools; Torch is optional.
- `uv sync --extra cpu`, `--extra cuda` or `--extra rocm`: choose one Torch backend for policy training/inference.
- `uv sync --extra sgf`: additionally install raw ISS SGF parsing support.
- `uv run pytest`: core tests; AI tests skip when Torch is absent.
- `uv run --extra cpu pytest`: all tests.
- `uv run ruff check --select F,E9,I .`: errors, unused imports and import order.
- `uv run ruff format --check .`: formatting.
- `uv run python -m skat play --rounds 33`
- `play --policy PATH` shares one checkpoint; `--policy0/1/2 PATH` overrides it
  per seat. Models are egocentric and can play either party from every seat.
- `uv run python -m skat collect --rounds 1000 --out data/run-1`: all available CPU workers.
- `play`, `collect` and `evaluate` use all available CPUs by default; add
  `--workers 1` for serial execution or `--workers N` to limit CPU use.
- `uv run --extra cpu python -m skat train --data data/run-1 --out trained_models/policy.pt`
- `uv run --extra cpu python -m skat evaluate --policy trained_models/policy.pt --out benchmark_runs/run-1.json`
- `evaluate --reference-policy PATH` fixes the model on the two reference seats
  while rotating the candidate. `--opponent-policy` remains a compatible alias.

Tests must stay short: only tiny training smoke checks are allowed. Real corpus
collection, long training and playing-strength benchmarks run separately.
`configs/teacher-2000.toml` is the opt-in 2,000-simulation / 100-epoch profile;
baseline defaults stay unchanged. `configs/iss.toml` selects 20 epochs for ISS
demonstrations. `docs/iss-training.md` documents the streaming import and
`scripts/run_iss_experiment.py` runs all stages with logs and durable status.
Training writes an atomic `.training.json`
after each epoch; cross-entropy and KL metrics use the same source targets.

## Boundaries

- `rules.py` owns declarations and official scoring. Engine and search call the
  same evaluator. Final overbidding depends on earned grades, not the initial value.
- Agents receive detached `PlayerView` snapshots, never `Round` or `Player`.
  Engine actions validate ownership, turn order and follow-suit before mutation.
- Card selection and pressing return decisions without removing cards.
- MCTS samples uniformly from feasible full worlds; never discard known voids.
- Bidding is separate from card play: conservative by default, optional ISS neural
  hand filter for any automatic agent. Both retain safe ceilings and covering
  declarations. Bidding features are the original ten cards plus relative position.
  Historical probabilities are not guaranteed calibrated for every card strategy.
  `docs/bidding-training.md` documents calibration and separate risk benchmarks.
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
  Bidding checkpoints have their own schema and are never card-policy checkpoints.
  Bidding comparisons optimize points per dealt round, including passes; bootstrap
  complete deals and freeze thresholds before independent test deals.
- One transformer and one win/loss reward. MCTS uses visit targets; the separate
  `iss-action-v1` corpus uses recorded legal cards, never fabricated visit counts. Archive experiments in
  separate branches rather than expanding the common CLI.
- Randomness belongs to individual agents/deals. Stable derived seeds use SHA-256.

Physical-table errors, referee procedures and informal variants are outside this
engine. Invalid digital actions are rejected without changing the round.
