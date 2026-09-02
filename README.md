Skat: The traditional card-game with automated agents!
=====

## Installation

###### Supported Python version
* Python 3.12+

###### Required packages

* numpy
* pytorch
* tensorboard
* tqdm

As this is a work in progress, install the latest development version by cloning this 
repository and install it locally:

    git clone https://github.com/gsilvan/Skat.git
    cd Skat
    uv sync

## Local web GUI

The Flask/React interface supports a human against two selectable agents and a
spectator table with three automated agents. Bidding, skat pickup, pressing,
declaration and card play all stay inside the existing Python game engine.

```bash
# One-time frontend setup and production build.
cd web
npm install
npm run build
cd ..

# Flask serves the built UI locally at http://127.0.0.1:5000.
uv run python -m skat.web
```

For frontend development, run `uv run python -m skat.web` and `npm run dev`
from `web/` in separate terminals, then open `http://127.0.0.1:5173`.

The UI includes a self-rendered fallback deck. The optional Skatstube sprite
requested for local use can be downloaded with `./web/scripts/fetch-cards.sh`.
Its redistribution rights are unclear, so the PNG is deliberately ignored by
Git and must not be committed. The card renderer is isolated so it can later be
replaced by an original or clearly licensed deck.

## Reproducible agent benchmark

The benchmark trains every supported trainable agent family, evaluates all
playable families on the same deals and writes a resumable, self-describing run:

```bash
# Resolve variants and costs without doing work.
uv run python -m skat.benchmark --profile standard --plan

# Fast end-to-end integration run; its scores are not statistically meaningful.
uv run python -m skat.benchmark --profile smoke

# Meaningful local comparison, or the expensive publication-sized profile.
uv run python -m skat.benchmark --profile standard --seed 20260829 --device cpu
uv run python -m skat.benchmark --profile full --seed 20260829 --device cpu

# Continue after an interruption; completed stages are not repeated.
uv run python -m skat.benchmark --resume benchmark_runs/<run>

# Paired progress comparison on identical deal seeds (with exact McNemar p).
uv run python -m skat.benchmark --compare benchmark_runs/<old> benchmark_runs/<new>
```

Primary ranking entries are `random`, `mcts`, `dqn-mlp`, `dqn-transformer`,
`policy-mlp`, `policy-transformer`, and `belief-mcts`, plus all existing policy
controls for the value target and raw transformer features. Use `--no-controls`
for a shorter primary-model-only run. The interactive CLI agent, recorder wrappers
and unfinished `DeepAgent` are recorded as exclusions in the manifest.

Every candidate sees the same deals, is rotated through all three seats, and is
tested against two fixed populations: seeded random opponents and the trained
transformer policy. The second panel is necessary because the learned belief is
population-specific and is known not to improve play against random opponents.
The ranking uses the designated candidate seat's party win rate across both
panels and reports its 95% Wilson interval, solo/defence rates and card points.
Every card-playing agent is evaluated with its conservative bidding baseline and
with aggressive, balanced and defensive variants of one shared neural bidder.
The modes use 35%, 50% and 65% predicted-win thresholds by default. Thus the
same deals isolate both card-play strength and the gain (or loss) from accepting
more bidding risk. `results.json` retains every per-deal outcome and deal seed,
allowing paired significance tests between runs rather than comparing only totals.

Each run contains:

- `manifest.json`: exact config, Git revision/dirty state, environment, derived
  seeds, stage status, durations, exclusions, artifact paths and SHA-256 hashes;
- `events.jsonl`: append-only progress, including failures and resumed skips;
- `checkpoints/`, `data/`, and `logs/`: models, fixed corpora and TensorBoard
  curves;
- `results.json`, `ranking.csv`, and `report.md`: machine- and human-readable
  results;
- `history.jsonl` in the parent directory: one compact record per completed run.

A seed controls the deck, each agent's private Python generator, replay sampling,
Torch/NumPy state, model initialisation, train/validation split and batch order.
Architecture and target variants within one family deliberately share the same
training seed, deals and held-out groups so the named variant is the variable
being compared.
Deterministic Torch kernels are enabled by default. Repetition is guaranteed on
the same software and device; CPU, CUDA and ROCm results are not promised to be
bit-identical, so the backend is part of the manifest. CPU is the benchmark
default for that reason.

## Neural bidding

`BiddingNet` learns whether a ten-card hand is winnable from 35 inputs: one bit
for each of the 32 cards and a one-hot front/middle/rear-hand position. The MCTS
collector plays every seeded deal three times, forcing each seat to be soloist.
This counterfactual collection is important: recording only normal auction
winners would hide most weak hands and teach the network almost nothing about
when to pass. A win is stored as `+1`, a loss as `-1`; binary cross entropy
trains the corresponding calibrated win probability. All three samples from a
deal stay together in either training or validation.

```bash
# Standalone collection and training.
uv run python -m skat.agents.neural_bidding.collect \
  --out data/bidding --deals 2000 --simulations 600 \
  --determinizations 20 --seed 20260829
uv run python -m skat.agents.neural_bidding.train \
  --data data/bidding --out trained_models/bidding.pt --seed 20260829

# Use one checkpoint for any/all agent families. 0.65 is defensive;
# 0.50 balanced; 0.35 aggressive.
uv run python -m skat --rounds 1000 \
  -p0 mcts -p1 policy -p2 dqn -b0 neural -b1 neural -b2 neural \
  --bidding-model trained_models/bidding.pt --bidding-threshold 0.50
```

The net is deliberately a play/pass gate, not a bid-value predictor: the ±1
outcome says whether the game was won, but does not identify an optimal rung on
the bid ladder. An accepted hand therefore keeps the conservative ceiling and
cannot overbid because of the classifier. Different risk modes are thresholds
on the same probability model, so they can be compared without confounding the
result with separately initialized networks.
