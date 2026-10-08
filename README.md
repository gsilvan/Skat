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

Card play uses a **win-first objective**. Before victory is secured, MCTS
optimises party win probability first. Exact ties in estimated win probability
are broken by independently estimated winning-party bonuses; no tolerance or
weighted sum can compensate for a lower win rate. Its primary rollout objective is frozen at the
search root: collecting more points in a simulated future win cannot compensate
for a lower chance of winning now. Once the party has secured victory with the
engine's 61/60 thresholds, search maximises a nonnegative bonus:
`0.6 * party_points / 120 + 0.2 * Schneider + 0.2 * Schwarz`.
Missing Schneider incurs no penalty, and points remain useful even when the
next stage is unreachable. The existing 91/90 Schneider thresholds and the
engine's 120-point Schwarz definition are retained. Null remains a no-trick goal.

DQN uses two independent Q-networks. One learns undiscounted win/loss returns;
the other learns bonus increments on decisions made after secured victory.
Only the win network chooses cards before victory, and only the bonus network
afterwards. Old bare DQN checkpoints must be kept as baselines and newly trained
agents written to fresh paths; the new checkpoints identify their objective.
Policy architectures and observation widths stay unchanged, but their teacher
data must be recollected. Recorded shards and policy checkpoints retain the
reward objective, and loading mixed old/new shards is rejected.

Observations now contain **known party-point lower bounds**, not hidden skat
points. Only a soloist who picked up the skat may include its points; in a hand
game everyone excludes the unknown skat. MCTS freezes secured victory using
this observer-visible information, and evaluates overbidding from sampled
hands rather than the engine's hidden real Spielwert. Actual game scoring and
training outcome rewards still include the skat normally.

The semantic schema is `skat-observer-points-v2` despite the unchanged 240/304
input widths. Old policy/belief corpora are refused and must be recollected;
new win-first DQN checkpoints also require this schema. Legacy policy/belief
models remain loadable with an explicit warning, but their learned point
features have shifted and should be retrained. Historical pre-fix benchmark
scores used privileged information and are **not fair strength baselines**.

MCTS records independent bonus sums alongside its primary values. Value-target
distillation uses them only to distinguish exactly tied primary maxima, so its
preferred target still matches the teacher's win-first choice. To disable
the experimental tie-breaker for comparisons, construct
`MCTSAgent(bonus_tiebreak=False)`. DQN still uses the secured-victory bonus gate.

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

## MCTS bidding

`-b{i} mcts` evaluates Karo, Herz, Pik, Kreuz and Grand before entering the
auction. It samples possible defender hands and skat cards, takes and presses
the skat for each fixed candidate, and uses binary win/loss UCT searches to
estimate its solo win rate. All five candidates share the first set of worlds;
the best game is then tested again on fresh worlds. The default confirmation
threshold is 65%.

```bash
uv run python -m skat --rounds 1000 --seed 42 \
  -p0 policy -p1 policy -p2 policy -b0 mcts \
  --policy trained_models/policy.pt \
  --mcts-bidding-threshold 0.65
```

The defaults spend 600 playouts over 20 worlds on each candidate, followed by
1,200 playouts over 40 fresh confirmation worlds. The cost and confidence can
be controlled independently:

```bash
--mcts-bidding-simulations 600 \
--mcts-bidding-determinizations 20 \
--mcts-bidding-confirmation-simulations 1200 \
--mcts-bidding-confirmation-determinizations 40
```

Once accepted, the selected game is kept through pickup, pressing and
declaration. Its conservative Spielwert ceiling is used, so the search changes
which games are attempted without introducing overbids. Null is deliberately
not considered: its objective and declaration values need a separate
loss-of-trick model rather than the point-game comparison requested here.
