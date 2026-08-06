# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pure-Python engine for the German card game Skat, built to train and run automated agents (rule-based and RL/DQN) against each other. There is no UI beyond a command-line agent; the primary entry points are the `skat` package (game engine + agents) and the `iss` package (parses game records from the International Skat Server for training data).

## Commands

Dependency management is via `uv` (see `pyproject.toml` / `uv.lock`). There is no dev-dependency group defined, so tools like `pytest`, `mypy`, and `ruff` are not installed into `.venv` by default — pull them in ad hoc with `uv run --with <tool>`.

```bash
# sync the environment (installs numpy, torch-cpu, tensorboard, tqdm)
uv sync

# ...plus pysgf, needed only to parse raw .sgf records (see ISS integration)
uv sync --extra sgf

# run the full test suite
uv run --with pytest pytest

# run a single test file / test
uv run --with pytest pytest tests/test_game.py
uv run --with pytest pytest tests/test_round.py::test_some_case -q

# type-check (currently NOT clean — pre-existing errors in skat/agents/dqn.py and skat/__main__.py)
uv run --with mypy mypy --ignore-missing-imports skat tests

# lint (currently NOT clean — a few pre-existing unused-import warnings)
uv run --with ruff ruff check .

# play/simulate a tournament from the CLI, e.g. 3 random agents for 33 rounds
uv run python -m skat --rounds 33 -p0 random -p1 random -p2 random

# extract won ISS games from a downloaded .sgf archive into per-game-type files
uv run python -m iss.extractor ~/Downloads/iss-games-07-2024.sgf

# record an MCTS self-play corpus, distil it into a policy net, then play it
uv run python -m skat --rounds 650 -p0 mcts -p1 mcts -p2 mcts \
    --simulations 2000 --determinizations 20 --seed 1 --record-mcts data/distill
uv run python -m skat.agents.distill.train --data data/distill \
    --arch transformer --out trained_models/policy.pt
uv run python -m skat --rounds 2000 -p0 policy -p1 random -p2 random --seed 7
```

Note: `tox.ini` predates the `uv` migration and references a `requirements.txt`/`setup.py` that no longer exist — don't rely on it; treat the `uv run` commands above as authoritative.

## Architecture

### Core game engine (`skat/`)

The engine is built around three layers that compose per-round:

- **`Card` / `Deck` / `Hand`** (`card.py`, `deck.py`, `hand.py`) — a `Card` is `(suit, rank)` with fixed point values (`0,0,0,2,3,4,10,11` for `7,8,9,J,Q,K,X,A`). `Card.np_index` maps every card to a fixed slot in a 32-element one-hot vector — this indexing is reused everywhere state is fed to a neural net. `Hand` is a `MutableSequence` with a custom bubble-sort (`Hand.sort`) driven by a `HandOrder` (suit/rank priority, plus an optional "pivot" rank/suit for trumps like Jacks) — each `Game` subclass defines its own `HandOrder`.

- **`Game` (ABC) → `SuitGame` / `Grand` / `Null`** (`skat/games/`) — each subclass defines trump rules (`trump_cards`), the base score multiplier (`value`), the Spielwert (`spielwert`/`matadors`, plus `spielwert_range` for the spread the skat can still cause — a "mit" hand can only gain, an "ohne" hand can only lose), and pairs with its own `Trick` subclass (`SuitGameTrick`/`GrandTrick`/`NullTrick`) that implements `forced_cards` (follow-suit rules) and `winner` (trick-resolution logic, including Jack-as-trump ordering). `Trick` is abstract; `TrickHistory` accumulates completed tricks across a round and can serialize to a one-hot vector per player.

- **`Round`** (`game.py`) — orchestrates one full hand: dealing, bidding (`bidding()`, alternating offers between front/middle/back hand per official Skat rules), skat pickup/press, game declaration, then trick-by-trick play via `step()`/`step_player()` until `is_finished`. Front/middle/back-hand seat rotation is derived from `dealer` and the previous trick's winner (`front_hand`/`middle_hand`/`back_hand` properties). `Round.get_state(player_id)` builds the observation fed to RL agents. It is **egocentric**: every per-player block is rotated by `seats_from(player_id)` so slot 0 is always the observer. That is what lets one network serve all three seats — indexed by absolute seat, the same situation gives a different vector from each chair and the net has to learn everything three times, which is why `-m0/-m1/-m2` used to mean three genuinely different nets rather than three instances. It also carries the observer's *role* (`am I the soloist`, plus where the soloist sits relative to it) and points as `[my party, their party]`; without those the same observation carried opposite rewards, since `cumulative_reward` flips sign on exactly that. `skat/models/observation.py`'s `INPUT_SIZE` (240) must stay in sync with this layout — `tests/test_state_vector.py` asserts it, and pins the seat-invariance property directly. `Round.cumulative_reward(player_id)` is the shaped reward signal used by `DQNAgent`.
  Once the game is declared, `Round.game_value` holds its Spielwert via `Game.spielwert(cards)` (the soloist's ten cards **plus the skat**, since the skat counts towards the matadors either way). `Round.overbid` compares it against `highest_bid` — an overbid soloist loses regardless of card points (überreizt, ISkO 5.2.3). Both are inert when bidding was skipped (`highest_bid` stays 0), which is the case for ISS replays and most tests.

- **`Tournament`** (`game.py`) — runs many `Round`s back to back, rotating the dealer seat, and can optionally sample deals from real ISS game records (`iss_file=`) instead of random shuffles. `seed` seeds one generator that yields a **fresh seed per round** (`next_seed()`); passing the tournament seed straight to every `Round` would deal the identical hand every time. `scores` is `[soloist_wins, defender_wins]` — by *role*, so it says nothing about which agent played well; `seat_scores` (a `SeatScore` per seat) and `summary()` are what compare agents.

- **`Player`** (`player.py`) — thin wrapper binding a seat (`seat_id`) to an `Agent` strategy; computes `valid_moves` (follow-suit enforcement) and delegates all decisions (bid, skat pickup, discard, declare, card choice) to `self.strategy`.

### Agents (`skat/agents/`)

`Agent` (ABC in `agents/__init__.py`) defines the strategy interface: `pickup_skat`, `press_skat`, `declare_game`, `choose_card`, plus optional `new_round_event` (called before dealing, for per-round state) and `trick_done_event` hooks.

**Bidding is shared, not per-agent** (`agents/bidding.py`). An agent's whole bidding character reduces to one number — the highest value it will say — so `Agent.bid` implements the Sagen/Hören mechanics once and only `Agent.max_bid` varies. That ceiling comes from a `BiddingStyle` passed to any agent's constructor:

| style | ceiling | overbids |
| --- | --- | --- |
| `CONSERVATIVE` (**default**) | `spielwert_range(hand)[0]` — the value guaranteed however the skat falls | never |
| `NOMINAL` | `spielwert(hand)` — the hand at face value | ~14% of solo rounds |
| `OPTIMISTIC` | `spielwert_range(hand)[1]` — the best the skat could give | ~81% |
| `RANDOM` | any rung on the ladder, hand ignored | ~98% |

The "never overbids" guarantee needs **two** things, not just a safe ceiling: the declared game must also cover the bid. Re-deriving `best_game` after picking the skat up can land on a cheaper suit and turn an affordable bid into an overbid (it did, in ~9% of deals), so agents that re-evaluate must declare via `covering_game(cards, highest_bid)`. An agent's `intended_game()` and `declare_game()` have to agree, or the style means nothing. CLI: `-b0/-b1/-b2 {conservative,nominal,optimistic,random}`.

`VALID_BIDS` (the official ladder) and the hand-evaluation helpers (`best_game`, `playable`, `trump_count`, `suit_length`, `aces`) live in `agents/bidding.py` too; `Agent.VALID_BIDS` re-exports the ladder.

- `RandomAgent` — uniform-random legal decisions at every step, *except* bidding: it inherits `CONSERVATIVE`, since a ceiling drawn from the whole ladder wins every auction and then loses to überreizt. Pass `BiddingStyle.RANDOM` for the old behaviour. It settles its declaration once per round in `new_round_event` so the game it bids on is the game it declares.
- `CommandLineAgent` — prompts a human via stdin/stdout. Nothing else subclasses it: `DQNAgent` used to, to borrow prompts for the decisions its network does not make, which meant unattended training blocked on `input()` at the first bid.
- `HeuristicDeclarer` (`agents/declaring.py`) — mixin giving any non-human agent automatic skat handling: `pickup_skat`, `press_skat`, `declare_game` and `intended_game`, all built on `agents/bidding.py`. `press_skat` and `declare_game` weigh up the *same* twelve cards (via `covering_game`), so the game the skat was pressed for is the game that gets declared. Used by both `MCTSAgent` and `DQNAgent`; mix it in *before* `Agent`.
- `DQNAgent` (`agents/dqn.py`) — wraps a `DQN` (`agents/rl/dqn.py`, a target/policy-network epsilon-greedy DQN with masked action selection via `agents/rl/mask.py:MaskedCategorical` so illegal cards are never sampled) and a `ReplayBuffer` (`agents/rl/buffer.py`). Logs to TensorBoard via `SummaryWriter`. `train=True` enables replay-buffer pushes and optimization on each `trick_done_event`; checkpoints are loaded/saved from a `.pt` path (`./trained_models/checkpoint_p{0,1,2}.pt` by CLI default). It learns card play only — bidding comes from `Agent`/`BiddingStyle` and the skat phase from `HeuristicDeclarer`, so training runs unattended. `--device {auto,cpu,cuda,cuda:1,…}` picks where the nets live, resolved by `agents/rl/dqn.py:resolve_device` (`auto` = cuda when usable, else cpu). Both NVIDIA and AMD are addressed as `"cuda"` — a ROCm build reports `torch.version.cuda is None` but `torch.version.hip` set and `torch.cuda.is_available()` True, which `gpu_backend()` disentangles for the logs. Which device `auto` picks therefore depends on *which interpreter launched the run*: this repo's venv pins the CPU-only torch wheel index (`[tool.uv.sources]`), while a system python may well have a CUDA or ROCm build — so `uv run python -m skat` and `python -m skat` can resolve differently. `DQN.__init__` prints the resolved device for exactly that reason. `Round.get_state_t` and `Hand.as_tensor_mask` deliberately build on the cpu — that is where the replay buffer keeps them — and `DQN.select_action`/`optimize_model` move what they need onto the training device. Training is split into a **warm-up** and a learning phase by `learn_start` (default 10000 transitions; a DQN agent contributes 10 per round, so ~1000 rounds). Throughout the warm-up `DQN.exploration_rate` is a flat 1.0 and `epsilon` is left untouched — the seed data is the only thing the net sees before its first gradient step, so it comes from uniformly random play rather than from a still-randomly-initialised network acting greedily. ε then decays *per action*, and only while `explore=True`. `learn_start` used to be `batch_size` doing double duty, which is why `batch_size` was 10000; it is now 256, and `learn_start < batch_size` is refused at construction. CLI: `--batch-size`, `--learn-start`, `--buffer-size`, `--gamma`, `--epsilon`, `--epsilon-decay` — all defaulting to `None` so `DQN.__init__` holds the actual defaults (`--epsilon-decay` used to default to `0.995` in argparse against `DQN`'s `0.99995` and silently won).
  `--dqn-arch {mlp,transformer}` picks the Q-network from `agents/rl/dqn.py:DQN_ARCHITECTURES` — deliberately a *separate* registry from `models/__init__.py:ARCHITECTURES`, whose `mlp` is `PolicyNet` and carries dropout for supervised training. The transformer is built with `dropout=0.0`, so the comparison against the dense net is architecture alone and nothing lands on the bootstrap target.
  Every forward that is not the gradient step runs through `agents/rl/dqn.py:evaluating`, a context manager that sets `eval()` and `no_grad()` and restores the previous mode. `DQN` never did either, which was invisible while `QNet` was the only model (no dropout, no batch norm) and becomes a correctness bug the moment `--dqn-arch transformer` is used: a training-mode forward makes `select_action` stochastic on top of ε-greedy — including on evaluation passes, where ε is never read — and turns the bootstrap target into a random variable. `tests/agents/rl/test_eval_mode.py` pins it against a deliberately dropout-heavy stub, because against `QNet` every one of those cases passes either way.
  Masking applies to the **bootstrap target** as well as to action selection: `Transition` carries a `next_mask` alongside `next_state` (both `None` exactly on a terminal transition), and `DQN.bootstrap` maxes only over cards still in hand. Unmasked it maxed over all 32, i.e. almost always over a card already played. The mask is the agent's remaining hand rather than its true legal moves, because the next trick has not been led when `trick_done_event` fires — it is the tightest mask that exists at that point.
  `gamma` is **1.0**: a round is exactly ten tricks and every terminal transition is flagged (`Round.step` passes `len(trick_history) == 10`), so returns are bounded without a discount. Each trick's reward is a *difference* of `Round.cumulative_reward`, so undiscounted they telescope to exactly the final outcome. It was 0.5, which left the terminal win/schneider/schwarz bonus worth `0.5^9 ≈ 0.002` by the first trick — the agent was effectively playing for card points and ignoring whether it won.
- `MCTSAgent` (`agents/mcts.py`) — determinized UCT / PIMC over the card-play phase, with its search internals in `agents/search/` (mirroring how `agents/rl/` backs `DQNAgent`). Each move samples `determinizations` complete deals consistent with what the agent has seen, runs an independent UCT tree per world (`search/uct.py`) and plays the best average value. Worlds come from `search/determinize.py`, which infers void suits by replaying the trick history through the game's own `Trick.forced_cards`. `search/playout.py:PlayoutState` is the search model — `Round` can't be cloned for search because it owns the agents that point back at it. Bidding/skat/declaration are not searched: the ceiling comes from its `BiddingStyle`, and it declares via `covering_game` so a post-skat re-evaluation can't undercut what it bid. It passes outright on a hand `playable()` rejects. CLI: `-p0 mcts --simulations N --determinizations M`. `search()` is split into `tally()` (run the worlds, return `{move: [visits, value_sum]}`) and `best()` (the argmax of `value_sum / visits`, `-1` for an unvisited move) so `RecordingMCTSAgent` can write the tally down on the way past; `search` itself still short-circuits a forced move without searching. Note `best` scores by *mean value*, not by visits — the most-visited move and the played move genuinely disagree, which is why the distillation corpus stores the choice rather than deriving it.
- `RecordingMCTSAgent` (`agents/mcts.py`) — an `MCTSAgent` that hands every searched decision to a recorder (`agents/distill/samples.py:ShardWriter`). Play is identical to the base agent given the same seed. It lives beside the agent it subclasses rather than in `agents/distill/`, because it is the one piece of the distillation that has to know about `Player`/`Round` — same split as `agents/dqn.py` above the `Round`-free `agents/rl/`. The observation is taken inside `search`, i.e. *before* `choose_card` removes the card from the hand, so it is the one the student will face. Forced moves are not recorded: a masked softmax over one legal card has probability 1 and gradient 0, and `PolicyAgent.select` short-circuits identically, so there is no train/serve skew. CLI: `--record-mcts DIR` on any run with an `mcts` seat, closed in a `finally` so a Ctrl-C keeps the tail.
- `PolicyAgent` (`agents/policy.py`) — a `PolicyNet` (`models/policy.py`) distilled from `MCTSAgent`, playing the masked argmax of 32 logits in one forward pass instead of `simulations` playouts: ~250 rounds/s against MCTS's ~4.4. Card play only, like `DQNAgent`. **One checkpoint serves all three seats** — `get_state` is egocentric, so the corpus pools every chair into one file and `--policy PATH` is a single flag, deliberately *not* the DQN's per-seat `-m{i}` (whose defaults are named `checkpoint_p{i}.pt` and which exist because the observation used to be seat-indexed). A missing checkpoint raises rather than starting from a random init the way `DQN` does: this agent never trains, so a random init would just be a bad player that reads like a trained one in `Tournament.summary()`. `--policy-temperature 0` (the default) is exactly deterministic and touches no RNG; `>0` samples, which is there for a future ExIt round where a greedy student would generate a degenerate self-play distribution.
- `DeepAgent` (`agents/deep.py`) — unfinished stub (not wired into `skat/__main__.py`'s `get_player`); `choose_card` still just picks randomly with a `# TODO: select card from model` marker, and the pickup/declare/press methods are unimplemented (`pass`). It does bid, via the shared machinery.

### Distillation (`skat/agents/distill/`)

Expert iteration, stages 1 and 2: turn `MCTSAgent`'s searches into a supervised corpus, train a network on it, play it with `PolicyAgent`. The motivation is that a trained `DQNAgent` beats `RandomAgent` but is far weaker than `MCTSAgent`, and the difference is search, not the RL algorithm — Double DQN or PPO would learn the same shaped reward through the same net. Stage 3 (feeding the logits back into `search/uct.py` as a PUCT prior, so the *teacher* gets stronger per simulation instead of merely cheaper) is not built.

Same layering rule as `agents/search/` and `agents/rl/`: **nothing in `agents/distill/` imports `Agent`, `Player` or `Round`** — arrays, tensors and `Card` only.

- `samples.py` — `ShardWriter` buffers `(state, visits, values, mask, group, choice)` rows and flushes compressed `.npz` shards named `mcts-{tag}-{pid}-{index}.npz`, so N collectors share one directory and a crash costs minutes rather than the corpus. `visits` are stored **raw**, not normalized, so temperature stays a knob on the training run; `values` and `choices` are stored because re-collecting to add a column costs an hour. `groups` is the round id — every sample of one round shares it, and `train.py` splits on it. `load_shards` concatenates, offsets the per-shard group ids so they stay distinct, and **refuses** a shard whose `input_size` differs from `PolicyNet.INPUT_SIZE`: a change to `Round.get_state` retires the corpus, and this is the only thing that says so.
- `targets.py` — `visit_targets` (normalized counts, temperature-sharpened) and `value_targets` (softmax over each move's *mean* value, which is what `MCTSAgent.best` maximises, and which is averaged over every determinization so it is the less noisy of the two when the trees are shallow). `--target {visits,value}` picks between them without re-collecting.
- `loss.py` — `masked_log_softmax` + `policy_loss` (cross-entropy against the soft target). Deliberately **out-of-place**, unlike `agents/rl/mask.py:MaskedCategorical`, whose `masked_fill_` mutates the network's own output — fine for a no-grad argmax, wrong in a backward pass. The second `masked_fill(~masks, 0.0)` makes the loss provably independent of an illegal logit rather than merely insensitive to it.
- `checkpoint.py` — `save_policy`/`load_policy` wrap the state dict in `{"format": "skat-policy-v1", …}`. This exists because `DQN` saves a *bare* state dict of an identically shaped net: without the guard, a DQN checkpoint loads into a `PolicyAgent` cleanly and it plays Q-values as if they were policy logits, silently. Both nets use fixed `nn.Linear` rather than `nn.LazyLinear`, for the same reason: a lazy layer materializes from whatever state dict it is handed, so a checkpoint from another observation layout loads cleanly and goes wrong later. `PolicyNet` has real `Dropout`; `QNet` deliberately has none, because dropout in a Q-network perturbs the bootstrap target as well as the prediction.
- `train.py` — `python -m skat.agents.distill.train --data DIR --out PATH`. AdamW, group-aware split (`split_by_group`), best-val checkpointing, TensorBoard `Loss/train`, `Loss/val`, `Agreement/train`, `Agreement/val` (top-1 against the teacher's actual choice). The split must cut on round boundaries: the ~22 decisions of a round share a hand, a soloist and a trick history, so a row-wise split puts near-duplicates on both sides and overfitting reads as convergence.

Collection throughput: ~0.5 rounds/s per process with three MCTS seats at `--simulations 2000 --determinizations 20`, ~23 recorded decisions per round, ~66 bytes/row compressed. Parallelise over seats with `--seed`; the CLI offsets the seed per seat (`args.seed + i`) so three MCTS agents at one table don't march their determinizations in step.

Measured on a 102k-decision corpus (~5200 rounds, 8 workers, ~20 min wall-clock, 6.5 MB), seat 0 against two `RandomAgent`s over the same seeded deals:

| agent | as soloist | on defence | params | speed |
| --- | --- | --- | --- | --- |
| `RandomAgent` | 29% | 69% | — | — |
| `PolicyAgent`, `--arch mlp` | 80% | 82% | 2.94M | ~250 rounds/s |
| `PolicyAgent`, `--arch transformer` | **86%** | **85%** | 536k | ~120 rounds/s |
| `MCTSAgent` (2000 sims) | 91% | 86% | — | ~1.5 rounds/s |

So the distillation recovers most of the gap for ~1/150th of the cost, and the transformer closes most of what was left — its defence is level with the teacher's, its soloing still ~5 points behind.

`--target value` was run as a control on the MLP and came out level (80.4% vs 81.6% of rounds won, ~1 standard error) despite being a far sharper target; see `targets.py`.

### Architectures (`--arch`)

Two networks are registered in `skat/models/__init__.py:ARCHITECTURES`, and `--arch` picks one at training time. Serving does **not** take a flag: `save_policy` writes the name into the checkpoint and `load_policy` dispatches on it, so `--policy PATH` keeps meaning "whatever is in this file" and `PolicyAgent` never learned there is more than one. A checkpoint with no `arch` key means `mlp` — the key arrived with the second architecture and the files that predate it are all MLPs.

- `mlp` — `models/policy.py:PolicyNet`, the 5-layer dense baseline over the flat 240-vector.
- `transformer` — `models/card_transformer.py:CardTransformer`, self-attention over **32 tokens, one per card in the pack**. The observation decomposes losslessly into 7 mutually exclusive per-card status columns (held, played by me/next/previous, on the table by me/next/previous), so the tokenisation is exact and needs no recollection — it happens inside `forward`, keeping the `(B,240) -> (B,32)` contract every caller depends on. Card identity is a suit+rank embedding, which is also why there is no positional encoding; the head is one shared `Linear(d_model, 1)` per token, which is what makes the output line up with the legal-move mask. `tests/test_card_tokens.py` pins the tokeniser against real `Round.get_state` output rather than synthetic vectors.

**The transformer wins, and it wins on variance, not capacity.** With a fifth of the parameters it reaches 60% top-1 agreement with the teacher against the MLP's ~54%, and its train/val gap is *zero* (59.4% / 59.8%) where the MLP's was 5–11 points. Paired over 2000 identical deals it takes 85.5% of rounds against the MLP's 81.7%, McNemar p < 0.001 (257 discordant rounds to 181); against the `--target value` MLP it is 84.4% to 80.8%, p = 0.001. Small data with a strong known symmetry is exactly where architecture pays, and the MLP was spending 2.94M parameters rediscovering that input slot 3 and output slot 3 are the same card.

`--features raw` is the control: it drops the derived per-card features (`is_trump`, trick order, led suit) and keeps only status. It came out **indistinguishable** from `full` (84.2% vs 84.2%, 197 discordant to 196, p = 1.000), so the win is attention over a card set, not the domain features — worth knowing before adding more of them.

Per-architecture training defaults live in one table, `train.py:ARCH_DEFAULTS`, with `default=None` in argparse and `resolve()` filling the gap; the run prints which defaults it used. This is the `--epsilon-decay` bug inverted — that one was a real argparse default silently outranking the class default.

### Hand prediction (`skat/agents/belief/`)

`MCTSAgent` samples worlds *uniformly* over everything consistent with what it has seen. Consistency is a weak filter — it rules out the observer's own cards, the played cards and shown voids, and nothing else — so a Grand declarer is as likely to be dealt no jacks as three, and the simulations spent on worlds nobody would reach are lost. `BeliefNet` estimates where the unseen cards actually are, and `BeliefMCTSAgent` samples its determinizations from that.

- `models/belief.py:BeliefNet` — 32×3 logits (next player / previous player / skat), reusing `CardTransformer`'s tokeniser and encoder through its `trunk()` seam. **Its input is wider than the observation**: `INPUT_SIZE = 304` = `get_state` (240) ++ both opponents' 32-wide void masks. That block exists because `Round.get_state` ORs all ten tricks into one set per seat, losing which cards fell *together* — so "failed to follow suit" is structurally unreconstructible from the observation, and void inference is the strongest read in Skat. The masks come from the search's own `infer_voids`, so the net is trained on exactly the evidence the sampler constrains with. Measured on the corpus, **69% of positions carry at least one void bit** (averaging 10).
- `belief_recorder.py` — ground truth, mirroring `MCTSAgent.determinize`'s exact definition of "already seen" (own hand, played cards, and the skat only for a soloist who picked it up). A mixin, because the labels do not depend on who is playing; the agent only decides which positions the corpus covers.
- `search/determinize.py:deal_unseen` gains an optional `weights` argument. Left `None` — which is what `MCTSAgent` passes — the behaviour is exactly what it always was. `_pick` documents the approximation: assigning card by card in proportion to a per-card belief is a *sequential* approximation of the belief-weighted distribution over whole worlds, since capacity couples the cards.
- `belief_mcts.py:BeliefMCTSAgent(MCTSAgent)` overrides only the new `deal()` seam, so any difference against the baseline is attributable to the sampling alone. The net runs **once per decision**, not once per determinization.

Collection is cheap — `policy` seats give ~1100 observations/s, so 304k observations from 10,200 rounds took under a minute against the MCTS corpus's 20. Training: ~40 min for 12 epochs.

| | cross-entropy | accuracy |
| --- | --- | --- |
| uniform determinization (`capacity_prior`) | 0.9026 | 46.5% |
| `BeliefNet` | **0.5884** | **68.5%** |

The baseline is deliberately `capacity_prior` — what the uniform dealer already believes given the slot sizes — not a flat 1/3, against which anything looks good.

**Playing strength depends on who is at the table, and this is the load-bearing caveat.** Seat 0 paired over 1000 identical deals at 600 simulations:

| opponents | `mcts` | `belief-mcts` | McNemar |
| --- | --- | --- | --- |
| two `RandomAgent`s | 86.4% | 86.1% | p = 0.88, indistinguishable |
| two `PolicyAgent`s | 47.7% | **51.7%** | 79 to 39, **p < 0.001** |

A belief is always relative to an opponent model. Trained on competent play, the net reads discards as evidence; a random player breaks that inference and the evidence becomes misleading, which is why the first table is flat. Against the population it was trained on, the gain is real and lands **almost entirely on defence** (47.4% vs 41.8%) — the soloist knows its own hand and has less to infer, while a defender must place every card between its partner and the declarer.

Retrain the belief on whatever population it will actually face; do not assume it transfers.

### ISS integration (`iss/`)

**`pysgf` is an optional dependency** (`[project.optional-dependencies] sgf`), and it is needed for exactly one thing: turning raw `.sgf` *text* into an `ISSGame`. Everything else here — the `GameType` enum, the card table, `.pkl` corpora, `sample()`, the `Deck` reconstruction and `game_converter` — is plain Python and works without it, so a corpus parsed once can be replayed through `Tournament(iss_file=...pkl)` forever afterwards on an install that has no pysgf. `iss.HAS_SGF_SUPPORT` is the capability flag; `iss.require_sgf_support()` guards `ISSGame._parse_sgf_line` and `ISSGames.read_sgf` (the latter *before* opening the file, since these archives are gigabytes) and raises `iss.MissingSgfSupport`, an `ImportError` subclass carrying install instructions. Tests that parse `.sgf` are `skipUnless(HAS_SGF_SUPPORT)`.

Parses SGF-formatted game logs from the International Skat Server into `ISSGame`/`ISSGames` objects (regex-based parsing of the SGF `MV`/`R` properties — deliberately tolerant of malformed/edge-case records via `is_valid`/`is_won` flags rather than raising). `ISSGames` supports `.sgf` (raw) and `.pkl` (pre-parsed, pickled) inputs and can `sample()` a seeded subset. `skat/utils/game_converter.py:get_game` bridges `iss.GameType` to the corresponding `skat.games.Game` subclass so sampled ISS deals can be replayed through `Round`/`Tournament`. `iss/extractor.py` is a standalone CLI (`python -m iss.extractor <file>`) that filters an `.sgf` archive down to won games, split by game type, into new `.sgf` files alongside the source.

### Data ingestion (`skat/utils/`)

`skatstube.py` parses a different, JSON-based game-log format (from skatstube.de) into the same card/hand representation — independent of the ISS pipeline, used for building additional training corpora. `crawler.py` fetches such records; `misc.py` holds small shared helpers (e.g. `disjoint()` used by `Round` to validate `initial_cards`).

## Working with this codebase

- Card/suit/rank indices are load-bearing: `SUITS = ("♦","♥","♠","♣")` and `RANKS = ("7","8","9","J","Q","K","X","A")` order is assumed throughout (`Card.np_index`, trump ordering, `HandOrder` defaults). Don't reorder these without auditing every consumer.
- `Trick.winner`/`forced_cards` logic differs meaningfully between `SuitGame`, `Grand`, and `Null` (trump handling, Jack precedence) — when fixing a rules bug, check whether it's specific to one game type or shared `Trick` behavior.
- State-vector shape (`Round.get_state`) and `skat/models/observation.py:INPUT_SIZE` are coupled by hand, not enforced by code — a change to one requires updating the other. `observation.py` is the single home for the width; the per-block *offsets* are deliberately restated in `tests/test_state_vector.py` and `models/card_transformer.py` rather than shared, so two independent statements catch a drift in one.
- `.venv`, `runs/`, `trained_models/`, `*.egg-info`, and various `__pycache__`/tool caches are present in the working tree; they're build/runtime artifacts, not source.
