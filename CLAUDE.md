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
```

Note: `tox.ini` predates the `uv` migration and references a `requirements.txt`/`setup.py` that no longer exist — don't rely on it; treat the `uv run` commands above as authoritative.

## Architecture

### Core game engine (`skat/`)

The engine is built around three layers that compose per-round:

- **`Card` / `Deck` / `Hand`** (`card.py`, `deck.py`, `hand.py`) — a `Card` is `(suit, rank)` with fixed point values (`0,0,0,2,3,4,10,11` for `7,8,9,J,Q,K,X,A`). `Card.np_index` maps every card to a fixed slot in a 32-element one-hot vector — this indexing is reused everywhere state is fed to a neural net. `Hand` is a `MutableSequence` with a custom bubble-sort (`Hand.sort`) driven by a `HandOrder` (suit/rank priority, plus an optional "pivot" rank/suit for trumps like Jacks) — each `Game` subclass defines its own `HandOrder`.

- **`Game` (ABC) → `SuitGame` / `Grand` / `Null`** (`skat/games/`) — each subclass defines trump rules (`trump_cards`), the base score multiplier (`value`), the Spielwert (`spielwert`/`matadors`, plus `spielwert_range` for the spread the skat can still cause — a "mit" hand can only gain, an "ohne" hand can only lose), and pairs with its own `Trick` subclass (`SuitGameTrick`/`GrandTrick`/`NullTrick`) that implements `forced_cards` (follow-suit rules) and `winner` (trick-resolution logic, including Jack-as-trump ordering). `Trick` is abstract; `TrickHistory` accumulates completed tricks across a round and can serialize to a one-hot vector per player.

- **`Round`** (`game.py`) — orchestrates one full hand: dealing, bidding (`bidding()`, alternating offers between front/middle/back hand per official Skat rules), skat pickup/press, game declaration, then trick-by-trick play via `step()`/`step_player()` until `is_finished`. Front/middle/back-hand seat rotation is derived from `dealer` and the previous trick's winner (`front_hand`/`middle_hand`/`back_hand` properties). `Round.get_state(player_id)` builds the observation fed to RL agents. It is **egocentric**: every per-player block is rotated by `seats_from(player_id)` so slot 0 is always the observer. That is what lets one network serve all three seats — indexed by absolute seat, the same situation gives a different vector from each chair and the net has to learn everything three times, which is why `-m0/-m1/-m2` used to mean three genuinely different nets rather than three instances. It also carries the observer's *role* (`am I the soloist`, plus where the soloist sits relative to it) and points as `[my party, their party]`; without those the same observation carried opposite rewards, since `cumulative_reward` flips sign on exactly that. `skat/models/suit_solo.py`'s `INPUT_SIZE` (240) must stay in sync with this layout — `tests/test_state_vector.py` asserts it, and pins the seat-invariance property directly. `Round.cumulative_reward(player_id)` is the shaped reward signal used by `DQNAgent`.
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
- `DQNAgent` (`agents/dqn.py`) — wraps a `DQN` (`agents/rl/dqn.py`, a target/policy-network epsilon-greedy DQN with masked action selection via `agents/rl/mask.py:MaskedCategorical` so illegal cards are never sampled) and a `ReplayBuffer` (`agents/rl/buffer.py`). Logs to TensorBoard via `SummaryWriter`. `train=True` enables replay-buffer pushes and optimization on each `trick_done_event`; checkpoints are loaded/saved from a `.pt` path (`./trained_models/checkpoint_p{0,1,2}.pt` by CLI default). It learns card play only — bidding comes from `Agent`/`BiddingStyle` and the skat phase from `HeuristicDeclarer`, so training runs unattended. `--device {auto,cpu,cuda,cuda:1,…}` picks where the nets live, resolved by `agents/rl/dqn.py:resolve_device` (`auto` = cuda when usable, else cpu). Both NVIDIA and AMD are addressed as `"cuda"` — a ROCm build reports `torch.version.cuda is None` but `torch.version.hip` set and `torch.cuda.is_available()` True, which `gpu_backend()` disentangles for the logs. Which device `auto` picks therefore depends on *which interpreter launched the run*: this repo's venv pins the CPU-only torch wheel index (`[tool.uv.sources]`), while a system python may well have a CUDA or ROCm build — so `uv run python -m skat` and `python -m skat` can resolve differently. `DQN.__init__` prints the resolved device for exactly that reason. `Round.get_state_t` and `Hand.as_tensor_mask` deliberately build on the cpu — that is where the replay buffer keeps them — and `DQN.select_action`/`optimize_model` move what they need onto the training device. Training is split into a **warm-up** and a learning phase by `learn_start` (default 10000 transitions; a DQN agent contributes 10 per round, so ~1000 rounds). Throughout the warm-up `DQN.exploration_rate` is a flat 1.0 and `epsilon` is left untouched — the seed data is the only thing the net sees before its first gradient step, so it comes from uniformly random play rather than from a still-randomly-initialised network acting greedily. ε then decays *per action*, and only while `explore=True`. `learn_start` used to be `batch_size` doing double duty, which is why `batch_size` was 10000; it is now 256, and `learn_start < batch_size` is refused at construction. CLI: `--batch-size`, `--learn-start`, `--buffer-size`, `--epsilon`, `--epsilon-decay` — all defaulting to `None` so `DQN.__init__` holds the actual defaults (`--epsilon-decay` used to default to `0.995` in argparse against `DQN`'s `0.99995` and silently won).
- `MCTSAgent` (`agents/mcts.py`) — determinized UCT / PIMC over the card-play phase, with its search internals in `agents/search/` (mirroring how `agents/rl/` backs `DQNAgent`). Each move samples `determinizations` complete deals consistent with what the agent has seen, runs an independent UCT tree per world (`search/uct.py`) and plays the best average value. Worlds come from `search/determinize.py`, which infers void suits by replaying the trick history through the game's own `Trick.forced_cards`. `search/playout.py:PlayoutState` is the search model — `Round` can't be cloned for search because it owns the agents that point back at it. Bidding/skat/declaration are not searched: the ceiling comes from its `BiddingStyle`, and it declares via `covering_game` so a post-skat re-evaluation can't undercut what it bid. It passes outright on a hand `playable()` rejects. CLI: `-p0 mcts --simulations N`.
- `DeepAgent` (`agents/deep.py`) — unfinished stub (not wired into `skat/__main__.py`'s `get_player`); `choose_card` still just picks randomly with a `# TODO: select card from model` marker, and the pickup/declare/press methods are unimplemented (`pass`). It does bid, via the shared machinery.

### ISS integration (`iss/`)

**`pysgf` is an optional dependency** (`[project.optional-dependencies] sgf`), and it is needed for exactly one thing: turning raw `.sgf` *text* into an `ISSGame`. Everything else here — the `GameType` enum, the card table, `.pkl` corpora, `sample()`, the `Deck` reconstruction and `game_converter` — is plain Python and works without it, so a corpus parsed once can be replayed through `Tournament(iss_file=...pkl)` forever afterwards on an install that has no pysgf. `iss.HAS_SGF_SUPPORT` is the capability flag; `iss.require_sgf_support()` guards `ISSGame._parse_sgf_line` and `ISSGames.read_sgf` (the latter *before* opening the file, since these archives are gigabytes) and raises `iss.MissingSgfSupport`, an `ImportError` subclass carrying install instructions. Tests that parse `.sgf` are `skipUnless(HAS_SGF_SUPPORT)`.

Parses SGF-formatted game logs from the International Skat Server into `ISSGame`/`ISSGames` objects (regex-based parsing of the SGF `MV`/`R` properties — deliberately tolerant of malformed/edge-case records via `is_valid`/`is_won` flags rather than raising). `ISSGames` supports `.sgf` (raw) and `.pkl` (pre-parsed, pickled) inputs and can `sample()` a seeded subset. `skat/utils/game_converter.py:get_game` bridges `iss.GameType` to the corresponding `skat.games.Game` subclass so sampled ISS deals can be replayed through `Round`/`Tournament`. `iss/extractor.py` is a standalone CLI (`python -m iss.extractor <file>`) that filters an `.sgf` archive down to won games, split by game type, into new `.sgf` files alongside the source.

### Data ingestion (`skat/utils/`)

`skatstube.py` parses a different, JSON-based game-log format (from skatstube.de) into the same card/hand representation — independent of the ISS pipeline, used for building additional training corpora. `crawler.py` fetches such records; `misc.py` holds small shared helpers (e.g. `disjoint()` used by `Round` to validate `initial_cards`).

## Working with this codebase

- Card/suit/rank indices are load-bearing: `SUITS = ("♦","♥","♠","♣")` and `RANKS = ("7","8","9","J","Q","K","X","A")` order is assumed throughout (`Card.np_index`, trump ordering, `HandOrder` defaults). Don't reorder these without auditing every consumer.
- `Trick.winner`/`forced_cards` logic differs meaningfully between `SuitGame`, `Grand`, and `Null` (trump handling, Jack precedence) — when fixing a rules bug, check whether it's specific to one game type or shared `Trick` behavior.
- State-vector shape (`Round.get_state`) and `SuitSoloNet.INPUT_SIZE` are coupled by hand, not enforced by code — a change to one requires updating the other.
- `.venv`, `runs/`, `trained_models/`, `*.egg-info`, and various `__pycache__`/tool caches are present in the working tree; they're build/runtime artifacts, not source.
