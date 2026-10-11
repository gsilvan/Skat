# Consolidation from master

Source: master `9df7c03`, vibe-coding `0507c3c`. Both branches are preserved.
New branch: `consolidation`. No old models or corpora are migrated.

## Adoption decisions

- Keep: official bidding, passed rounds, matadors, seeded deals, seat statistics,
  fixed-deal skat pickup, optional SGF, egocentric observations and hidden-information fixes.
- Adapt: MCTS/UCT to shared official scoring and outcome-only reward; transformer
  distillation to one architecture, one target and strictly versioned artifacts.
- Archive in vibe-coding: DQN, belief, neural/search bidding, bonus objectives,
  alternative policy controls, large training orchestrator, web UI, unfinished DeepAgent.
- Add: complete declarations, final scoring, validated engine actions, restricted
  player views, concession/claims, four CLI commands and TOML configuration.

Rulebook: https://dskv.de/app/uploads/sites/43/2022/11/ISkO-2022.pdf

## Original changes reviewed

```text
8f2283e Implement Sagen/Hören bidding and make tournaments measurable
5d11076 Score the Spielwert and enforce the überreizt rule
7c80c5f Make bidding behaviour a setting shared by every agent
60f2c36 Add an MCTS agent using determinized UCT
79206e2 Ignore training artifacts and scraped corpora
0774ff9 Document the engine, agents and bidding in CLAUDE.md
a909d00 Precompute the trump and suit tables
635900a Make pysgf an optional dependency
94cf297 Make DQNAgent trainable without a human at the keyboard
c1b78cd Add a --device option for DQN training
050fe65 Make the RL observation egocentric and tell it which side it is on
3d80081 Give the DQN a warm-up instead of a 10000-sample batch
b81dd66 Mask the DQN bootstrap target and stop discounting the win
e12ac39 Distil the MCTS agent into a policy network
396da46 implement transformer
8a57366 Predict where the hidden cards are and sample worlds from that
490f56d Rename SuitSoloNet to QNet and give the observation its own module
1a2fc80 Give the DQN eval-mode discipline and a transformer Q-network
10c81b6 docs: move agent instructions to AGENTS.md
2b4b8c1 Add reproducible agent benchmark and neural bidding
e3e1350 Add local Flask and React game interface
2f71a71 Add MCTS-based bidding strategy
19fb644 Prioritize party victory and separate bonuses after secured wins
5a3efba Break exact MCTS win-rate ties with independent outcome bonuses
b2c3459 Document larger win-first policy training and independent confirmation
63a1c81 Remove hidden skat and game-value leakage from agent decisions
0507c3c Document rollback of perfect-information teacher experiment
```

## Scope and checks

The production Python code (`skat`, `iss`, former `web` backend) has 48 files
and about 4,550 lines, compared with 85 files / 10,488 lines on vibe-coding
and 34 files / 2,675 lines on master. The engine grew to cover the official
contracts and scoring; experimental learning systems and web code are excluded.

Automated tests contain only a four-round, one-epoch, tiny-network training
smoke check. Long collection, training and playing-strength evaluation are
separate commands and are never invoked by pytest. Core installation is
verified without Torch; the full installation also tests optional ISS parsing.

The new CLI regression caught an empty recorder being treated as absent.
Collection now checks its presence explicitly. Corpus group identifiers are
validated and preserved across shards and independent collectors. Evaluation
chunks retain global deal indices, including dealer rotation, and are checked
against an unpartitioned evaluation.

ISS parsing preserves Hand and Ouvert separately: Null ouvert picks up the
skat, while Null Hand ouvert does not. Replays use the recorded pressed hand
and keep the soloist's rightful skat knowledge. Legacy ISS pickle records that
lack Ouvert metadata are rejected for replay rather than silently declaring a
different game; re-extract them from SGF. Invalid agent claims are validated
before exposing the soloist's cards.

## Measured validation

See [validation.md](validation.md) for the completed fresh collection, GPU
training, full paired comparison and the small-test boundary. Raw artifacts
remain local; the compact benchmark manifest/results are versioned.

## Parallel MCTS collection

`collect` now runs independent games in separate CPU processes, using all
available CPUs by default (`--workers N` limits them). A global progress bar
counts completed rounds. Each worker owns its agents and recorder; global
round and seat seeds preserve the games and search tallies across worker counts.
Worker shards are published atomically and remain available after interruption.
`collection.json` records completion state and provenance. Single-game `play`
searches remain serial.

A standalone 32-round check at 600 simulations / 20 determinizations, seed 123,
on the Ryzen 7 5800X measured 33.9 seconds with one worker, 5.6 with eight and
5.4 with sixteen logical-CPU workers: about 6.3× faster than serial. All runs
recorded identical arrays for all 679 decisions. Shards load in global round
order to keep the row order and seeded training split independent of worker
count. Short regressions compare the full recorded
search arrays across worker counts; a separate real Ctrl-C check verifies that
worker processes exit and valid decisions are retained. These timing runs are
not part of the automated tests.
