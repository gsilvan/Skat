# Win-first card-play objective

> These historical measurements predate the hidden-skat observation fix and
> must not be treated as fair baselines for `skat-observer-points-v2` agents.
> Old policy/belief corpora require recollection; legacy model loading warns.

MCTS now scores win/loss alone until either party has secured victory at the
search root. This decision is frozen through every cloned rollout: a simulated
future bonus cannot compensate for a reduced current win probability. After
secured victory it uses the winning party's nonnegative bonus:

`0.6 * points / 120 + 0.2 * Schneider + 0.2 * Schwarz`.

The existing engine thresholds are unchanged. Null remains a no-trick outcome.
DQN uses independent win and bonus networks, consults the bonus only after a
secured own victory, and stops primary bootstrapping after a resolved result.
Bonus learning cannot change the win network's weights. Existing policy files
still encode their original teacher; new policies need freshly collected data.

## Paired comparison

Evaluation seed: `3096181844`, derived with
`stage_seed(20261008, "reward-comparison")`. Each objective played 200 new deals
from every seat against two RandomAgents and against two copies of the original
benchmark's `policy-transformer.pt`: 1,200 played candidate outcomes per
objective. MCTS budgets were 600 simulations / 20 determinizations. Policy
inference used CPU with one Torch thread. Bidding remained conservative.

The baseline reinstated exactly the previous rollout utility
`0.9 * won + soloist_points / 1200`; all other search machinery was shared.
Same deals and opponent seeds do not mean identical random-opponent actions:
their RNG streams evolve with legal moves, so subsequent random declarations
can differ. Against the deterministic policy panel, all declarations and
soloist seats were identical between objectives.

| Metric | Previous utility | Win-first |
| --- | ---: | ---: |
| Total party wins | 900/1,200 (75.00%) | 905/1,200 (75.42%) |
| Against random | 85.67% | 84.83% |
| Against original policy | 64.33% | 66.00% |
| As soloist | 93.38% | 94.44% |
| On defence | 69.61% | 69.89% |
| Party Schneider wins | 332 | 259 |
| Party Schneider losses | 50 | 53 |
| Party Schwarz wins (engine definition) | 11 | 7 |
| Party Schwarz losses (engine definition) | 1 | 2 |
| Party card points per round | 74.39 | 72.60 |

106 previously lost outcomes became wins; 101 previously won outcomes became
losses. Total win-rate delta: +0.42 percentage points. A paired bootstrap with
10,000 resamples of entire deal seeds (all seats and panels kept together)
gave a descriptive 95% interval of -1.92 to +2.75 percentage points. Per-panel
intervals were -4.33 to +2.67 for random and -1.17 to +4.50 for policy.

This run provides no statistically established playing-strength improvement.
It also shows a tradeoff: postponing the bonus objective until secured victory
reduces Schneider frequency and total points. Bonuses after victory do not
recover preparation opportunities deliberately ignored beforehand. A larger
collection/training run has therefore not been launched automatically on the
basis of this comparison.

Raw results: `benchmark_runs/reward-comparison-win-first.json`.
Training and old checkpoints/corpora were preserved. The implementation was
verified with 523 passing tests and four optional tests skipped.
