# Exact bonus tie-breaking in MCTS

The initial win-first implementation is committed as `19fb644`. This variant
keeps its primary search value, and adds an independently accumulated bonus.
Root moves are ordered by `(mean_primary_value, mean_bonus)`, without rounding
or an epsilon. During UCT, bonuses only break exact UCB score ties. Bidding's
outcome-only searches ignore bonuses completely. DQN is unchanged by this
experiment and retains its separate bonus gate after secured victory.

Before secured victory, primary values are binary party outcomes. After secured
victory, primary values already maximise the winning party's bonus. Bonuses
reward points, Schneider and Schwarz in won outcomes; they do not penalise
missing Schneider. The existing engine thresholds remain unchanged.

## Comparison on common deals

The two reference rows reuse the prior comparison's exact raw outcomes. The new
row uses the same seed `3096181844`, 200 deals per seat/panel, 600 simulations
over 20 determinizations, and the original transformer opponent policy, with
CPU inference and one Torch thread: 1,200 outcomes per variant.

| Metric | Previous shaped reward | Secured-victory bonus only | Exact bonus tie-break |
| --- | ---: | ---: | ---: |
| Party wins | 75.00% | 75.42% | 75.25% |
| Against random | 85.67% | 84.83% | 86.17% |
| Against policy | 64.33% | 66.00% | 64.33% |
| Party Schneider wins | 332 | 259 | 326 |
| Party Schwarz wins (engine definition) | 11 | 7 | 11 |
| Party card points per round | 74.39 | 72.60 | 74.17 |

The tie-break variant recovers most of the point and Schneider reduction.
There is no established win-rate difference: the paired, deal-cluster bootstrap
95% interval is -2.08 to +2.58 percentage points against the previous shaped
reward, and -2.50 to +2.17 against the secured-victory-only bonus. This is an
exploratory comparison on deals already used to assess the previous variant,
not an independent statistical confirmation. Equal sampled win rates do not
guarantee equal real win probabilities.

The variant is selected as the teacher because it matches the requested
priority and retains bonus outcomes in this comparison, not because this run
establishes superior playing strength.

## Distillation contract

New v2 shards store independent `bonuses` alongside primary `values`. The
reader still accepts v1 shards, and rejects mixtures of reward objectives.
Value-target training only boosts bonuses among exactly tied primary maxima,
so lower primary values cannot become the target's preferred move. Exact
comparison uses float64 division. Policy checkpoints retain the teacher's
`win-first-exact-tie-bonus-v1` objective.

To repeat just the third row with the existing baseline file:

```bash
OMP_NUM_THREADS=1 python -m skat.benchmark.reward_tiebreak \
  --baseline benchmark_runs/reward-comparison-win-first.json \
  --opponent-policy benchmark_runs/20261007T162312Z-2f71a71-seed20260829/checkpoints/policy-transformer.pt \
  --out benchmark_runs/reward-comparison-tiebreak-repeat.json \
  --device cpu
```

Raw results: `benchmark_runs/reward-comparison-tiebreak.json`.
