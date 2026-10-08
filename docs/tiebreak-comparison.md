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

## Larger policy corpus and independent confirmation

With teacher commit `5a3efba`, eight collectors played 650 rounds each at
600 simulations / 20 determinizations, with seeds `20261018` through
`20261025`. The resulting eight shards contain 110,156 decisions from
4,936 played, recorded rounds. Group-aware splitting gives 104,623 training
and 5,533 validation examples. Each worker produced one shard, keeping a
round's decisions together across the split.

The full-feature value-target transformer was trained for 40 epochs on the
RX 5600 XT / ROCm 7.2 with seed `394376469`. The best validation checkpoint
is epoch 29: loss 1.2299468 and teacher agreement 57.4914%. Collection took
10 minutes; training took 20.5 minutes. The checkpoint is
`trained_models/policy-transformer-value-win-first-tiebreak.pt`.

```bash
python -m skat.agents.distill.train \
  --data benchmark_runs/win-first-tiebreak-training/data \
  --arch transformer --features full --target value \
  --epochs 40 --seed 394376469 --device cuda \
  --out trained_models/policy-transformer-value-win-first-repeat.pt \
  --log-dir runs/policy-transformer-value-win-first-repeat
```

On the original 200 evaluation deals (seed `3030215365`), the new model wins
74.25% of 1,200 seat/panel games versus 72.58% for the old 120-epoch policy.
The paired deal-cluster bootstrap interval for the advantage is -0.75 to
+4.08 percentage points: promising, but not conclusive on that sample.

A single independent confirmation used 1,000 fresh deals, seed `291503412`
(`stage_seed(20261008, "policy-win-first-independent-confirmation")`). Both
policies were reevaluated on the same deals, three seats and the two original
opponent panels, with CUDA inference, Torch 2.14.0 and one Torch thread.
Neither model was retrained or selected using these confirmation outcomes.

| Metric | Old value policy, 120 epochs | New value policy, larger corpus |
| --- | ---: | ---: |
| Party wins | 4,338 / 6,000 (72.30%) | 4,556 / 6,000 (75.93%) |
| Against random | 82.93% | 86.00% |
| Against original transformer policy | 61.67% | 65.87% |
| As soloist | 80.72% | 84.12% |
| On defence | 67.41% | 71.15% |
| Party Schneider wins | 1,281 | 1,329 |
| Party Schwarz wins (engine definition) | 25 | 62 |
| Party card points per round | 72.13 | 73.72 |

The paired advantage is **+3.63 percentage points**, with a 95% deal-cluster
bootstrap interval of **+2.65 to +4.67** points (10,000 resamples, seed 42).
There are 593 improved and 375 regressed seat/panel outcomes. The new
checkpoint is preferred for these opponent populations. This compares the
whole training recipe: more data and a changed teacher objective, not an
isolated causal test of the reward change. A distilled policy approximates
the teacher; its neural predictions do not guarantee exact win-first choices.
No new DQN checkpoint was trained in this experiment.

Raw outcomes, checkpoint hashes and training provenance are in
`benchmark_runs/win-first-tiebreak-training/{manifest,policy-evaluation,policy-confirmation}.json`.
The two-panel evaluation uses `skat.benchmark.evaluation.evaluate_candidate`;
the bootstrap uses `skat.benchmark.reward_tiebreak.paired_comparison`.
