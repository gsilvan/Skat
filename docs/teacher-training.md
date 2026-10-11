# Training with more MCTS search

`configs/teacher-2000.toml` uses 2,000 total simulations across 20 sampled
worlds, giving each tree 100 simulations instead of the baseline's 30. It also
sets training to 100 epochs, which improved the preceding policy run. Model
architecture, targets, rules and observation layout stay compatible with the
existing reference checkpoints. More search is a candidate for a stronger
teacher, not a guarantee of higher playing strength.

The following commands use the system Python so training can use its installed
CUDA/ROCm Torch build. Collection uses CPU processes and three MCTS seats.
Use fresh corpus and evaluation paths for each run.

## Check the teacher before the larger collection

```bash
python -m skat evaluate \
  --config configs/teacher-2000.toml --agents mcts \
  --policy trained_models/transformer-run-2.pt \
  --reference-policy trained_models/transformer-run-2.pt \
  --deals 1000 --seed 42 --device cpu \
  --out benchmark_runs/teacher-2000-eval.json
```

The policy checkpoint fixes the reference seats; the candidate here is MCTS.
Compare its panel results on the same deals with the 600-simulation MCTS row
in `benchmark_runs/transformer-run-2-eval.json`. Both use the 12-epoch reference
checkpoint. If more search does not help, reassess the teacher before scaling
collection. This is a standalone benchmark, not a test-suite training run.

## Collection, training, evaluation, play

```bash
# 1. Ten thousand rounds of MCTS/MCTS/MCTS, all available CPU workers.
python -m skat collect \
  --config configs/teacher-2000.toml \
  --rounds 10000 --seed 7 --out data/teacher-2000-10k

# 2. Train a transformer for 100 epochs on the GPU.
python -m skat train \
  --config configs/teacher-2000.toml \
  --data data/teacher-2000-10k \
  --out trained_models/transformer-teacher-2000-10k.pt \
  --device cuda --seed 7

# 3. Common deals and all seats, with the old 12-epoch model as fixed reference.
#    The MCTS candidate uses the same budget as the collection teacher.
python -m skat evaluate \
  --config configs/teacher-2000.toml \
  --agents random mcts policy \
  --policy trained_models/transformer-teacher-2000-10k.pt \
  --reference-policy trained_models/transformer-run-2.pt \
  --deals 1000 --seed 42 --device cpu \
  --out benchmark_runs/transformer-teacher-2000-10k-eval.json

# 4. Put the new policy, the previous best policy and the new teacher at a table.
python -m skat play \
  --config configs/teacher-2000.toml \
  --agents policy policy mcts \
  --policy0 trained_models/transformer-teacher-2000-10k.pt \
  --policy1 trained_models/transformer-run-2-100epochs.pt \
  --rounds 1000 --seed 42 --device cpu
```

The fourth command uses fixed seats and is useful for watching tournament
scores. The seat-rotated third command is the controlled strength comparison.
Random and policy reference populations match the preceding runs. The teacher
has a larger search budget, so its own change must be distinguished from a
change in student performance.

## Training diagnostics

Every completed epoch atomically updates a `.training.json` file next to the
checkpoint. It records the complete history, best epoch, settings, corpus hash
and collection metadata where available. The console summary no longer repeats
the full history after the last epoch. A stopped run leaves the history of its
completed epochs; it does not provide optimizer state for resuming.

For each split, the trainer reports the uniform legal-card cross-entropy and
the empirical target entropy. `kl_train` and `kl_val` subtract that entropy
from cross-entropy, measuring the average KL divergence from the teacher target
to the model prediction. Smaller values indicate closer distribution matching;
they do not directly measure game-win rates. The best checkpoint is still
selected by validation cross-entropy, equivalent to validation KL on a fixed
split. Existing checkpoints remain loadable.

A single run that increases teacher budget and corpus size measures the
combined improvement, not the isolated contribution of each change. Keep the
old corpus and checkpoints for separate comparisons if attribution matters.
