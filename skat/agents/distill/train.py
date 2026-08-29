"""Train a `PolicyNet` on what `RecordingMCTSAgent` wrote down.

    python -m skat.agents.distill.train --data data/distill \\
        --out trained_models/policy.pt
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator

import torch
from torch.utils.tensorboard import SummaryWriter

from skat.agents.distill.checkpoint import save_policy
from skat.agents.distill.loss import policy_loss
from skat.agents.distill.samples import Samples, load_shards
from skat.agents.distill.targets import (
    DEFAULT_VALUE_TEMPERATURE,
    DEFAULT_VISIT_TEMPERATURE,
    value_targets,
    visit_targets,
)
from skat.agents.rl.dqn import describe_device, resolve_device
from skat.models import ARCHITECTURES
from skat.models.policy import PolicyNet
from skat.utils.reproducibility import seed_everything

DEFAULT_OUT = "./trained_models/policy.pt"

#: Per-architecture training defaults. Every knob listed here has
#: `default=None` in argparse, so this table is the only place a default is
#: written down. The `--epsilon-decay` bug was exactly the other shape: two
#: defaults for one knob, one of them winning silently.
ARCH_DEFAULTS: dict[str, dict] = {
    "mlp": {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 512,
        "epochs": 30,
        "schedule": "none",
        "warmup": 0,
    },
    "transformer": {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "dropout": 0.1,
        "batch_size": 256,
        "epochs": 40,
        "schedule": "warmup-cosine",
        "warmup": 500,
    },
}


def resolve(args):
    """
    Fill in whatever the command line left unset, and say what was filled.

    Called from `get_args` and again at the top of `train`, so a hand-built
    `Namespace` (which is what the tests pass) does not have to enumerate every
    knob that will ever exist.
    """
    filled = {}
    for key, value in ARCH_DEFAULTS[args.arch].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
            filled[key] = value
    if args.target_temperature is None:
        args.target_temperature = (
            DEFAULT_VISIT_TEMPERATURE
            if args.target == "visits"
            else DEFAULT_VALUE_TEMPERATURE
        )
    if filled:
        # The run says which defaults it used, so a runs/ directory is
        # self-describing months later.
        print(
            f"{args.arch} defaults: "
            + " ".join(f"{k}={v}" for k, v in sorted(filled.items()))
        )
    return args


def warmup_cosine(total_steps: int, warmup: int):
    """Linear warm-up, then cosine to zero. `LambdaLR`-shaped."""

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def parameter_groups(net, weight_decay: float):
    """
    Decay the weights and nothing else.

    `LayerNorm` gains and biases and `Embedding` rows are not the kind of
    parameter weight decay is for, and at 0.01 shrinking them does visible
    harm. One-dimensional parameters are exactly that set here.
    """
    decayed = [p for p in net.parameters() if p.requires_grad and p.dim() > 1]
    plain = [p for p in net.parameters() if p.requires_grad and p.dim() <= 1]
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def split_by_group(
    groups: torch.Tensor, val_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cut train from validation on round boundaries.

    The ~22 decisions of one round share a hand, a soloist and a trick history.
    Split row-wise and the validation set is mostly positions whose siblings
    were trained on, which flatters the curve by several points of agreement
    and hides exactly the overfitting the split exists to detect.
    """
    ids = torch.unique(groups)
    shuffled = ids[torch.randperm(len(ids), generator=_generator(seed))]
    held_out = max(1, round(len(ids) * val_fraction))
    validation = set(shuffled[:held_out].tolist())
    is_val = torch.tensor([int(g) in validation for g in groups.tolist()])
    return (~is_val).nonzero().flatten(), is_val.nonzero().flatten()


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def batches(
    index: torch.Tensor, batch_size: int, generator: torch.Generator
) -> Iterator[torch.Tensor]:
    """Shuffle `index` and hand it out in batches."""
    order = index[torch.randperm(len(index), generator=generator)]
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def build_targets(samples: Samples, kind: str, temperature: float) -> torch.Tensor:
    if kind == "visits":
        return visit_targets(samples.visits, samples.masks, temperature)
    return value_targets(samples.values, samples.visits, samples.masks, temperature)


def agreement(
    logits: torch.Tensor, masks: torch.Tensor, choices: torch.Tensor
) -> float:
    """Share of positions where the net's best legal card is the teacher's."""
    picked = logits.masked_fill(~masks, torch.finfo(logits.dtype).min).argmax(dim=-1)
    return (picked == choices).float().mean().item()


def evaluate(
    net: PolicyNet,
    samples: Samples,
    targets: torch.Tensor,
    index: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    net.eval()
    total_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = index[start : start + batch_size]
            masks = samples.masks[rows].to(device)
            logits = net(samples.states[rows].to(device))
            total_loss += policy_loss(
                logits, targets[rows].to(device), masks
            ).item() * len(rows)
            correct += agreement(logits, masks, samples.choices[rows].to(device)) * len(
                rows
            )
    return total_loss / len(index), correct / len(index)


def train(args) -> None:
    resolve(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"distill device: {describe_device(device)}")

    architecture = ARCHITECTURES[args.arch]
    samples = load_shards(args.data, input_size=architecture.INPUT_SIZE)
    targets = build_targets(samples, args.target, args.target_temperature)
    train_idx, val_idx = split_by_group(samples.groups, args.val_fraction, args.seed)
    print(
        f"{len(samples)} decisions from "
        f"{len(torch.unique(samples.groups))} rounds: "
        f"{len(train_idx)} train / {len(val_idx)} validation"
    )

    arch_kwargs = {"features": args.features} if args.arch == "transformer" else {}
    net = architecture(dropout=args.dropout, **arch_kwargs).to(device)
    print(f"{args.arch}: {sum(p.numel() for p in net.parameters())} parameters")
    if args.arch == "mlp":
        # Left byte-identical to how the measured baseline was trained. The
        # parameter-group split below is a real change to what gets decayed,
        # and the MLP's number is the one this is all being compared against.
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            parameter_groups(net, args.weight_decay), lr=args.lr
        )
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    scheduler = None
    if args.schedule == "warmup-cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine(args.epochs * steps_per_epoch, args.warmup)
        )
    writer = SummaryWriter(log_dir=getattr(args, "log_dir", None))
    generator = _generator(args.seed)

    step = 0
    best = math.inf
    for epoch in range(args.epochs):
        net.train()
        for rows in batches(train_idx, args.batch_size, generator):
            masks = samples.masks[rows].to(device)
            logits = net(samples.states[rows].to(device))
            loss = policy_loss(logits, targets[rows].to(device), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            writer.add_scalar("Loss/train", loss.item(), step)
            step += 1

        train_loss, train_top1 = evaluate(
            net, samples, targets, train_idx, args.batch_size, device
        )
        val_loss, val_top1 = evaluate(
            net, samples, targets, val_idx, args.batch_size, device
        )
        writer.add_scalar("Loss/epoch", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Agreement/train", train_top1, epoch)
        writer.add_scalar("Agreement/val", val_top1, epoch)
        marker = ""
        if val_loss < best:
            # Best validation loss rather than the last epoch: three million
            # parameters against a corpus this size will start memorising, and
            # the epoch that memorises best is not the one that plays best.
            best = val_loss
            save_policy(
                args.out,
                net,
                arch=args.arch,
                arch_kwargs=arch_kwargs,
                meta={
                    "samples": len(samples),
                    "rounds": len(torch.unique(samples.groups)),
                    "epoch": epoch,
                    "arch": args.arch,
                    "target": args.target,
                    "target_temperature": args.target_temperature,
                    "val_loss": val_loss,
                    "val_agreement": val_top1,
                },
            )
            marker = " *"
        print(
            f"epoch {epoch:3d}  train {train_loss:.4f} ({train_top1:.1%})  "
            f"val {val_loss:.4f} ({val_top1:.1%}){marker}"
        )

    writer.flush()
    writer.close()
    print(f"best validation loss {best:.4f}, checkpoint at {args.out}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Distil an MCTS corpus into a policy network."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Directory of .npz shards."
    )
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument(
        "--arch",
        type=str,
        choices=sorted(ARCHITECTURES),
        default="mlp",
        help="mlp is the dense baseline; transformer is self-attention over "
        "the 32 cards, with roughly a fifth of the parameters.",
    )
    parser.add_argument(
        "--features",
        type=str,
        choices=("full", "raw"),
        default="full",
        help="Transformer only. `raw` drops the derived per-card features "
        "(trump, trick order, led suit) and is the control that separates "
        "them from the architecture itself.",
    )
    # These default to None so `ARCH_DEFAULTS` stays the one place a default is
    # written down - see `resolve`.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--schedule", type=str, choices=("none", "warmup-cosine"), default=None
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument(
        "--target",
        type=str,
        choices=("visits", "value"),
        default="visits",
        help="What the teacher's opinion is read as: how often the search "
        "looked at a move, or how well it scored. `value` is averaged over "
        "every determinization and is the less noisy of the two when the "
        "trees are shallow.",
    )
    parser.add_argument("--target-temperature", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard run directory (default: a fresh one under runs/).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return resolve(parser.parse_args())


if __name__ == "__main__":
    train(get_args())
