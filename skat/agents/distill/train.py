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
from skat.models.policy import PolicyNet

DEFAULT_OUT = "./trained_models/policy.pt"


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
    device = resolve_device(args.device)
    print(f"distill device: {describe_device(device)}")

    samples = load_shards(args.data, input_size=PolicyNet.INPUT_SIZE)
    targets = build_targets(samples, args.target, args.target_temperature)
    train_idx, val_idx = split_by_group(samples.groups, args.val_fraction, args.seed)
    print(
        f"{len(samples)} decisions from "
        f"{len(torch.unique(samples.groups))} rounds: "
        f"{len(train_idx)} train / {len(val_idx)} validation"
    )

    net = PolicyNet(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
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
                meta={
                    "samples": len(samples),
                    "rounds": len(torch.unique(samples.groups)),
                    "epoch": epoch,
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
    print(f"best validation loss {best:.4f}, checkpoint at {args.out}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Distil an MCTS corpus into a policy network."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Directory of .npz shards."
    )
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
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
    args = parser.parse_args()
    if args.target_temperature is None:
        args.target_temperature = (
            DEFAULT_VISIT_TEMPERATURE
            if args.target == "visits"
            else DEFAULT_VALUE_TEMPERATURE
        )
    return args


if __name__ == "__main__":
    train(get_args())
