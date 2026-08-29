"""Train the neural play/pass gate on counterfactual MCTS outcomes."""

from __future__ import annotations

import argparse
import math

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from skat.agents.distill.train import _generator, batches, split_by_group
from skat.agents.neural_bidding.checkpoint import save_bidding
from skat.agents.neural_bidding.samples import BiddingSamples, load_bidding_shards
from skat.agents.rl.dqn import describe_device, resolve_device
from skat.models.bidding import BiddingNet
from skat.utils.reproducibility import seed_everything

DEFAULT_OUT = "./trained_models/bidding.pt"


def evaluate(
    net: BiddingNet,
    samples: BiddingSamples,
    index: torch.Tensor,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> tuple[float, float, float]:
    net.eval()
    loss = correct = brier = 0.0
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = index[start : start + batch_size]
            labels = samples.labels[rows].to(device)
            logits = net(samples.states[rows].to(device))
            probabilities = torch.sigmoid(logits)
            loss += F.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            ).item()
            correct += ((probabilities >= threshold) == labels.bool()).sum().item()
            brier += ((probabilities - labels) ** 2).sum().item()
    size = max(1, len(index))
    return loss / size, correct / size, brier / size


def train(args: argparse.Namespace) -> dict[str, float | int]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"bidding device: {describe_device(device)}")
    samples = load_bidding_shards(args.data)
    groups = torch.unique(samples.groups)
    if len(groups) < 2:
        raise ValueError("bidding training needs at least two deals for validation")
    train_idx, val_idx = split_by_group(samples.groups, args.val_fraction, args.seed)
    print(
        f"{len(samples)} hands from {len(groups)} deals: "
        f"{len(train_idx)} train / {len(val_idx)} validation, "
        f"{samples.labels.mean().item():.1%} wins"
    )

    net = BiddingNet(hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    writer = SummaryWriter(log_dir=args.log_dir)
    generator = _generator(args.seed)
    best = math.inf
    best_metrics = (math.inf, 0.0, math.inf)
    step = 0
    for epoch in range(args.epochs):
        net.train()
        for rows in batches(train_idx, args.batch_size, generator):
            labels = samples.labels[rows].to(device)
            loss = F.binary_cross_entropy_with_logits(
                net(samples.states[rows].to(device)), labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), step)
            step += 1
        val_loss, val_accuracy, val_brier = evaluate(
            net, samples, val_idx, args.batch_size, device, args.threshold
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("Calibration/brier", val_brier, epoch)
        marker = ""
        if val_loss < best:
            best = val_loss
            best_metrics = (val_loss, val_accuracy, val_brier)
            save_bidding(
                args.out,
                net,
                meta={
                    "samples": len(samples),
                    "deals": len(groups),
                    "epoch": epoch,
                    "seed": args.seed,
                    "threshold": args.threshold,
                    "win_rate": samples.labels.mean().item(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_brier": val_brier,
                    "reward": "win=+1, loss=-1",
                },
            )
            marker = " *"
        print(
            f"epoch {epoch:3d}  val {val_loss:.4f} "
            f"({val_accuracy:.1%}, brier {val_brier:.4f}){marker}"
        )
    writer.flush()
    writer.close()
    print(f"best validation loss {best:.4f}, checkpoint at {args.out}")
    return {
        "samples": len(samples),
        "deals": len(groups),
        "val_loss": best_metrics[0],
        "val_accuracy": best_metrics[1],
        "val_brier": best_metrics[2],
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural MCTS-derived play/pass bidding gate."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
