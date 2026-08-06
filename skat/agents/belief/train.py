"""Train a `BeliefNet` on where the hidden cards actually were.

    python -m skat.agents.belief.train --data data/belief \\
        --out trained_models/belief.pt

The number that matters is not the loss but the margin over `capacity_prior` -
what uniform determinization already believes. A belief net that merely matches
it would make `BeliefMCTSAgent` a slower `MCTSAgent`.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch.utils.tensorboard import SummaryWriter

from skat.agents.belief.checkpoint import save_belief
from skat.agents.belief.loss import (
    accuracy_of,
    belief_loss,
    belief_probabilities,
    capacity_prior,
    cross_entropy_of,
)
from skat.agents.belief.samples import BeliefSamples, load_belief_shards
from skat.agents.distill.train import _generator, batches, split_by_group, warmup_cosine
from skat.agents.rl.dqn import describe_device, resolve_device
from skat.models.belief import BeliefNet

DEFAULT_OUT = "./trained_models/belief.pt"


def baseline(samples: BeliefSamples, index: torch.Tensor) -> tuple[float, float]:
    """What the uniform dealer scores on the same rows."""
    locations = samples.locations[index]
    prior = capacity_prior(locations)
    return (
        cross_entropy_of(prior, locations).item(),
        accuracy_of(prior, locations).item(),
    )


def evaluate(
    net: BeliefNet,
    samples: BeliefSamples,
    index: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    net.eval()
    total_loss, correct, cards = 0.0, 0.0, 0
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = index[start : start + batch_size]
            locations = samples.locations[rows].to(device)
            logits = net(samples.states[rows].to(device))
            n = int((locations >= 0).sum())
            total_loss += belief_loss(logits, locations).item() * n
            correct += accuracy_of(belief_probabilities(logits), locations).item() * n
            cards += n
    return total_loss / max(1, cards), correct / max(1, cards)


def train(args) -> None:
    device = resolve_device(args.device)
    print(f"belief device: {describe_device(device)}")

    samples = load_belief_shards(args.data, input_size=BeliefNet.INPUT_SIZE)
    train_idx, val_idx = split_by_group(samples.groups, args.val_fraction, args.seed)
    print(
        f"{len(samples)} observations from "
        f"{len(torch.unique(samples.groups))} rounds: "
        f"{len(train_idx)} train / {len(val_idx)} validation"
    )
    prior_loss, prior_accuracy = baseline(samples, val_idx)
    print(
        f"uniform determinization scores {prior_loss:.4f} / "
        f"{prior_accuracy:.1%} on the validation rounds - the bar to clear"
    )

    net = BeliefNet(dropout=args.dropout).to(device)
    print(f"belief net: {sum(p.numel() for p in net.parameters())} parameters")
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
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
            locations = samples.locations[rows].to(device)
            loss = belief_loss(net(samples.states[rows].to(device)), locations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar("Loss/train", loss.item(), step)
            step += 1

        val_loss, val_accuracy = evaluate(
            net, samples, val_idx, args.batch_size, device
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("Accuracy/prior", prior_accuracy, epoch)
        marker = ""
        if val_loss < best:
            best = val_loss
            save_belief(
                args.out,
                net,
                meta={
                    "observations": len(samples),
                    "rounds": int(len(torch.unique(samples.groups))),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "prior_loss": prior_loss,
                    "prior_accuracy": prior_accuracy,
                },
            )
            marker = " *"
        print(
            f"epoch {epoch:3d}  val {val_loss:.4f} ({val_accuracy:.1%})  "
            f"vs prior {prior_loss:.4f} ({prior_accuracy:.1%}){marker}"
        )

    writer.flush()
    print(f"best validation loss {best:.4f}, checkpoint at {args.out}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a hand-prediction network for determinized search."
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
