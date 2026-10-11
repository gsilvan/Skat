"""Single-architecture, visit-target policy training with a round-aware split."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from skat.agents.distill.checkpoint import save_policy
from skat.agents.distill.loss import policy_loss
from skat.agents.distill.samples import load_shards
from skat.agents.distill.targets import visit_targets
from skat.config import TrainConfig
from skat.models.card_transformer import CardTransformer
from skat.utils.reproducibility import resolve_device, seed_everything


def split_by_group(groups, seed=1):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("training requires at least two complete-round groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * 0.2))
    is_val = np.isin(groups, unique[:n_val])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def train(directory, output, *, config=None, seed=1, device="cpu"):
    config = config or TrainConfig()
    seed_everything(seed)
    torch.set_num_threads(config.threads)
    device = resolve_device(device)
    samples = load_shards(directory)
    train_rows, val_rows = split_by_group(samples.groups, seed)
    states = torch.from_numpy(samples.states).float()
    masks = torch.from_numpy(samples.masks).bool()
    targets = visit_targets(torch.from_numpy(samples.visits), masks)
    choices = torch.from_numpy(samples.choices).long()

    def dataset(rows):
        return TensorDataset(states[rows], masks[rows], targets[rows], choices[rows])

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset(train_rows), config.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(dataset(val_rows), config.batch_size)
    model = CardTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=2 * config.d_model,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []
    best = float("inf")
    corpus_hash = hashlib.sha256()
    for file in sorted(Path(directory).glob("*.npz")):
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                corpus_hash.update(block)
    metadata = dict(
        seed=seed,
        training=asdict(config),
        corpus_sha256=corpus_hash.hexdigest(),
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        device=str(device),
        torch_version=str(torch.__version__),
    )
    for epoch in range(config.epochs):
        stats = {}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            total = correct = count = 0
            with torch.set_grad_enabled(phase == "train"):
                for batch in loader:
                    x, mask, target, choice = (v.to(device) for v in batch)
                    logits = model(x)
                    loss = policy_loss(logits, target, mask)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    n = len(x)
                    total += loss.item() * n
                    correct += int(
                        (
                            logits.masked_fill(~mask, -torch.inf).argmax(-1) == choice
                        ).sum()
                    )
                    count += n
            stats[f"loss_{phase}"] = total / count
            stats[f"agreement_{phase}"] = correct / count
        history.append(stats)
        if stats["loss_val"] < best:
            best = stats["loss_val"]
            save_policy(
                output, model, {**metadata, "best_epoch": epoch + 1, "metrics": stats}
            )
        print(f"epoch {epoch + 1}/{config.epochs}: {stats}", flush=True)
    return {**metadata, "best_val_loss": best, "history": history}
