"""Shared transformer training for MCTS visits and ISS action demonstrations."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from skat.agents.distill.checkpoint import load_policy, save_policy
from skat.agents.distill.demonstrations import Demonstrations
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


def target_baselines(targets, masks):
    probabilities = targets.double()
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(1)
    return dict(
        target_entropy=float(entropy.mean()),
        uniform_loss=float(masks.sum(1).double().log().mean()),
    )


def save_training_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def train(directory, output, *, config=None, seed=1, device="cpu"):
    config = config or TrainConfig()
    seed_everything(seed)
    torch.set_num_threads(config.threads)
    device = resolve_device(device)
    demonstrations = None
    if (Path(directory) / "manifest.json").exists():
        demonstrations = Demonstrations(directory)
        baselines = demonstrations.manifest["target_baselines"]
        train_rows = range(demonstrations.manifest["rows"]["train"])
        val_rows = range(demonstrations.manifest["rows"]["val"])
    else:
        samples = load_shards(directory)
        train_rows, val_rows = split_by_group(samples.groups, seed)
        states = torch.from_numpy(samples.states).float()
        masks = torch.from_numpy(samples.masks).bool()
        targets = visit_targets(torch.from_numpy(samples.visits), masks)
        choices = torch.from_numpy(samples.choices).long()
        baselines = {
            phase: target_baselines(targets[rows], masks[rows])
            for phase, rows in (("train", train_rows), ("val", val_rows))
        }

        def dataset(rows):
            return TensorDataset(
                states[rows], masks[rows], targets[rows], choices[rows]
            )

        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            dataset(train_rows), config.batch_size, shuffle=True, generator=generator
        )
        val_loader = DataLoader(dataset(val_rows), config.batch_size)

    def batches(phase, epoch=0):
        if demonstrations is None:
            yield from train_loader if phase == "train" else val_loader
        else:
            for batch in demonstrations.batches(
                phase,
                config.batch_size,
                seed=seed + epoch if phase == "train" else None,
            ):
                x = torch.from_numpy(batch["states"])
                mask = torch.from_numpy(batch["masks"])
                choice = torch.from_numpy(batch["choices"])
                target = torch.nn.functional.one_hot(choice, 32).float()
                yield x, mask, target, choice, batch["solo"], batch["kinds"]

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
    best_epoch = None
    corpus_hash = hashlib.sha256()
    if demonstrations is None:
        for file in sorted(Path(directory).glob("*.npz")):
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    corpus_hash.update(block)
    corpus_sha256 = demonstrations.sha256 if demonstrations else corpus_hash.hexdigest()
    metadata = dict(
        seed=seed,
        training=asdict(config),
        corpus_sha256=corpus_sha256,
        target_source="iss-action-v1" if demonstrations else "mcts-visits",
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        device=str(device),
        torch_version=str(torch.__version__),
        target_baselines=baselines,
    )
    collection_path = Path(directory) / "collection.json"
    if collection_path.exists():
        metadata["collection"] = json.loads(collection_path.read_text())
    if demonstrations:
        metadata["iss"] = demonstrations.manifest["provenance"]
        metadata["test_rows"] = demonstrations.manifest["rows"]["test"]
    report_path = Path(output).with_suffix(".training.json")
    print(f"Target baselines: {baselines}", flush=True)

    def run_phase(phase, epoch, model):
        model.train(phase == "train")
        total = correct = count = 0
        breakdown = {}
        with torch.set_grad_enabled(phase == "train"):
            for batch in tqdm(
                batches(phase, epoch),
                desc=f"{phase} {epoch + 1}/{config.epochs}",
                unit="batch",
                dynamic_ncols=True,
            ):
                x, mask, target, choice = (v.to(device) for v in batch[:4])
                logits = model(x)
                loss = policy_loss(logits, target, mask)
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                predictions = logits.masked_fill(~mask, -torch.inf).argmax(-1)
                agreement = predictions == choice
                n = len(x)
                total += loss.item() * n
                correct += int(agreement.sum())
                count += n
                if demonstrations:
                    # Diagnostics only: identities/outcomes never enter the model.
                    losses = (
                        -(
                            target
                            * torch.log_softmax(
                                logits.masked_fill(~mask, -torch.inf), -1
                            ).masked_fill(~mask, 0)
                        )
                        .sum(1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    hits = agreement.detach().cpu().numpy()
                    for label, selection in (
                        ("soloist", batch[4]),
                        ("defence", ~batch[4]),
                        *(
                            (kind, batch[5] == i)
                            for i, kind in enumerate(("suit", "grand", "null"))
                        ),
                    ):
                        item = breakdown.setdefault(label, [0.0, 0, 0])
                        item[0] += float(losses[selection].sum())
                        item[1] += int(hits[selection].sum())
                        item[2] += int(selection.sum())
        stats = {
            f"loss_{phase}": total / count,
            f"agreement_{phase}": correct / count,
            f"kl_{phase}": total / count - baselines[phase]["target_entropy"],
        }
        if demonstrations:
            stats[f"breakdown_{phase}"] = {
                label: dict(
                    loss=values[0] / values[2],
                    agreement=values[1] / values[2],
                    rows=values[2],
                )
                for label, values in breakdown.items()
                if values[2]
            }
        return stats

    for epoch in range(config.epochs):
        stats = {**run_phase("train", epoch, model), **run_phase("val", epoch, model)}
        history.append(stats)
        if stats["loss_val"] < best:
            best = stats["loss_val"]
            best_epoch = epoch + 1
            save_policy(
                output, model, {**metadata, "best_epoch": epoch + 1, "metrics": stats}
            )
        print(f"epoch {epoch + 1}/{config.epochs}: {stats}", flush=True)
        report = {
            **metadata,
            "best_val_loss": best,
            "best_epoch": best_epoch,
            "completed_epochs": epoch + 1,
            "history": history,
            "report_path": str(report_path),
        }
        save_training_report(report_path, report)
    if demonstrations:
        # The held-out temporal test is evaluated once using the selected model.
        best_model, checkpoint_metadata = load_policy(output, device)
        test = run_phase("test", best_epoch - 1, best_model)
        report["test"] = test
        save_policy(output, best_model, {**checkpoint_metadata, "test": test})
        save_training_report(report_path, report)
    return report
