"""Train and calibrate a conditional pickup-declaration win model."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from skat.agents.distill.train import save_training_report
from skat.agents.neural_bidding.train import calibrate, probability_metrics
from skat.agents.neural_declaring.checkpoint import save_declaration
from skat.agents.neural_declaring.features import FEATURE_SCHEMA, INPUT_SIZE
from skat.agents.neural_declaring.samples import FORMAT, SPLITS
from skat.config import TrainConfig
from skat.models.declaring import DeclarationNet
from skat.rules import RULES_SCHEMA
from skat.utils.reproducibility import resolve_device, seed_everything


def load_corpus(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())

    def sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if (
        manifest.get("format") != FORMAT
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("input_size") != INPUT_SIZE
        or manifest.get("rules_schema") != RULES_SCHEMA
    ):
        raise ValueError("incompatible declaration corpus")
    arrays = {}
    for split in SPLITS:
        state_path = directory / f"{split}-states.npy"
        label_path = directory / f"{split}-labels.npy"
        if (
            sha256(state_path) != manifest["checksums"][split]["states"]
            or sha256(label_path) != manifest["checksums"][split]["labels"]
        ):
            raise ValueError("declaration corpus checksum mismatch")
        states = np.load(state_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if (
            states.shape != (manifest["rows"][split], INPUT_SIZE)
            or labels.shape != (manifest["rows"][split],)
            or states.dtype != np.float32
            or labels.dtype != np.float32
            or not np.isin(labels, (0, 1)).all()
        ):
            raise ValueError("invalid declaration corpus arrays")
        arrays[split] = (states, labels)
    return manifest, arrays


def train(directory, output, *, config=None, seed=7, device="cpu"):
    config = config or TrainConfig(
        epochs=20, batch_size=512, d_model=64, nhead=4, num_layers=2
    )
    seed_everything(seed)
    torch.set_num_threads(config.threads)
    device = resolve_device(device)
    manifest, arrays = load_corpus(directory)
    model = DeclarationNet(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    report = dict(
        seed=seed,
        training=asdict(config),
        corpus_sha256=hashlib.sha256(
            (Path(directory) / "manifest.json").read_bytes()
        ).hexdigest(),
        policy_sha256=manifest["policy_sha256"],
        rows=manifest["rows"],
        device=str(device),
        history=[],
    )
    output = Path(output)
    report_path = output.with_suffix(".training.json")
    best = float("inf")

    def run(split, *, epoch=0, optimize=False):
        states, labels = arrays[split]
        indices = np.arange(len(labels))
        if optimize:
            np.random.default_rng(seed + epoch).shuffle(indices)
        model.train(optimize)
        total = count = 0
        logits_all = []
        with torch.set_grad_enabled(optimize):
            for start in range(0, len(indices), config.batch_size):
                batch = indices[start : start + config.batch_size]
                x = torch.from_numpy(np.asarray(states[batch])).to(device)
                y = torch.from_numpy(np.asarray(labels[batch])).to(device)
                logits = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                if optimize:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total += float(loss.detach()) * len(batch)
                count += len(batch)
                if not optimize:
                    logits_all.append(logits.detach().cpu().numpy())
        if optimize:
            return total / count
        return total / count, np.concatenate(logits_all), np.array(labels, copy=True)

    for epoch in range(config.epochs):
        loss_train = run("train", epoch=epoch, optimize=True)
        loss_val, _, _ = run("val")
        print(
            f"declaration epoch {epoch + 1}/{config.epochs}: "
            f"train={loss_train:.5f} val={loss_val:.5f}",
            flush=True,
        )
        report["history"].append(dict(loss_train=loss_train, loss_val=loss_val))
        if loss_val < best:
            best = loss_val
            report["best_epoch"] = epoch + 1
            save_declaration(output, model, metadata=report)
        report["completed_epochs"] = epoch + 1
        save_training_report(report_path, report)
    from skat.agents.neural_declaring.checkpoint import load_declaration

    best_model, _, _ = load_declaration(output, device)
    model = best_model
    _, calib_logits, calib_labels = run("calibration")
    temperature = calibrate(calib_logits, calib_labels)
    _, test_logits, test_labels = run("test")
    report["temperature"] = temperature
    report["calibration"] = probability_metrics(calib_logits, calib_labels, temperature)
    report["test"] = probability_metrics(test_logits, test_labels, temperature)
    save_declaration(output, model, temperature=temperature, metadata=report)
    save_training_report(report_path, report)
    return report
