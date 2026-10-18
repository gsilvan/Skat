"""Train and calibrate paired pickup/Hand win estimates."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from iss.pipeline import sha256
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.distill.train import save_training_report
from skat.agents.neural_bidding.train import calibrate, probability_metrics
from skat.agents.neural_pickup.checkpoint import load_pickup, save_pickup
from skat.agents.neural_pickup.features import FEATURE_SCHEMA, INPUT_SIZE
from skat.agents.neural_pickup.samples import FORMAT, SPLITS
from skat.config import TrainConfig
from skat.hand import FULL_HAND
from skat.models.pickup import PickupNet
from skat.rules import RULES_SCHEMA
from skat.utils.reproducibility import resolve_device, seed_everything


def load_corpus(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("input_size") != INPUT_SIZE
        or manifest.get("rules_schema") != RULES_SCHEMA
    ):
        raise ValueError("incompatible pickup corpus")
    arrays = {}
    for split in SPLITS:
        state_path = directory / f"{split}-states.npy"
        label_path = directory / f"{split}-labels.npy"
        if (
            sha256(state_path) != manifest["checksums"][split]["states"]
            or sha256(label_path) != manifest["checksums"][split]["labels"]
        ):
            raise ValueError("pickup corpus checksum mismatch")
        states = np.load(state_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if (
            states.shape != (manifest["rows"][split], INPUT_SIZE)
            or labels.shape != (manifest["rows"][split], 2)
            or states.dtype != np.float32
            or labels.dtype != np.float32
            or not np.isin(labels, (0, 1)).all()
        ):
            raise ValueError("invalid pickup corpus arrays")
        arrays[split] = (states, labels)
    return manifest, arrays


def decision_metrics(states, labels, probabilities, margin=0.0):
    def safe_hand(state):
        cards = {FULL_HAND[i] for i in np.flatnonzero(state[:32])}
        game = HeuristicDeclarer().choose_hand(SimpleNamespace(hand=cards))
        return game.spielwert_range(cards)[0] >= round(float(state[35]) * 500)

    safe = np.fromiter(
        (safe_hand(state) for state in states),
        dtype=bool,
        count=len(states),
    )
    pick_hand = safe & (probabilities[:, 1] > probabilities[:, 0] + margin)
    selected = np.where(pick_hand, labels[:, 1], labels[:, 0])
    return dict(
        deals=len(labels),
        hand_decisions=int(pick_hand.sum()),
        wins=int(selected.sum()),
        pickup_baseline_wins=int(labels[:, 0].sum()),
        gained_wins=int(((selected == 1) & (labels[:, 0] == 0)).sum()),
        lost_wins=int(((selected == 0) & (labels[:, 0] == 1)).sum()),
        hand_baseline_wins=int(labels[:, 1].sum()),
        paired_oracle_wins=int(labels.max(axis=1).sum()),
    )


def train(directory, output, *, config=None, seed=7, device="cpu"):
    config = config or TrainConfig(
        epochs=20, batch_size=512, d_model=64, nhead=4, num_layers=2
    )
    seed_everything(seed)
    torch.set_num_threads(config.threads)
    device = resolve_device(device)
    manifest, arrays = load_corpus(directory)
    model = PickupNet(
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
        declaration_model_sha256=manifest["declaration_model_sha256"],
        rows=manifest["rows"],
        device=str(device),
        history=[],
    )
    output = Path(output)
    report_path = output.with_suffix(".training.json")
    best = float("inf")

    def run(split, *, epoch=0, optimize=False, net=model):
        states, labels = arrays[split]
        indices = np.arange(len(labels))
        if optimize:
            np.random.default_rng(seed + epoch).shuffle(indices)
        net.train(optimize)
        total = count = 0
        logits_all = []
        with torch.set_grad_enabled(optimize):
            for start in range(0, len(indices), config.batch_size):
                batch = indices[start : start + config.batch_size]
                x = torch.from_numpy(np.asarray(states[batch])).to(device)
                y = torch.from_numpy(np.asarray(labels[batch])).to(device)
                logits = net(x)
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
            f"pickup epoch {epoch + 1}/{config.epochs}: "
            f"train={loss_train:.5f} val={loss_val:.5f}",
            flush=True,
        )
        report["history"].append(dict(loss_train=loss_train, loss_val=loss_val))
        if loss_val < best:
            best = loss_val
            report["best_epoch"] = epoch + 1
            save_pickup(output, model, metadata=report)
        report["completed_epochs"] = epoch + 1
        save_training_report(report_path, report)
    selected, _, _ = load_pickup(output, device)
    _, calibration_logits, calibration_labels = run("calibration", net=selected)
    temperatures = tuple(
        calibrate(calibration_logits[:, i], calibration_labels[:, i]) for i in range(2)
    )
    _, test_logits, test_labels = run("test", net=selected)
    test_probabilities = 1 / (1 + np.exp(-test_logits / temperatures))
    report["temperatures"] = temperatures
    report["calibration"] = {
        name: probability_metrics(
            calibration_logits[:, i], calibration_labels[:, i], temperatures[i]
        )
        for i, name in enumerate(("pickup", "hand"))
    }
    report["test"] = {
        name: probability_metrics(test_logits[:, i], test_labels[:, i], temperatures[i])
        for i, name in enumerate(("pickup", "hand"))
    }
    report["test_decision"] = decision_metrics(
        arrays["test"][0], test_labels, test_probabilities
    )
    save_pickup(output, selected, temperatures=temperatures, metadata=report)
    save_training_report(report_path, report)
    return report
