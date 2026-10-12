"""Historical win classifier; calibration never uses the test partition."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from skat.agents.distill.train import save_training_report
from skat.agents.neural_bidding.checkpoint import load_bidding, save_bidding
from skat.agents.neural_bidding.samples import BiddingCorpus
from skat.config import TrainConfig
from skat.models.bidding import BiddingNet
from skat.utils.reproducibility import resolve_device, seed_everything


def probability_metrics(logits, labels, temperature=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float64) / temperature
    labels = torch.as_tensor(labels, dtype=torch.float64)
    probabilities = logits.sigmoid()
    bins = []
    for index in range(10):
        selected = (probabilities >= index / 10) & (probabilities < (index + 1) / 10)
        if index == 9:
            selected |= probabilities == 1
        n = int(selected.sum())
        bins.append(
            dict(
                lower=index / 10,
                upper=(index + 1) / 10,
                rows=n,
                predicted=float(probabilities[selected].mean()) if n else None,
                observed=float(labels[selected].mean()) if n else None,
            )
        )
    return dict(
        rows=len(labels),
        win_rate=float(labels.mean()),
        log_loss=float(
            torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        ),
        brier=float(((probabilities - labels) ** 2).mean()),
        bins=bins,
    )


def calibrate(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.5, max_iter=40, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / log_temperature.clamp(-4, 4).exp(), labels
        )
        loss.backward()
        return loss

    before = float(closure().detach())
    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-4, 4).exp())
    after = probability_metrics(logits, labels, temperature)["log_loss"]
    return temperature if after <= before else 1.0


def train(directory, output, *, config=None, seed=7, device="cpu"):
    config = config or TrainConfig(
        epochs=20, batch_size=512, d_model=64, nhead=4, num_layers=2
    )
    seed_everything(seed)
    torch.set_num_threads(config.threads)
    device = resolve_device(device)
    corpus = BiddingCorpus(directory)
    model = BiddingNet(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    metadata = dict(
        seed=seed,
        training=asdict(config),
        corpus_sha256=corpus.sha256,
        provenance=corpus.manifest["provenance"],
        rows=corpus.manifest["rows"],
        device=str(device),
        torch_version=str(torch.__version__),
        applicability="historical ordinary pickup point games; not tied to a card policy",
    )
    report = {**metadata, "history": []}
    report_path = Path(output).with_suffix(".training.json")
    best = float("inf")

    def run(split, epoch=0, net=model, collect=False):
        net.train(split == "train")
        total = count = 0
        logits_all, labels_all = [], []
        batches = corpus.batches(
            split, config.batch_size, seed + epoch if split == "train" else None
        )
        with torch.set_grad_enabled(split == "train"):
            for states, labels in tqdm(
                batches,
                desc=f"bidding {split} {epoch + 1}",
                unit="batch",
                dynamic_ncols=True,
            ):
                x, y = (
                    torch.from_numpy(states).to(device),
                    torch.from_numpy(labels).to(device),
                )
                logits = net(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total += float(loss.detach()) * len(y)
                count += len(y)
                if collect:
                    logits_all.append(logits.detach().cpu().numpy())
                    labels_all.append(labels)
        return (
            (np.concatenate(logits_all), np.concatenate(labels_all))
            if collect
            else total / count
        )

    for epoch in range(config.epochs):
        stats = dict(loss_train=run("train", epoch), loss_val=run("val", epoch))
        report["history"].append(stats)
        report["completed_epochs"] = epoch + 1
        if stats["loss_val"] < best:
            best = stats["loss_val"]
            report["best_epoch"] = epoch + 1
            save_bidding(
                output,
                model,
                metadata={**metadata, "best_epoch": epoch + 1, "metrics": stats},
            )
        report["best_val_loss"] = best
        save_training_report(report_path, report)
        print(f"bidding epoch {epoch + 1}/{config.epochs}: {stats}", flush=True)
    selected, _, checkpoint_meta = load_bidding(output, device)
    calibration_logits, calibration_labels = run(
        "calibration", net=selected, collect=True
    )
    temperature = calibrate(calibration_logits, calibration_labels)
    test_logits, test_labels = run("test", net=selected, collect=True)
    report["temperature"] = temperature
    report["calibration"] = probability_metrics(
        calibration_logits, calibration_labels, temperature
    )
    report["test_raw"] = probability_metrics(test_logits, test_labels)
    report["test"] = probability_metrics(test_logits, test_labels, temperature)
    prior = corpus.manifest["wins"].get("train", 0) / corpus.manifest["rows"]["train"]
    prior = np.clip(prior, 1e-7, 1 - 1e-7)
    report["test_constant_prior"] = probability_metrics(
        np.full(len(test_labels), np.log(prior / (1 - prior))), test_labels
    )
    save_bidding(
        output,
        selected,
        temperature=temperature,
        metadata={
            **checkpoint_meta,
            "calibration": report["calibration"],
            "test": report["test"],
        },
    )
    save_training_report(report_path, report)
    return report
