"""Train and calibrate ISS hidden-hand predictions on bounded shards."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from iss.belief import BeliefCorpus
from skat.agents.belief.checkpoint import load_belief, save_belief
from skat.agents.belief.features import KNOWN, OBSERVATION_SIZE
from skat.agents.belief.model import BeliefTransformer, masked_log_probs
from skat.agents.search.determinize import SKAT_SLOT, WeightedDealer
from skat.hand import FULL_HAND
from skat.utils.reproducibility import resolve_device, seed_everything


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def card_loss(model, states, targets):
    logits = model(states)
    log_probs = masked_log_probs(logits, states)
    count = int((targets != KNOWN).sum())
    loss = F.nll_loss(
        log_probs.reshape(-1, 3),
        targets.reshape(-1),
        ignore_index=KNOWN,
        reduction="sum",
    )
    return loss, count


def phase_metrics(model, corpus, split, batch_size, device):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for batch in corpus.batches(split, batch_size):
            states = torch.from_numpy(batch["states"]).to(device)
            targets = torch.from_numpy(batch["locations"].astype(np.int64)).to(device)
            loss, n = card_loss(model, states, targets)
            total += float(loss)
            count += n
    return total / count


def test_breakdown(model, corpus, batch_size, device):
    """Held-out card NLL by party, contract and stage, weighted by hidden cards."""
    groups = {}
    model.eval()
    with torch.no_grad():
        for batch in corpus.batches("test", batch_size):
            states = torch.from_numpy(batch["states"]).to(device)
            targets = torch.from_numpy(batch["locations"].astype(np.int64)).to(device)
            log_probs = masked_log_probs(model(states), states)
            unknown = targets != KNOWN
            losses = -log_probs.gather(2, targets.clamp(min=0).unsqueeze(-1)).squeeze(
                -1
            )
            losses = (losses * unknown).sum(1).cpu().numpy()
            counts = unknown.sum(1).cpu().numpy()
            for state, loss, count in zip(batch["states"], losses, counts):
                role = "soloist" if state[38] else "defence"
                kind = "null" if state[37] else "grand" if state[36] else "suit"
                played = int(
                    (state[OBSERVATION_SIZE + 64 : OBSERVATION_SIZE + 96] > 0).sum()
                )
                stage = "early" if played < 10 else "middle" if played < 20 else "late"
                for label in (role, kind, stage):
                    item = groups.setdefault(label, [0.0, 0, 0])
                    item[0] += float(loss)
                    item[1] += int(count)
                    item[2] += 1
    return {
        label: dict(
            card_nll=values[0] / values[1], hidden_cards=values[1], positions=values[2]
        )
        for label, values in groups.items()
    }


def world_log_probability(logits, state, labels, strength):
    """True complete-world likelihood under the exact constrained sampler."""
    unseen = [FULL_HAND[i] for i, location in enumerate(labels) if location != KNOWN]
    slots = (0, 1, SKAT_SLOT)
    capacities = {
        slot: int((labels == location).sum()) for location, slot in enumerate(slots)
    }
    voids = {
        slot: frozenset(
            FULL_HAND[i]
            for i in range(32)
            if state[OBSERVATION_SIZE + location * 32 + i]
        )
        for location, slot in enumerate(slots[:2])
    }
    centered = logits - logits.max(axis=1, keepdims=True)
    weights = np.clip(np.exp(strength * centered), 1e-4, 1.0)
    weights = {
        i: {slot: float(weights[i, location]) for location, slot in enumerate(slots)}
        for i in range(32)
    }
    truth = {
        slot: [FULL_HAND[i] for i in range(32) if labels[i] == location]
        for location, slot in enumerate(slots)
    }
    return WeightedDealer(unseen, capacities, voids, weights).log_probability(truth)


def calibration_rows(corpus, split, limit=1000, seed=7):
    """Deterministic bounded sample from a fixed split, independent of training."""
    rng = np.random.default_rng(seed)
    reservoir = []
    seen = 0
    for batch in corpus.batches(split, 256):
        for state, labels in zip(batch["states"], batch["locations"]):
            seen += 1
            row = (state.copy(), labels.copy())
            if len(reservoir) < limit:
                reservoir.append(row)
            else:
                index = rng.integers(seen)
                if index < limit:
                    reservoir[index] = row
    return reservoir


def world_scores(model, rows, device, strengths):
    model.eval()
    totals = {strength: 0.0 for strength in strengths}
    with torch.no_grad():
        for state, labels in tqdm(rows, desc="belief calibration", leave=False):
            logits = model(torch.from_numpy(state).unsqueeze(0).to(device))[0]
            logits = logits.cpu().numpy()
            for strength in strengths:
                totals[strength] -= world_log_probability(
                    logits, state, labels, strength
                )
    return {str(strength): total / len(rows) for strength, total in totals.items()}


def train(
    directory,
    output,
    *,
    epochs=12,
    batch_size=256,
    seed=7,
    device="cpu",
    d_model=128,
    num_layers=4,
    calibration_size=1000,
):
    if any(
        type(x) is not int or x < 1
        for x in (epochs, batch_size, d_model, num_layers, calibration_size)
    ):
        raise ValueError("belief training sizes must be positive integers")
    seed_everything(seed)
    torch.set_num_threads(1)
    device = resolve_device(device)
    corpus = BeliefCorpus(directory)
    model = BeliefTransformer(
        d_model=d_model, num_layers=num_layers, nhead=4, dim_feedforward=2 * d_model
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    output = Path(output)
    report_path = output.with_suffix(".training.json")
    history = []
    best = math.inf
    metadata = dict(
        corpus_sha256=corpus.sha256,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=str(device),
    )
    for epoch in range(epochs):
        model.train()
        total = count = 0
        for batch in tqdm(
            corpus.batches("train", batch_size, seed=seed + epoch),
            desc=f"belief epoch {epoch + 1}/{epochs}",
        ):
            states = torch.from_numpy(batch["states"]).to(device)
            targets = torch.from_numpy(batch["locations"].astype(np.int64)).to(device)
            loss, n = card_loss(model, states, targets)
            optimizer.zero_grad(set_to_none=True)
            (loss / n).backward()
            optimizer.step()
            total += float(loss.detach())
            count += n
        val = phase_metrics(model, corpus, "val", batch_size, device)
        row = dict(
            epoch=epoch + 1,
            train_nll=total / count,
            val_nll=val,
            train_kl=total / count,
            val_kl=val,
        )
        history.append(row)
        if val < best:
            best = val
            save_belief(
                output,
                model,
                {
                    **metadata,
                    "best_epoch": epoch + 1,
                    "val_nll": val,
                    "sampling_strength": 1.0,
                },
            )
        report = {
            **metadata,
            "best_val_nll": best,
            "history": history,
            "completed_epochs": epoch + 1,
        }
        save_report(report_path, report)
        print(row, flush=True)
    model, checkpoint_meta = load_belief(output, device)
    strengths = (0.0, 0.5, 1.0, 1.5, 2.0)
    calibration = world_scores(
        model,
        calibration_rows(corpus, "val", calibration_size, seed),
        device,
        strengths,
    )
    chosen = min(strengths, key=lambda strength: (calibration[str(strength)], strength))
    checkpoint_meta.update(sampling_strength=chosen, calibration_world_nll=calibration)
    save_belief(output, model, checkpoint_meta)
    test_rows = calibration_rows(corpus, "test", calibration_size, seed)
    report.update(
        sampling_strength=chosen,
        calibration_world_nll=calibration,
        test_world_nll=world_scores(model, test_rows, device, (0.0, chosen)),
        test_card_nll=phase_metrics(model, corpus, "test", batch_size, device),
        test_breakdown=test_breakdown(model, corpus, batch_size, device),
    )
    save_report(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--calibration-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    train(
        args.data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        calibration_size=args.calibration_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
