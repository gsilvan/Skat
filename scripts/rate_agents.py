"""Refresh the README ratings: 1,000 rounds, seed 42, two conservative random seats."""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from skat.__main__ import get_player
from skat.agents.neural_bidding.inference import NeuralBidder
from skat.agents.search.parallel import SearchPool
from skat.config import MCTSConfig
from skat.game import Tournament

START = "<!-- agent-ratings:start -->"
END = "<!-- agent-ratings:end -->"
ROUNDS = 1000
SEED = 42
BIDDER = "benchmark_runs/bidding-run-1/bidding.pt"
PROFILES = (
    ("random", "Random (Baseline)", "random", None, 600, "conservative"),
    ("random-neural", "Random", "random", None, 600, "neural"),
    (
        "mcts-600",
        "MCTS · 600 Simulationen / 20 Welten",
        "mcts",
        None,
        600,
        "conservative",
    ),
    (
        "mcts-2000",
        "MCTS · 2.000 Simulationen / 20 Welten",
        "mcts",
        None,
        2000,
        "conservative",
    ),
    (
        "mcts-iss-belief",
        "MCTS · ISS-Handmodell 100k / 600 Simulationen / 20 Welten",
        "mcts",
        "trained_models/iss-belief-100k.pt",
        600,
        "conservative",
    ),
    (
        "mcts-iss-belief-300k",
        "MCTS · ISS-Handmodell 300k / 600 Simulationen / 20 Welten",
        "mcts",
        "trained_models/iss-belief-300k.pt",
        600,
        "conservative",
    ),
    (
        "policy-mcts-12",
        "Transformer · MCTS 5k / 12 Epochen",
        "policy",
        "trained_models/transformer-run-2.pt",
        600,
        "conservative",
    ),
    (
        "policy-mcts-100",
        "Transformer · MCTS 5k / 100 Epochen",
        "policy",
        "trained_models/transformer-run-2-100epochs.pt",
        600,
        "conservative",
    ),
    (
        "policy-teacher-2000",
        "Transformer · MCTS 2.000 / 10k / 100 Epochen",
        "policy",
        "trained_models/transformer-teacher-2000-10k.pt",
        600,
        "conservative",
    ),
    (
        "policy-iss",
        "Transformer · ISS 100k / 20 Epochen",
        "policy",
        "benchmark_runs/iss-run-1/policy.pt",
        600,
        "conservative",
    ),
    (
        "policy-iss-neural",
        "Transformer · ISS 100k / 20 Epochen",
        "policy",
        "benchmark_runs/iss-run-1/policy.pt",
        600,
        "neural",
    ),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def fraction(wins, played):
    return f"{wins}/{played} ({wins / played:.1%})" if played else "—"


def rating_table(rows):
    baseline = rows[0]["score"]["game_points"] / ROUNDS
    lines = [
        "| Agent / Kartenmodell | Reizer | Siege / gespielt | Als Alleinspieler | Im Gegenspiel | Eingepasst | Punkte / Runde | Δ Punkte zur Baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: r["score"]["game_points"], reverse=True):
        score = row["score"]
        played = score["solo_rounds"] + score["defence_rounds"]
        wins = score["solo_wins"] + score["defence_wins"]
        points = score["game_points"] / ROUNDS
        lines.append(
            f"| {row['label']} | {row['bidder']} | {fraction(wins, played)} | "
            f"{fraction(score['solo_wins'], score['solo_rounds'])} | "
            f"{fraction(score['defence_wins'], score['defence_rounds'])} | "
            f"{row['passed']} | {points:+.2f} | {points - baseline:+.2f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--out", required=True, type=Path, help="New directory for full round records"
    )
    parser.add_argument("--report", type=Path, default=Path("docs/agent-ratings.json"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    readme = args.readme.read_text()
    if readme.count(START) != 1 or readme.count(END) != 1:
        parser.error("README needs exactly one agent-ratings marker pair")
    # Check every required checkpoint before starting the benchmark.
    for path in {BIDDER, *(p[3] for p in PROFILES if p[3])}:
        if not Path(path).is_file():
            parser.error(f"missing checkpoint: {path}")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = hashlib.sha256()
    for path in sorted(Path("skat").rglob("*.py")):
        source.update(path.as_posix().encode())
        source.update(path.read_bytes())
    report = dict(
        protocol="play-seat0-random-conservative-v1",
        measured_at=datetime.now(timezone.utc).isoformat(),
        rounds=ROUNDS,
        seed=SEED,
        candidate_seat=0,
        dealer_rotation=True,
        reference_agents=["random", "random"],
        reference_bidding="conservative",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=source.hexdigest(),
        device=args.device,
        torch_version=str(torch.__version__),
        profiles=[],
        status="running",
    )
    for index, (key, label, name, policy, simulations, bidding) in enumerate(PROFILES):
        print(f"[{index + 1}/{len(PROFILES)}] {label}, {bidding}", flush=True)
        config = MCTSConfig(simulations=simulations, determinizations=20)
        bidder = (
            NeuralBidder(BIDDER, device=args.device, threshold=0.55)
            if bidding == "neural"
            else None
        )
        with SearchPool(args.workers, capacity=config.determinizations) as pool:
            belief = policy if key.startswith("mcts-iss-belief") else None
            candidate = get_player(
                name,
                seed=SEED,
                policy=None if belief else policy,
                belief_model=belief,
                device=args.device,
                mcts=config,
                search_pool=pool,
                bidding=bidding,
                bidder=bidder,
            )
            agents = [
                candidate,
                get_player("random", seed=SEED + 1),
                get_player("random", seed=SEED + 2),
            ]
            tournament = Tournament(ROUNDS, agents, seed=SEED, progress=True)
            started = time.perf_counter()
            tournament.start()
            elapsed = time.perf_counter() - started
            row = dict(
                id=key,
                label=label,
                agent=name,
                bidder=candidate.bidding_label,
                policy=None if belief else policy,
                policy_sha256=sha256(policy) if policy and not belief else None,
                belief_model=belief,
                belief_model_sha256=sha256(belief) if belief else None,
                bidding_model=BIDDER if bidder else None,
                bidding_model_sha256=sha256(BIDDER) if bidder else None,
                bidding_threshold=0.55 if bidder else None,
                mcts=asdict(config) if name == "mcts" else None,
                search_workers=pool.workers if name == "mcts" else None,
                score=asdict(tournament.seat_scores[0]),
                passed=tournament.passed,
                seconds=elapsed,
            )
            atomic_json(
                args.out / f"{key}.json",
                dict(
                    profile=row,
                    rounds=tournament.round_records,
                    seats=[asdict(s) for s in tournament.seat_scores],
                ),
            )
        report["profiles"].append(row)
        atomic_json(args.out / "summary.json", report)
        print(tournament.summary(), flush=True)
    report["status"] = "complete"
    atomic_json(args.out / "summary.json", report)
    atomic_json(args.report, report)
    prefix, tail = args.readme.read_text().split(START, 1)
    _, suffix = tail.split(END, 1)
    args.readme.write_text(
        prefix + START + "\n" + rating_table(report["profiles"]) + "\n" + END + suffix
    )
    print(f"Ratings written to {args.readme} and {args.report}", flush=True)


if __name__ == "__main__":
    main()
