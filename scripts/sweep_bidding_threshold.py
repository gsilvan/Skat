"""Compare conservative bidding with neural thresholds on identical deals."""

import argparse
import csv
import hashlib
import importlib.util
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
from skat.config import load_config
from skat.evaluation import paired_points_comparison, wilson
from skat.game import Tournament

THRESHOLDS = tuple(i / 20 for i in range(21))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def metrics(score, total):
    """Party wins exclude passes; expected points and solo frequency include them."""
    played = score.solo_rounds + score.defence_rounds
    wins = score.solo_wins + score.defence_wins
    return dict(
        **asdict(score),
        played=played,
        wins=wins,
        losses=played - wins,
        solo_losses=score.solo_rounds - score.solo_wins,
        defence_losses=score.defence_rounds - score.defence_wins,
        passed=total - played,
        win_rate=wins / played if played else None,
        win_interval=wilson(wins, played),
        solo_win_rate=score.solo_wins / score.solo_rounds
        if score.solo_rounds
        else None,
        defence_win_rate=score.defence_wins / score.defence_rounds
        if score.defence_rounds
        else None,
        points_per_round=score.game_points / total,
        solo_fraction=score.solo_rounds / total,
        passed_fraction=(total - played) / total,
    )


def point_outcomes(run, seat):
    return [
        dict(
            panel="fixed-table",
            seat=seat,
            seed=row["seed"],
            official_points=row["result"]["score"]
            if row["result"]["soloist"] == seat
            else 0,
        )
        for row in run["rounds"]
    ]


def play(args, config, pool, threshold):
    # Fresh agent state and identical RNG seeds for every baseline/sweep run.
    bidder = (
        NeuralBidder(args.bidding_model, device=args.device, threshold=threshold)
        if threshold is not None
        else None
    )
    agents = [
        get_player(
            name,
            seed=args.seed + seat,
            policy=getattr(args, f"policy{seat}") or args.policy,
            device=args.device,
            mcts=config,
            search_pool=pool,
            bidding="neural"
            if bidder is not None and seat == args.bidding_seat
            else "conservative",
            bidder=bidder if seat == args.bidding_seat else None,
        )
        for seat, name in enumerate(args.agents)
    ]
    tournament = Tournament(args.rounds, agents, seed=args.seed, progress=True)
    started = time.perf_counter()
    try:
        tournament.start()
    finally:
        # Keep completed rounds even when interrupted halfway through a threshold.
        completed = len(tournament.round_records)
        run = dict(
            threshold=threshold,
            completed_rounds=completed,
            status="complete" if completed == args.rounds else "interrupted",
            bidders=[agent.bidding_label for agent in agents],
            seats=[metrics(score, completed) for score in tournament.seat_scores]
            if completed
            else [],
            seconds=time.perf_counter() - started,
            rounds=tournament.round_records,
        )
        name = "baseline" if threshold is None else f"threshold-{threshold:.2f}"
        write_json(args.out / f"{name}.json", run)
    print(tournament.summary(), flush=True)
    return run


def plot(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seat = report["manifest"]["bidding_seat"]
    baseline = report["baseline"]["seats"][seat]
    runs = report["sweep"]
    xs = [r["threshold"] for r in runs]
    scores = [r["seats"][seat] for r in runs]
    best = report["best_threshold"]
    most_wins = max(runs, key=lambda r: (r["seats"][seat]["wins"], -r["threshold"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    ax = axes[0]
    ax.plot(xs, [s["game_points"] for s in scores], "o-", label="Neural-Reizer")
    ax.axhline(
        baseline["game_points"],
        color="black",
        linestyle="--",
        label=f"Conservative: {baseline['game_points']:+d} Punkte",
    )
    ax.set(ylabel="Gesamtpunkte", title=f"Sitz {seat}: Gesamtpunkte nach dem Lauf")
    ax.axvline(
        best, color="grey", linestyle=":", label=f"Höchste Punkte bei {best:.2f}"
    )
    ax.legend()
    ax = axes[1]
    for key, label, color in (
        ("wins", "Siege", "C2"),
        ("losses", "Niederlagen", "C3"),
        ("passed", "Eingepasst", "C7"),
    ):
        values = [
            s["played"] - s["wins"] if key == "losses" else s[key] for s in scores
        ]
        ax.plot(xs, values, "o-", color=color, label=label)
        base = (
            baseline["played"] - baseline["wins"] if key == "losses" else baseline[key]
        )
        ax.axhline(
            base, color=color, linestyle=":", label=f"Conservative: {label} {base}"
        )
    ax.set(
        ylabel="Anzahl Runden",
        title=f"Sitz {seat}: Siege und Niederlagen",
        ylim=(0, report["manifest"]["rounds"]),
    )
    ax.axvline(
        most_wins["threshold"],
        color="grey",
        linestyle=":",
        label=f"Meiste Siege bei {most_wins['threshold']:.2f}",
    )
    ax.legend(fontsize=9)
    for ax in axes:
        ax.set(xlabel="Neural-Threshold", xlim=(0, 1))
        ax.grid(alpha=0.2)
    manifest = report["manifest"]
    lineup = " / ".join(
        row["agent"] + (f" ({Path(row['policy']).name})" if row["policy"] else "")
        for row in manifest["seats"]
    )
    fig.suptitle(
        f"{lineup}\n{manifest['rounds']} Runden je Wert · Seed {manifest['seed']} · "
        f"Neural nur Sitz {seat} · Siege zählen Alleinspiel und Gegenspiel"
    )
    fig.savefig(directory / "threshold-sweep.png", dpi=180)
    fig.savefig(directory / "threshold-sweep.pdf")
    plt.close(fig)


def write_csv(report, directory):
    seat = report["manifest"]["bidding_seat"]
    keys = (
        "game_points",
        "wins",
        "losses",
        "win_rate",
        "solo_win_rate",
        "defence_win_rate",
        "solo_rounds",
        "played",
        "passed",
        "overbids",
    )
    with (directory / "threshold-sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("bidder", "threshold", *keys))
        writer.writeheader()
        for run in [report["baseline"], *report["sweep"]]:
            score = dict(run["seats"][seat])
            score.setdefault("losses", score["played"] - score["wins"])
            writer.writerow(
                dict(
                    bidder="conservative" if run["threshold"] is None else "neural",
                    threshold=run["threshold"],
                    **{k: score[k] for k in keys},
                )
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        nargs=3,
        choices=("random", "mcts", "policy"),
        default=["random"] * 3,
    )
    parser.add_argument("--policy", help="Shared card-policy checkpoint")
    for seat in range(3):
        parser.add_argument(
            f"--policy{seat}", help=f"Card policy for seat {seat}; overrides --policy"
        )
    parser.add_argument("--bidding-model", required=True, type=Path)
    parser.add_argument("--bidding-seat", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--config", help="MCTS TOML configuration")
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("rounds must be positive")
    config, _ = load_config(args.config)
    policies = [
        getattr(args, f"policy{s}") or args.policy if name == "policy" else None
        for s, name in enumerate(args.agents)
    ]
    paths = {args.bidding_model, *(Path(p) for p in policies if p)}
    for seat, name in enumerate(args.agents):
        if name == "policy" and not policies[seat]:
            parser.error(
                f"policy agent at seat {seat} needs --policy or --policy{seat}"
            )
    for path in paths:
        if not path.is_file():
            parser.error(f"missing checkpoint: {path}")
    if importlib.util.find_spec("matplotlib") is None:
        parser.error(
            "matplotlib is required; use the system Python with Matplotlib and Torch"
        )
    torch.set_num_threads(1)
    # Validate checkpoint schemas and devices before reserving the results path.
    NeuralBidder(args.bidding_model, device=args.device)
    for path in set(p for p in policies if p):
        get_player("policy", policy=path, device=args.device)
    with SearchPool(args.workers, capacity=config.determinizations) as pool:
        args.out.mkdir(parents=True, exist_ok=False)
        source = hashlib.sha256()
        for path in sorted(Path("skat").rglob("*.py")):
            source.update(path.as_posix().encode())
            source.update(path.read_bytes())
        report = dict(
            status="running",
            manifest=dict(
                measured_at=datetime.now(timezone.utc).isoformat(),
                rounds=args.rounds,
                seed=args.seed,
                bidding_seat=args.bidding_seat,
                thresholds=THRESHOLDS,
                dealer_rotation=True,
                objective="own official total points after a fixed number of dealt rounds",
                seats=[
                    dict(
                        agent=name,
                        policy=policies[s],
                        policy_sha256=hashlib.sha256(
                            Path(policies[s]).read_bytes()
                        ).hexdigest()
                        if policies[s]
                        else None,
                    )
                    for s, name in enumerate(args.agents)
                ],
                bidding_model=str(args.bidding_model),
                bidding_model_sha256=hashlib.sha256(
                    args.bidding_model.read_bytes()
                ).hexdigest(),
                mcts=asdict(config),
                workers=pool.workers,
                device=args.device,
                torch_version=str(torch.__version__),
                source_sha256=source.hexdigest(),
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            ),
            sweep=[],
        )
        write_json(args.out / "summary.json", report)
        try:
            print("[1/22] Baseline: conservative an allen Sitzen", flush=True)
            baseline = play(args, config, pool, None)
            report["baseline"] = {k: v for k, v in baseline.items() if k != "rounds"}
            write_json(args.out / "summary.json", report)
            for index, threshold in enumerate(THRESHOLDS, start=2):
                print(
                    f"[{index}/22] Sitz {args.bidding_seat}: neural ({threshold:.2f})",
                    flush=True,
                )
                run = play(args, config, pool, threshold)
                run["paired_points"] = paired_points_comparison(
                    point_outcomes(baseline, args.bidding_seat),
                    point_outcomes(run, args.bidding_seat),
                    args.seed,
                )
                report["sweep"].append({k: v for k, v in run.items() if k != "rounds"})
                write_json(args.out / "summary.json", report)
            best = max(
                report["sweep"],
                key=lambda r: (
                    r["seats"][args.bidding_seat]["game_points"],
                    -r["threshold"],
                ),
            )
            report["best_threshold"] = best["threshold"]
            report["best_points_threshold"] = best["threshold"]
            report["best_wins_threshold"] = max(
                report["sweep"],
                key=lambda r: (r["seats"][args.bidding_seat]["wins"], -r["threshold"]),
            )["threshold"]
            write_csv(report, args.out)
            plot(report, args.out)
            report["status"] = "complete"
            print(
                f"Bester gemessener Threshold: {best['threshold']:.2f}; "
                f"{best['seats'][args.bidding_seat]['game_points']:+d} Gesamtpunkte",
                flush=True,
            )
        except BaseException as error:
            report["status"], report["error"] = "failed", repr(error)
            raise
        finally:
            write_json(args.out / "summary.json", report)


if __name__ == "__main__":
    main()
