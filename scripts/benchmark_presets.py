"""Paired, seat-balanced strategy benchmark; separate from README Random ratings.

Every candidate plays the same deals at all three seats against each fixed
reference pair. Whole deals, including all seats, form the bootstrap units.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random

import numpy as np

from skat.config import MCTSConfig
from skat.evaluation import summarize
from skat.game import Round
from skat.models.observation import OBSERVATION_SCHEMA
from skat.rules import RULES_SCHEMA
from skat.utils.processes import available_cpus, worker_environment
from skat.utils.reproducibility import resolve_device, stage_seed

PROFILES = {
    "best": dict(
        agent="policy", bidding="neural", declaration="heuristic", pickup="always"
    ),
    "declaration": dict(
        agent="policy", bidding="neural", declaration="neural", pickup="always"
    ),
    "pickup": dict(
        agent="policy", bidding="neural", declaration="heuristic", pickup="neural"
    ),
    "all-neural": dict(
        agent="policy", bidding="neural", declaration="neural", pickup="neural"
    ),
    "random": dict(
        agent="random", bidding="conservative", declaration="heuristic", pickup="always"
    ),
    "mcts": dict(agent="mcts", bidding="neural", declaration="neural", pickup="neural"),
    "mcts-default": dict(
        agent="mcts", bidding="conservative", declaration="heuristic", pickup="always"
    ),
}
MODELS = dict(
    policy="benchmark_runs/iss-run-1/policy.pt",
    bidding="benchmark_runs/bidding-run-1/bidding.pt",
    declaration="benchmark_runs/declaration-run-3/declaration.pt",
    pickup="benchmark_runs/pickup-run-2/pickup.pt",
)
PROTOCOL = "paired-presets-fixed-panels-v1"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_models(config):
    actual = {name: digest(path) for name, path in config["models"].items()}
    if actual != config["checkpoint_sha256"]:
        raise ValueError("benchmark checkpoints changed; keep reference weights frozen")


def agent_factory(config):
    """One set of immutable scorers and separate mutable agents per worker/seat."""
    import torch

    from skat.__main__ import get_player
    from skat.agents.neural_bidding.inference import NeuralBidder
    from skat.agents.neural_declaring.inference import DeclarationScorer
    from skat.agents.neural_pickup.inference import PickupScorer

    torch.set_num_threads(1)
    models = config["models"]
    bidder = NeuralBidder(
        models["bidding"],
        device=config["device"],
        threshold=config["bidding_threshold"],
    )
    declarer = DeclarationScorer(models["declaration"], device=config["device"])
    pickup = PickupScorer(models["pickup"], device=config["device"])
    agents = {}

    def get(profile, seat):
        key = (profile, seat)
        if key not in agents:
            spec = config["profiles"][profile]
            agents[key] = get_player(
                spec["agent"],
                policy=models["policy"],
                device=config["device"],
                bidding=spec["bidding"],
                bidder=bidder,
                declaration=spec["declaration"],
                declaration_model=declarer,
                pickup=spec["pickup"],
                pickup_model=pickup,
                pickup_margin=config["pickup_margin"],
                mcts=MCTSConfig(**config["mcts"]),
            )
        return agents[key]

    return get


def play_deal(config, seed, index, get_agent):
    """All paired games for one independent deal; RNG never crosses deals."""
    deal_seed = stage_seed(seed, f"preset-benchmark-deal-{index}")
    rows = []
    for panel in config["panels"]:
        for candidate in config["candidates"]:
            for seat in range(3):
                lineup = [candidate if s == seat else panel for s in range(3)]
                agents = [get_agent(profile, s) for s, profile in enumerate(lineup)]
                for s, agent in enumerate(agents):
                    agent.rng = Random(stage_seed(deal_seed, f"seat-{s}"))
                started = time.perf_counter()
                round_ = Round(
                    dealer=index % 3, seed=deal_seed, agents=agents, start=False
                )
                won = round_.start()
                soloist = round_.result.soloist
                points = [round_.result.score if s == soloist else 0 for s in range(3)]
                rows.append(
                    dict(
                        agent=candidate,
                        panel=panel,
                        seat=seat,
                        seed=deal_seed,
                        deal_id=f"{seed}:{index}",
                        deal_index=index,
                        run_seed=seed,
                        dealer=index % 3,
                        deal_code=round_.deal_code,
                        lineup=lineup,
                        soloist=soloist,
                        won=None
                        if won is None
                        else (won if soloist == seat else not won),
                        official_points=points[seat],
                        seat_points=points,
                        relative_points=points[seat]
                        - sum(p for s, p in enumerate(points) if s != seat) / 2,
                        party_win=int(
                            won is not None and (won if soloist == seat else not won)
                        ),
                        result=asdict(round_.result),
                        seconds=time.perf_counter() - started,
                    )
                )
    return rows


def paired_delta(first, second, metric, *, seed=42, resamples=2000):
    """Complete-pair checks, with all correlated seats/panels kept in one deal."""

    def index(rows):
        indexed = {(r["deal_id"], r["panel"], r["seat"]): r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate benchmark outcomes")
        return indexed

    a, b = index(first), index(second)
    if not a or a.keys() != b.keys():
        raise ValueError(
            "paired benchmark requires identical complete deals/seats/panels"
        )
    groups = {}
    for key in sorted(a):
        if a[key]["deal_code"] != b[key]["deal_code"]:
            raise ValueError("paired benchmark deals do not match")
        groups.setdefault(key[0], []).append(b[key][metric] - a[key][metric])
    values = np.asarray([np.mean(group) for group in groups.values()])
    if len({len(group) for group in groups.values()}) != 1:
        raise ValueError("unbalanced deal groups")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, 100):
        draws = rng.integers(
            len(values), size=(min(100, resamples - start), len(values))
        )
        means.extend(values[draws].mean(axis=1))
    return dict(
        delta=float(values.mean()),
        interval_95=np.quantile(means, [0.025, 0.975]).tolist(),
        rounds=len(a),
        independent_deals=len(values),
    )


def validate_outcomes(config, rows):
    expected = {
        (f"{seed}:{i}", panel, candidate, seat)
        for seed in config["seeds"]
        for i in range(config["deals_per_seed"])
        for panel in config["panels"]
        for candidate in config["candidates"]
        for seat in range(3)
    }
    actual = {(r["deal_id"], r["panel"], r["agent"], r["seat"]) for r in rows}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError("benchmark has missing, duplicate or unexpected outcomes")


def report(config, rows):
    validate_outcomes(config, rows)
    panels = {}
    for panel in config["panels"]:
        selected = {
            c: [r for r in rows if r["panel"] == panel and r["agent"] == c]
            for c in config["candidates"]
        }
        summaries = {}
        comparisons = {}
        for candidate, outcomes in selected.items():
            summary = summarize(outcomes)
            # Wilson assumes independent rows; the three seats share each deal.
            summary.pop("wilson_interval")
            summary["hand_games"] = sum(
                r["soloist"] == r["seat"]
                and bool((r["result"]["declaration"] or {}).get("hand"))
                for r in outcomes
            )
            summary["relative_points_per_deal"] = float(
                np.mean([r["relative_points"] for r in outcomes])
            )
            summary["by_seed"] = {
                str(seed): summarize([r for r in outcomes if r["run_seed"] == seed])
                for seed in config["seeds"]
            }
            for per_seed in summary["by_seed"].values():
                per_seed.pop("wilson_interval")
            summary["by_seat"] = {
                str(seat): summarize([r for r in outcomes if r["seat"] == seat])
                for seat in range(3)
            }
            summaries[candidate] = summary
            if candidate != "best":
                comparisons[candidate] = {
                    metric: paired_delta(
                        selected["best"], outcomes, metric, seed=config["seeds"][0]
                    )
                    for metric in ("official_points", "relative_points", "party_win")
                }
        # Isolate pickup's incremental contribution after neural declaration.
        for baseline in ("declaration", "mcts", "mcts-default"):
            if baseline in selected:
                comparisons[f"all-neural-minus-{baseline}"] = {
                    metric: paired_delta(
                        selected[baseline],
                        selected["all-neural"],
                        metric,
                        seed=config["seeds"][0],
                    )
                    for metric in ("official_points", "relative_points", "party_win")
                }
        panels[panel] = dict(summary=summaries, differences_from_best=comparisons)
    mixed = {}
    for candidate, panel in (
        (candidate, panel)
        for candidate in config["candidates"]
        for panel in config["panels"]
        if candidate != panel
    ):
        selected = [r for r in rows if r["agent"] == candidate and r["panel"] == panel]
        zeros = [dict(r, relative_points=0) for r in selected]
        mixed[f"{candidate}-vs-two-{panel}"] = dict(
            candidate_points_per_round=float(
                np.mean([r["official_points"] for r in selected])
            ),
            reference_points_per_round=float(
                np.mean(
                    [
                        (sum(r["seat_points"]) - r["official_points"]) / 2
                        for r in selected
                    ]
                )
            ),
            candidate_minus_average_reference=paired_delta(
                zeros, selected, "relative_points", seed=config["seeds"][0]
            ),
        )
    return dict(
        manifest=config,
        status="complete",
        panels=panels,
        mixed_tables=mixed,
        actual_rounds=len(rows),
        independent_deals=len(config["seeds"]) * config["deals_per_seed"],
    )


def worker(config_path, start, count, output):
    config = json.loads(Path(config_path).read_text())
    verify_models(config)
    get_agent = agent_factory(config)
    verify_models(config)
    with Path(output).open("x") as stream:
        for job in range(start, start + count):
            seed = config["seeds"][job // config["deals_per_seed"]]
            index = job % config["deals_per_seed"]
            rows = play_deal(config, seed, index, get_agent)
            for row in rows:
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            atomic_json(
                Path(output).with_suffix(".status.json"),
                dict(completed_deals=job - start + 1, assigned_deals=count),
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--deals",
        type=int,
        default=1000,
        help="Independent deals PER seed; each is played at all seats/panels",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261020, 20261021])
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=[k for k in PROFILES if k != "random"],
        help="Candidate profiles; must include best and all-neural",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=["random", "best", "all-neural", "mcts", "mcts-default"],
        help="Fixed reference pairs (default: random, best, all-neural)",
    )
    parser.add_argument(
        "--include-mcts",
        action="store_true",
        help="Add MCTS with the same neural bidding/declaration/pickup",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--count", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    for name, path in MODELS.items():
        parser.add_argument(f"--{name}-model", default=path)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.config, args.start, args.count, args.out)
        return
    if args.deals < 1 or args.workers < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive deals/workers and distinct seeds required")
    models = {name: getattr(args, f"{name}_model") for name in MODELS}
    for path in models.values():
        if not Path(path).is_file():
            parser.error(f"missing checkpoint: {path}")
    if args.out.exists():
        parser.error("use a new output directory; existing benchmarks are preserved")
    import torch

    device = resolve_device(args.device)
    candidates = args.candidates or ["best", "declaration", "pickup", "all-neural"]
    panels = args.panels or ["random", "best", "all-neural"]
    if args.include_mcts:
        if "mcts" not in candidates:
            candidates.append("mcts")
        if args.panels is None:
            panels.append("mcts")
    if (
        not {"best", "all-neural"}.issubset(candidates)
        or len(set(candidates)) != len(candidates)
        or len(set(panels)) != len(panels)
    ):
        parser.error(
            "distinct candidates/panels and both best/all-neural candidates required"
        )
    source = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in [*sorted((root / "skat").rglob("*.py")), Path(__file__).resolve()]:
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())
    config = dict(
        protocol=PROTOCOL,
        measured_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=source.hexdigest(),
        rules_schema=RULES_SCHEMA,
        observation_schema=OBSERVATION_SCHEMA,
        torch_version=str(torch.__version__),
        hip_version=torch.version.hip,
        cuda_version=torch.version.cuda,
        accelerator_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        candidates=candidates,
        panels=panels,
        profiles={k: PROFILES[k] for k in set(candidates + panels)},
        models=models,
        checkpoint_sha256={name: digest(path) for name, path in models.items()},
        seeds=args.seeds,
        deals_per_seed=args.deals,
        workers=min(args.workers, args.deals * len(args.seeds)),
        device=str(device),
        requested_device=args.device,
        bidding_threshold=0.55,
        pickup_margin=0.0,
        mcts=asdict(MCTSConfig()),
        rng_scheme="sha256-preset-deal-and-seat-v1",
        bootstrap="2000-resamples-complete-deals-including-all-seats",
        primary_comparison="all-neural minus best, separately for each fixed reference panel",
    )
    args.out.mkdir(parents=True)
    config_path = args.out / "manifest.json"
    atomic_json(config_path, config)
    atomic_json(args.out / "status.json", dict(status="running", completed_workers=0))
    jobs = args.deals * len(args.seeds)
    processes = []

    def launch(i):
        start, stop = i * jobs // config["workers"], (i + 1) * jobs // config["workers"]
        with (args.out / f"worker-{i}.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.benchmark_presets",
                    "--worker",
                    "--config",
                    str(config_path),
                    "--start",
                    str(start),
                    "--count",
                    str(stop - start),
                    "--out",
                    str(args.out / f"worker-{i}.jsonl"),
                ],
                env=worker_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            if process.wait():
                raise RuntimeError(f"benchmark worker {i} failed; see worker-{i}.log")
        return i

    try:
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(launch, i) for i in range(config["workers"])]
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    i = future.result()
                    atomic_json(
                        args.out / "status.json",
                        dict(status="running", completed_workers=done),
                    )
                    print(
                        f"Worker {i} complete ({done}/{config['workers']})", flush=True
                    )
            except BaseException:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                raise
        rows = []
        for i in range(config["workers"]):
            rows.extend(
                json.loads(line)
                for line in (args.out / f"worker-{i}.jsonl").read_text().splitlines()
            )
        verify_models(config)
        summary = report(config, rows)
        atomic_json(args.out / "summary.json", summary)
        atomic_json(
            args.out / "status.json", dict(status="complete", actual_rounds=len(rows))
        )
        print(json.dumps(summary["mixed_tables"], indent=2), flush=True)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        atomic_json(
            args.out / "status.json",
            dict(status="failed", partial_results="worker-*.jsonl"),
        )
        raise


if __name__ == "__main__":
    main()
