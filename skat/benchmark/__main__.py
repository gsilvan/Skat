from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from skat.agents.rl.dqn import DQN_ARCHITECTURES
from skat.benchmark.compare import comparison_markdown
from skat.benchmark.config import BenchmarkConfig, policy_variants
from skat.benchmark.runner import BenchmarkRunner


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Train every supported agent family and rank them reproducibly."
    )
    result.add_argument(
        "--profile", choices=("smoke", "standard", "full"), default="standard"
    )
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--device", default=None)
    result.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=None
    )
    result.add_argument(
        "--controls",
        dest="include_controls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train policy target/feature controls too (default: enabled).",
    )
    result.add_argument("--output-root", type=Path, default=Path("benchmark_runs"))
    result.add_argument("--run-dir", type=Path, default=None)
    result.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("OLD_RUN", "NEW_RUN"),
        help="Compare common agents on their identical raw deal outcomes.",
    )
    result.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Resume an interrupted run using its exact manifest configuration.",
    )
    result.add_argument(
        "--plan",
        action="store_true",
        help="Print the resolved work without running it.",
    )
    for flag in (
        "dqn-rounds",
        "dqn-batch-size",
        "dqn-learn-start",
        "dqn-buffer-size",
        "distill-rounds",
        "distill-simulations",
        "distill-determinizations",
        "policy-epochs",
        "belief-rounds",
        "belief-epochs",
        "bidding-deals",
        "bidding-simulations",
        "bidding-determinizations",
        "bidding-epochs",
        "evaluation-rounds",
        "evaluation-simulations",
        "evaluation-determinizations",
    ):
        result.add_argument(f"--{flag}", type=int, default=None)
    result.add_argument("--bidding-threshold", type=float, default=None)
    result.add_argument("--bidding-aggressive-threshold", type=float, default=None)
    result.add_argument("--bidding-defensive-threshold", type=float, default=None)
    return result


def _config(args: argparse.Namespace) -> BenchmarkConfig:
    if args.resume:
        manifest = args.resume / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"no manifest at {manifest}")
        return BenchmarkConfig(
            **json.loads(manifest.read_text(encoding="utf-8"))["config"]
        )
    overrides = {}
    for key in BenchmarkConfig.__dataclass_fields__:
        if key == "profile":
            continue
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return BenchmarkConfig.from_profile(args.profile, **overrides)


def _run_dir(args: argparse.Namespace, config: BenchmarkConfig) -> Path:
    if args.resume:
        return args.resume
    if args.run_dir:
        return args.run_dir
    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "nogit"
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return args.output_root / f"{timestamp}-{commit}-seed{config.seed}"


def _plan(config: BenchmarkConfig) -> dict:
    policies = policy_variants(config.include_controls)
    return {
        "config": config.to_dict(),
        "train": [
            "bidding (counterfactual MCTS win/loss model)",
            *[f"dqn-{arch}" for arch in sorted(DQN_ARCHITECTURES)],
            *[variant.name for variant in policies],
            "belief (used by belief-mcts)",
        ],
        "evaluate": [
            "random",
            "mcts",
            *[f"dqn-{arch}" for arch in sorted(DQN_ARCHITECTURES)],
            *[variant.name for variant in policies],
            "belief-mcts",
            "all entries above with aggressive/balanced/defensive neural bidding",
        ],
        "excluded": ["cli", "deep", "recording wrappers"],
    }


def main() -> None:
    args = parser().parse_args()
    if args.compare:
        print(comparison_markdown(*args.compare))
        return
    config = _config(args)
    if args.plan:
        print(json.dumps(_plan(config), indent=2, sort_keys=True))
        return
    run_dir = _run_dir(args, config)
    ranking = BenchmarkRunner(config, run_dir, resume=bool(args.resume)).run()
    print(f"benchmark complete: {run_dir.resolve()}")
    for row in ranking:
        print(f"{row['rank']:2d}. {row['name']:<30} {row['win_rate']:.1%}")


if __name__ == "__main__":
    main()
