"""Run ISS bidding import/training and separate risk sweeps for three card players."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    output = Path(args.out).resolve()
    if output.exists():
        raise ValueError("use a new bidding experiment directory")
    audit = Path(args.audit).resolve()
    policies = {
        "iss": Path(args.iss_policy).resolve(),
        "large": Path(args.large_policy).resolve(),
    }
    config = Path(args.config).resolve()
    search_config = Path(args.search_config).resolve()
    for path in (
        audit / "audit.json",
        audit / "index.sqlite",
        *policies.values(),
        config,
        search_config,
    ):
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
    sgf_python = str(Path(args.sgf_python).absolute())
    subprocess.run(
        [
            sgf_python,
            "-c",
            "from iss import require_sgf_support; require_sgf_support()",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from skat.utils.reproducibility import resolve_device; torch.zeros(1,device=resolve_device("
            + repr(args.device)
            + "))",
        ],
        cwd=ROOT,
        check=True,
    )
    output.mkdir(parents=True)
    corpus, model = output / "corpus", output / "bidding.pt"
    status = dict(
        pid=os.getpid(),
        started=datetime.now(timezone.utc).isoformat(),
        state="running",
        stages=[],
    )

    def save():
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "status.json")

    def stage(name, command):
        item = dict(
            name=name,
            command=command,
            state="running",
            started=datetime.now(timezone.utc).isoformat(),
        )
        status["stages"].append(item)
        save()
        print(f"Starting {name}: {output / (name + '.log')}", flush=True)
        try:
            with (output / (name + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
        except BaseException as error:
            status["state"] = item["state"] = "failed"
            item["error"] = repr(error)
            save()
            raise
        item["state"] = "complete"
        item["finished"] = datetime.now(timezone.utc).isoformat()
        save()

    module = ["-m", "skat.agents.neural_bidding"]
    stage(
        "import",
        [
            sgf_python,
            *module,
            "import",
            "--audit",
            str(audit),
            "--out",
            str(corpus),
            "--max-games",
            str(args.max_games),
            "--seed",
            "7",
        ],
    )
    stage(
        "train",
        [
            sys.executable,
            *module,
            "train",
            "--data",
            str(corpus),
            "--out",
            str(model),
            "--config",
            str(config),
            "--device",
            args.device,
            "--seed",
            "7",
        ],
    )
    for name, agent, policy in (
        ("iss", "policy", policies["iss"]),
        ("large", "policy", policies["large"]),
        ("mcts", "mcts", policies["iss"]),
    ):
        stage(
            "benchmark-" + name,
            [
                sys.executable,
                *module,
                "benchmark",
                "--agent",
                agent,
                "--policy",
                str(policy),
                "--reference-policy",
                str(policies["iss"]),
                "--bidding-model",
                str(model),
                "--config",
                str(search_config),
                "--out",
                str(output / ("benchmark-" + name)),
                "--device",
                "cpu",
                "--validation-deals",
                str(args.validation_deals),
                "--deals",
                str(args.deals),
            ],
        )
    selected = json.loads((output / "benchmark-iss/report.json").read_text())[
        "selected_threshold"
    ]
    stage(
        "play",
        [
            sys.executable,
            "-m",
            "skat",
            "play",
            "--config",
            str(search_config),
            "--agents",
            "policy",
            "policy",
            "mcts",
            "--policy0",
            str(policies["iss"]),
            "--policy1",
            str(policies["large"]),
            "--bidding0",
            "neural",
            "--bidding-model",
            str(model),
            "--bidding-threshold",
            str(selected),
            "--rounds",
            str(args.play_rounds),
            "--seed",
            "44",
            "--device",
            "cpu",
        ],
    )
    status["state"] = "complete"
    status["finished"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"Bidding experiment complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default=str(ROOT / "benchmark_runs/iss-run-1/audit"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--sgf-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=str(ROOT / "configs/bidding.toml"))
    parser.add_argument(
        "--search-config", default=str(ROOT / "configs/teacher-2000.toml")
    )
    parser.add_argument("--max-games", type=int, default=200000)
    parser.add_argument("--validation-deals", type=int, default=500)
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--play-rounds", type=int, default=1000)
    parser.add_argument(
        "--iss-policy", default=str(ROOT / "benchmark_runs/iss-run-1/policy.pt")
    )
    parser.add_argument(
        "--large-policy",
        default=str(ROOT / "trained_models/transformer-teacher-2000-10k.pt"),
    )
    args = parser.parse_args()
    if (
        args.max_games < 3
        or args.validation_deals < 1
        or args.deals < 1
        or args.play_rounds < 1
    ):
        parser.error("at least three corpus games and positive deal counts required")
    run(args)


if __name__ == "__main__":
    main()
