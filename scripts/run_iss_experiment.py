"""Run audit, pilot import, GPU training, fixed evaluation and play sequentially.

Every stage has a separate log and durable status. A failed stage stops the run;
failed/partial outputs are never passed off as a completed experiment.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    directory = Path(args.out).resolve()
    if directory.exists():
        raise ValueError("use a new experiment directory")
    reference = Path(args.reference_policy).resolve()
    baseline = Path(args.baseline_policy).resolve()
    archive = Path(args.input).resolve()
    config = Path(args.config).resolve()
    for path in (reference, baseline, archive, config):
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
    # Resolve before changing cwd; support separate SGF and GPU interpreters.
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
            "from skat.utils.reproducibility import resolve_device; import torch; device=resolve_device("
            + repr(args.device)
            + "); print(device); torch.zeros(1, device=device)",
        ],
        cwd=ROOT,
        check=True,
    )
    directory.mkdir(parents=True)
    corpus = directory / "corpus"
    policy = directory / "policy.pt"
    baseline_eval = directory / "baseline-eval.json"
    candidate_eval = directory / "iss-eval.json"
    status = dict(
        pid=os.getpid(),
        started=datetime.now(timezone.utc).isoformat(),
        input=str(archive),
        policy=str(policy),
        corpus=str(corpus),
        stages=[],
    )

    def save():
        temporary = directory / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(directory / "status.json")

    def stage(name, command):
        item = dict(
            name=name,
            command=command,
            state="running",
            started=datetime.now(timezone.utc).isoformat(),
        )
        status["stages"].append(item)
        status["state"] = "running"
        save()
        print(f"Starting {name}: {directory / (name + '.log')}", flush=True)
        try:
            with (directory / (name + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
        except BaseException as error:
            item["state"] = status["state"] = "failed"
            item["error"] = repr(error)
            save()
            raise
        item["state"] = "complete"
        item["finished"] = datetime.now(timezone.utc).isoformat()
        save()

    stage(
        "audit",
        [
            sgf_python,
            "-m",
            "iss",
            "audit",
            "--input",
            str(archive),
            "--out",
            str(directory / "audit"),
        ],
    )
    stage(
        "import",
        [
            sgf_python,
            "-m",
            "iss",
            "import",
            "--audit",
            str(directory / "audit"),
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
            "-m",
            "skat",
            "train",
            "--config",
            str(config),
            "--data",
            str(corpus),
            "--out",
            str(policy),
            "--device",
            args.device,
            "--seed",
            "7",
        ],
    )
    common = [
        "--reference-policy",
        str(reference),
        "--deals",
        str(args.deals),
        "--seed",
        "42",
        "--device",
        "cpu",
    ]
    stage(
        "baseline-evaluate",
        [
            sys.executable,
            "-m",
            "skat",
            "evaluate",
            "--agents",
            "random",
            "mcts",
            "policy",
            "--policy",
            str(baseline),
            *common,
            "--out",
            str(baseline_eval),
        ],
    )
    stage(
        "iss-evaluate",
        [
            sys.executable,
            "-m",
            "skat",
            "evaluate",
            "--agents",
            "policy",
            "--policy",
            str(policy),
            *common,
            "--out",
            str(candidate_eval),
        ],
    )
    stage(
        "compare",
        [
            sys.executable,
            "-m",
            "skat.compare",
            str(baseline_eval),
            str(candidate_eval),
            "--out",
            str(directory / "comparison.json"),
        ],
    )
    stage(
        "play",
        [
            sys.executable,
            "-m",
            "skat",
            "play",
            "--agents",
            "policy",
            "policy",
            "mcts",
            "--policy0",
            str(policy),
            "--policy1",
            str(baseline),
            "--rounds",
            str(args.play_rounds),
            "--seed",
            "43",
            "--device",
            "cpu",
        ],
    )
    status["state"] = "complete"
    status["finished"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"ISS experiment complete: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sgf-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=str(ROOT / "configs/iss.toml"))
    parser.add_argument("--play-rounds", type=int, default=1000)
    parser.add_argument("--max-games", type=int, default=100000)
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument(
        "--reference-policy", default=str(ROOT / "trained_models/transformer-run-2.pt")
    )
    parser.add_argument(
        "--baseline-policy",
        default=str(ROOT / "trained_models/transformer-run-2-100epochs.pt"),
    )
    args = parser.parse_args()
    if args.max_games < 3 or args.deals < 1 or args.play_rounds < 1:
        parser.error("positive deals and at least three games required")
    run(args)


if __name__ == "__main__":
    main()
