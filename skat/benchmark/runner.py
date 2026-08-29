from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from skat.agents.belief.samples import BeliefShardWriter
from skat.agents.belief.train import train as train_belief
from skat.agents.belief_mcts import BeliefMCTSAgent
from skat.agents.belief_recorder import BeliefRecorder
from skat.agents.bidding import BiddingStyle
from skat.agents.distill.samples import ShardWriter
from skat.agents.distill.train import train as train_policy
from skat.agents.dqn import DQNAgent
from skat.agents.mcts import MCTSAgent, RecordingMCTSAgent
from skat.agents.neural_bidding import NeuralBidder
from skat.agents.neural_bidding.collect import collect as collect_bidding
from skat.agents.neural_bidding.train import train as train_bidding
from skat.agents.policy import PolicyAgent
from skat.agents.random import RandomAgent
from skat.agents.rl.dqn import DQN_ARCHITECTURES
from skat.benchmark.config import BenchmarkConfig, PolicyVariant, policy_variants
from skat.benchmark.evaluation import AgentFactory, evaluate_candidate, rank_results
from skat.game import Tournament
from skat.models import PolicyNet
from skat.models.belief import BeliefNet
from skat.utils.reproducibility import seed_everything

SCHEMA_VERSION = 2
EXCLUDED_AGENTS = {
    "cli": "interactive; cannot be trained or evaluated unattended",
    "deep": "unfinished stub: three decisions are unimplemented and card play is random",
    "recording-mcts": "data-collection wrapper; play is identical to mcts",
    "belief-recorders": "data-collection mixins, not distinct playing policies",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stage_seed(root: int, name: str) -> int:
    digest = hashlib.sha256(f"skat-benchmark-v1:{root}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def provenance() -> dict[str, Any]:
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "torch_hip": getattr(torch.version, "hip", None),
        "command": sys.argv,
    }


class Journal:
    """Crash-safe manifest and append-only event stream for one benchmark run."""

    def __init__(self, run_dir: Path, config: BenchmarkConfig, resume: bool) -> None:
        self.run_dir = run_dir.resolve()
        self.path = self.run_dir / "manifest.json"
        self.events = self.run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not resume:
                raise FileExistsError(
                    f"{self.path} already exists; pass --resume {self.run_dir}"
                )
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data["config"] != config.to_dict():
                raise ValueError("resume configuration differs from manifest.json")
        else:
            self.data = {
                "schema": SCHEMA_VERSION,
                "status": "running",
                "started_at": utc_now(),
                "finished_at": None,
                "config": config.to_dict(),
                "provenance": provenance(),
                "seed_strategy": (
                    "SHA-256-derived per-stage seeds; per-agent private Python/Torch "
                    "generators; deterministic Torch kernels"
                ),
                "evaluation_protocol": (
                    "same deals against random and trained-policy panels; candidate "
                    "rotated through all three seats; conservative plus aggressive/"
                    "balanced/defensive neural bidding; rank by designated-seat "
                    "party win rate across both populations"
                ),
                "bidding_training_protocol": (
                    "each seeded deal replayed three times with every seat forced "
                    "soloist; all card play by MCTS; win=+1/loss=-1; split by deal"
                ),
                "excluded_agents": EXCLUDED_AGENTS,
                "stages": {},
            }
            self.save()

    def save(self) -> None:
        _atomic_json(self.path, self.data)

    def event(self, kind: str, stage: str, **details: Any) -> None:
        record = {"at": utc_now(), "event": kind, "stage": stage, **details}
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def run_stage(self, name: str, function: Callable[[], dict[str, Any]]) -> dict:
        existing = self.data["stages"].get(name)
        if existing and existing.get("status") == "complete":
            self.event("skipped", name, reason="already complete")
            return existing.get("details", {})

        started = time.monotonic()
        self.data["stages"][name] = {"status": "running", "started_at": utc_now()}
        self.save()
        self.event("started", name)
        try:
            details = function()
        except BaseException as error:
            self.data["stages"][name].update(
                {"status": "failed", "finished_at": utc_now(), "error": repr(error)}
            )
            self.data["status"] = "failed"
            self.save()
            self.event("failed", name, error=repr(error))
            raise
        duration = time.monotonic() - started
        self.data["stages"][name].update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "duration_seconds": duration,
                "details": details,
            }
        )
        self.data["status"] = "running"
        self.save()
        self.event("completed", name, duration_seconds=duration, **details)
        return details


class BenchmarkRunner:
    def __init__(
        self, config: BenchmarkConfig, run_dir: Path, resume: bool = False
    ) -> None:
        config.validate()
        self.config = config
        self.run_dir = run_dir.resolve()
        self.journal = Journal(self.run_dir, config, resume)
        self.checkpoints = self.run_dir / "checkpoints"
        self.data_dir = self.run_dir / "data"
        self.logs = self.run_dir / "logs"
        for directory in (self.checkpoints, self.data_dir, self.logs):
            directory.mkdir(parents=True, exist_ok=True)

    def _seed(self, name: str) -> int:
        return stage_seed(self.config.seed, name)

    def _reset_dir(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.run_dir):
            raise ValueError(f"refusing to reset path outside run: {resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True)

    def train_dqn(self, arch: str) -> dict[str, Any]:
        name = f"train-dqn-{arch}"
        # Architecture is the variable under test; deals, opponents and model
        # initialisation seed stay common across the DQN family.
        seed = self._seed("dqn-training")
        seed_everything(seed, self.config.deterministic)
        checkpoint = self.checkpoints / f"dqn-{arch}.pt"
        checkpoint.unlink(missing_ok=True)
        log_dir = self.logs / name
        self._reset_dir(log_dir)
        learner = DQNAgent(
            train=True,
            path=str(checkpoint),
            arch=arch,
            device=self.config.device,
            batch_size=self.config.dqn_batch_size,
            learn_start=self.config.dqn_learn_start,
            buffer_size=self.config.dqn_buffer_size,
            seed=seed,
            log_dir=str(log_dir),
        )
        agents = [
            learner,
            RandomAgent(seed=stage_seed(seed, "opponent-1")),
            RandomAgent(seed=stage_seed(seed, "opponent-2")),
        ]
        tournament = Tournament(rounds=self.config.dqn_rounds, agents=agents, seed=seed)
        try:
            tournament.start()
            learner.save()
        finally:
            learner.close()
        return {
            "seed": seed,
            "rounds": self.config.dqn_rounds,
            "transitions": len(learner.dqn.replay_buffer),
            "gradient_steps": learner.dqn.gradient_steps,
            "checkpoint": str(checkpoint.relative_to(self.run_dir)),
            "checkpoint_sha256": file_sha256(checkpoint),
        }

    def collect_policy(self) -> dict[str, Any]:
        name = "collect-policy-corpus"
        seed = self._seed(name)
        seed_everything(seed, self.config.deterministic)
        directory = self.data_dir / "distill"
        self._reset_dir(directory)
        recorder = ShardWriter(
            str(directory),
            input_size=PolicyNet.INPUT_SIZE,
            tag=str(seed),
            provenance={
                "seed": seed,
                "simulations": self.config.distill_simulations,
                "determinizations": self.config.distill_determinizations,
            },
        )
        agents = [
            RecordingMCTSAgent(
                recorder=recorder,
                simulations=self.config.distill_simulations,
                determinizations=self.config.distill_determinizations,
                seed=stage_seed(seed, f"seat-{seat}"),
            )
            for seat in range(3)
        ]
        try:
            Tournament(
                rounds=self.config.distill_rounds, agents=agents, seed=seed
            ).start()
        finally:
            recorder.close()
        if not recorder.written:
            raise RuntimeError("MCTS collection produced no trainable decisions")
        return {
            "seed": seed,
            "rounds": self.config.distill_rounds,
            "decisions": recorder.written,
            "shards": recorder.shards,
            "directory": str(directory.relative_to(self.run_dir)),
        }

    def train_policy_variant(self, variant: PolicyVariant) -> dict[str, Any]:
        name = f"train-{variant.name}"
        # Every policy sees the identical group split and batch permutation.
        # Otherwise an architecture/target comparison would also compare a
        # different 5% of held-out rounds.
        seed = self._seed("policy-training")
        checkpoint = self.checkpoints / f"{variant.name}.pt"
        checkpoint.unlink(missing_ok=True)
        log_dir = self.logs / name
        self._reset_dir(log_dir)
        args = Namespace(
            data=str(self.data_dir / "distill"),
            out=str(checkpoint),
            arch=variant.arch,
            features=variant.features,
            epochs=self.config.policy_epochs,
            batch_size=None,
            lr=None,
            weight_decay=None,
            dropout=None,
            schedule=None,
            warmup=None,
            target=variant.target,
            target_temperature=None,
            val_fraction=0.05,
            log_dir=str(log_dir),
            seed=seed,
            device=self.config.device,
        )
        train_policy(args)
        return {
            "seed": seed,
            "arch": variant.arch,
            "target": variant.target,
            "features": variant.features,
            "checkpoint": str(checkpoint.relative_to(self.run_dir)),
            "checkpoint_sha256": file_sha256(checkpoint),
        }

    def collect_belief(self) -> dict[str, Any]:
        name = "collect-belief-corpus"
        seed = self._seed(name)
        seed_everything(seed, self.config.deterministic)
        directory = self.data_dir / "belief"
        self._reset_dir(directory)
        recorder = BeliefShardWriter(
            str(directory), input_size=BeliefNet.INPUT_SIZE, tag=str(seed)
        )
        checkpoint = self.checkpoints / "policy-transformer.pt"
        recording_policy = type(
            "BenchmarkBeliefRecordingPolicy", (BeliefRecorder, PolicyAgent), {}
        )
        agents = [
            recording_policy(
                recorder=recorder,
                path=str(checkpoint),
                device=self.config.device,
                seed=stage_seed(seed, f"seat-{seat}"),
            )
            for seat in range(3)
        ]
        try:
            Tournament(
                rounds=self.config.belief_rounds, agents=agents, seed=seed
            ).start()
        finally:
            recorder.close()
        if not recorder.written:
            raise RuntimeError("belief collection produced no observations")
        return {
            "seed": seed,
            "rounds": self.config.belief_rounds,
            "observations": recorder.written,
            "shards": recorder.shards,
            "teacher": str(checkpoint.relative_to(self.run_dir)),
            "directory": str(directory.relative_to(self.run_dir)),
        }

    def train_belief_model(self) -> dict[str, Any]:
        name = "train-belief"
        seed = self._seed(name)
        checkpoint = self.checkpoints / "belief.pt"
        checkpoint.unlink(missing_ok=True)
        log_dir = self.logs / name
        self._reset_dir(log_dir)
        args = Namespace(
            data=str(self.data_dir / "belief"),
            out=str(checkpoint),
            epochs=self.config.belief_epochs,
            batch_size=512,
            lr=3e-4,
            weight_decay=0.01,
            dropout=0.1,
            warmup=500,
            val_fraction=0.05,
            log_dir=str(log_dir),
            seed=seed,
            device=self.config.device,
        )
        train_belief(args)
        return {
            "seed": seed,
            "checkpoint": str(checkpoint.relative_to(self.run_dir)),
            "checkpoint_sha256": file_sha256(checkpoint),
        }

    def collect_bidding_corpus(self) -> dict[str, Any]:
        name = "collect-bidding-corpus"
        seed = self._seed(name)
        directory = self.data_dir / "bidding"
        self._reset_dir(directory)
        details = collect_bidding(
            str(directory),
            deals=self.config.bidding_deals,
            simulations=self.config.bidding_simulations,
            determinizations=self.config.bidding_determinizations,
            seed=seed,
        )
        return {
            "seed": seed,
            **details,
            "directory": str(directory.relative_to(self.run_dir)),
        }

    def train_bidding_model(self) -> dict[str, Any]:
        name = "train-bidding"
        seed = self._seed(name)
        checkpoint = self.checkpoints / "bidding.pt"
        checkpoint.unlink(missing_ok=True)
        log_dir = self.logs / name
        self._reset_dir(log_dir)
        metrics = train_bidding(
            Namespace(
                data=str(self.data_dir / "bidding"),
                out=str(checkpoint),
                epochs=self.config.bidding_epochs,
                batch_size=256,
                lr=3e-4,
                weight_decay=0.01,
                dropout=0.1,
                hidden_size=128,
                val_fraction=0.1,
                threshold=self.config.bidding_threshold,
                log_dir=str(log_dir),
                seed=seed,
                device=self.config.device,
            )
        )
        return {
            "seed": seed,
            **metrics,
            "checkpoint": str(checkpoint.relative_to(self.run_dir)),
            "checkpoint_sha256": file_sha256(checkpoint),
        }

    def factories(self, variants: list[PolicyVariant]) -> dict[str, AgentFactory]:
        factories: dict[str, AgentFactory] = {
            "random": lambda seed, _seat: RandomAgent(seed=seed),
            "mcts": lambda seed, _seat: MCTSAgent(
                simulations=self.config.evaluation_simulations,
                determinizations=self.config.evaluation_determinizations,
                seed=seed,
            ),
        }
        for arch in sorted(DQN_ARCHITECTURES):
            checkpoint = self.checkpoints / f"dqn-{arch}.pt"

            def dqn_factory(
                seed: int, seat: int, *, arch: str = arch, checkpoint: Path = checkpoint
            ) -> DQNAgent:
                return DQNAgent(
                    train=False,
                    path=str(checkpoint),
                    arch=arch,
                    device=self.config.device,
                    seed=seed,
                    log_dir=str(self.logs / "evaluation" / f"dqn-{arch}" / str(seat)),
                )

            factories[f"dqn-{arch}"] = dqn_factory
        for variant in variants:
            checkpoint = self.checkpoints / f"{variant.name}.pt"

            def policy_factory(
                seed: int,
                _seat: int,
                *,
                checkpoint: Path = checkpoint,
            ) -> PolicyAgent:
                return PolicyAgent(
                    path=str(checkpoint), device=self.config.device, seed=seed
                )

            factories[variant.name] = policy_factory
        belief = self.checkpoints / "belief.pt"
        factories["belief-mcts"] = lambda seed, _seat: BeliefMCTSAgent(
            path=str(belief),
            device=self.config.device,
            simulations=self.config.evaluation_simulations,
            determinizations=self.config.evaluation_determinizations,
            seed=seed,
        )
        conservative = dict(factories)
        modes = {
            "aggressive": self.config.bidding_aggressive_threshold,
            "balanced": self.config.bidding_threshold,
            "defensive": self.config.bidding_defensive_threshold,
        }
        checkpoint = str(self.checkpoints / "bidding.pt")
        bidders = {
            mode: NeuralBidder(
                checkpoint, device=self.config.device, threshold=threshold
            )
            for mode, threshold in modes.items()
        }
        for name, base_factory in conservative.items():
            for mode, bidder in bidders.items():

                def neural_factory(
                    seed: int,
                    seat: int,
                    *,
                    base_factory: AgentFactory = base_factory,
                    bidder: NeuralBidder = bidder,
                ):
                    agent = base_factory(seed, seat)
                    agent.bidding = BiddingStyle.NEURAL
                    agent.set_neural_bidder(bidder)
                    return agent

                factories[f"{name}-neural-{mode}"] = neural_factory
        return factories

    def evaluate(self, variants: list[PolicyVariant]) -> dict[str, Any]:
        seed = self._seed("evaluation")
        seed_everything(seed, self.config.deterministic)
        results = []
        partial = self.run_dir / "results.partial.json"
        factories = self.factories(variants)
        anchors = {
            "random": factories["random"],
            "policy-transformer": factories["policy-transformer"],
        }
        for name, factory in factories.items():
            result = evaluate_candidate(
                name, factory, self.config.evaluation_rounds, seed, anchors
            )
            results.append(result)
            _atomic_json(partial, {"seed": seed, "results": rank_results(results)})
        ranking = rank_results(results)
        _atomic_json(self.run_dir / "results.json", {"seed": seed, "ranking": ranking})
        partial.unlink(missing_ok=True)
        self.write_csv(ranking)
        self.write_report(ranking)
        return {
            "seed": seed,
            "agents": len(ranking),
            "winner": ranking[0]["name"],
            "results": "results.json",
            "ranking": "ranking.csv",
            "report": "report.md",
        }

    def write_csv(self, ranking: list[dict[str, Any]]) -> None:
        fields = [
            "rank",
            "name",
            "win_rate",
            "solo_win_rate",
            "defence_win_rate",
            "card_points_per_round",
            "wins",
            "rounds",
            "passed",
            "overbids",
        ]
        with (self.run_dir / "ranking.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ranking)

    def write_report(self, ranking: list[dict[str, Any]]) -> None:
        lines = [
            "# Skat agent benchmark",
            "",
            (
                f"Profile: `{self.config.profile}` · seed: `{self.config.seed}` · "
                f"device: `{self.config.device}`"
            ),
            "",
            (
                "Each candidate played the same deals against two fixed panels—two "
                "seeded RandomAgents and two copies of the trained transformer policy—"
                "and occupied every seat once per deal and panel. The competent panel "
                "matters because the belief model is population-specific. The interval "
                "is a 95% Wilson interval over the candidate's party wins."
            ),
            (
                "Every agent also has aggressive, balanced and defensive neural "
                "bidding variants. They share one calibrated MCTS-trained model and "
                "differ only in accepted loss risk (thresholds "
                f"{self.config.bidding_aggressive_threshold:.0%}, "
                f"{self.config.bidding_threshold:.0%}, and "
                f"{self.config.bidding_defensive_threshold:.0%}). A hand accepted "
                "by the net still uses the conservative bid ceiling."
            ),
            "",
            "| rank | agent | win rate | 95% CI | solo | defence | card points/round |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in ranking:
            low, high = row["win_rate_95"]
            lines.append(
                f"| {row['rank']} | {row['name']} | {row['win_rate']:.1%} | "
                f"{low:.1%}–{high:.1%} | {row['solo_win_rate']:.1%} | "
                f"{row['defence_win_rate']:.1%} | "
                f"{row['card_points_per_round']:.2f} |"
            )
        lines.extend(
            [
                "",
                "## Reproduce or resume",
                "",
                f"`python -m skat.benchmark --resume {self.run_dir}`",
                "",
                (
                    "Exact configuration, Git revision, dependency versions, stage "
                    "seeds and durations are in `manifest.json`; incremental events "
                    "are in `events.jsonl` and training curves under `logs/`."
                ),
                "",
                (
                    "Results are expected to repeat on the same software and device. "
                    "Different CPU/GPU backends are recorded but are not guaranteed "
                    "to be bit-identical."
                ),
            ]
        )
        (self.run_dir / "report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def run(self) -> list[dict[str, Any]]:
        variants = policy_variants(self.config.include_controls)
        self.journal.run_stage("collect-bidding-corpus", self.collect_bidding_corpus)
        self.journal.run_stage("train-bidding", self.train_bidding_model)
        for arch in sorted(DQN_ARCHITECTURES):
            self.journal.run_stage(
                f"train-dqn-{arch}", lambda arch=arch: self.train_dqn(arch)
            )
        self.journal.run_stage("collect-policy-corpus", self.collect_policy)
        for variant in variants:
            self.journal.run_stage(
                f"train-{variant.name}",
                lambda variant=variant: self.train_policy_variant(variant),
            )
        self.journal.run_stage("collect-belief-corpus", self.collect_belief)
        self.journal.run_stage("train-belief", self.train_belief_model)
        self.journal.run_stage("evaluation", lambda: self.evaluate(variants))

        results = json.loads(
            (self.run_dir / "results.json").read_text(encoding="utf-8")
        )["ranking"]
        self.journal.data["status"] = "complete"
        self.journal.data["finished_at"] = utc_now()
        self.journal.data["winner"] = results[0]["name"]
        if not self.journal.data.get("history_recorded"):
            history = self.run_dir.parent / "history.jsonl"
            with history.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "finished_at": self.journal.data["finished_at"],
                            "run_dir": str(self.run_dir),
                            "profile": self.config.profile,
                            "seed": self.config.seed,
                            "git_commit": self.journal.data["provenance"]["git_commit"],
                            "winner": results[0]["name"],
                            "ranking": [row["name"] for row in results],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            self.journal.data["history_recorded"] = True
        self.journal.save()
        self.journal.event("completed", "benchmark", winner=results[0]["name"])
        return results
