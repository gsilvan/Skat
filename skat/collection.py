"""Collect independent MCTS games in CPU worker processes."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from skat.agents.distill.samples import ShardWriter
from skat.agents.mcts import RecordingMCTSAgent
from skat.config import MCTSConfig
from skat.game import Round
from skat.utils.processes import available_cpus, worker_environment
from skat.utils.reproducibility import stage_seed


def _publish(directory, output):
    for shard in directory.glob("*.npz"):
        shard.replace(output / shard.name)
    if directory.exists() and not any(directory.iterdir()):
        directory.rmdir()


def _worker(
    output, index, start, rounds, seed, mcts, report, belief_model=None, device="cpu"
):
    output = Path(output)
    directory = output / f".worker-{index}"
    provenance = dict(
        seed=seed,
        round_start=start,
        rounds=rounds,
        mcts_rng_scheme="independent-world-v1",
        **mcts,
    )
    try:
        with ShardWriter(directory, provenance=provenance) as writer:
            writer.group = 3 * start - 1
            if belief_model:
                from skat.agents.belief.inference import BeliefSampler

            agents = [
                RecordingMCTSAgent(
                    writer,
                    sampler=BeliefSampler(belief_model, device=device)
                    if belief_model
                    else None,
                    **mcts,
                )
                for _ in range(3)
            ]
            for deal in range(start, start + rounds):
                deal_seed = stage_seed(seed, f"collect-deal-{deal}")
                for seat, agent in enumerate(agents):
                    agent.rng.seed(stage_seed(deal_seed, f"seat-{seat}"))
                Round(dealer=deal % 3, agents=agents, seed=deal_seed)
                report(deal, len(writer))
    finally:
        _publish(directory, output)


def _stop(processes):
    for process in processes:
        if process.poll() is None:
            try:
                if os.name == "posix":
                    process.send_signal(signal.SIGINT)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def collect(
    output,
    *,
    rounds=1000,
    seed=1,
    mcts=None,
    workers=None,
    progress=True,
    belief_model=None,
    device="cpu",
):
    """Return collection metadata; worker count never changes a deal's searches."""
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("workers must be a positive integer")
    mcts = mcts or MCTSConfig()
    count = min(workers or available_cpus(), rounds) if rounds else 0
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        rounds=rounds,
        completed_rounds=0,
        decisions=0,
        workers=count,
        seed=seed,
        mcts=asdict(mcts),
        status="running",
        seed_scheme="sha256-collect-deal-v1",
        mcts_rng_scheme="independent-world-v1",
        belief_model=belief_model,
        belief_model_sha256=hashlib.sha256(Path(belief_model).read_bytes()).hexdigest()
        if belief_model
        else None,
    )
    processes, logs = [], []
    readers = None

    def save():
        temporary = output / "collection.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "collection.json")

    save()
    try:
        with tqdm(
            total=rounds,
            desc=f"MCTS ({count} workers)",
            unit="round",
            dynamic_ncols=True,
            disable=not progress,
        ) as bar:

            def report(deal, decisions):
                with tqdm.get_lock():
                    manifest["completed_rounds"] += 1
                    bar.set_postfix(
                        remaining=rounds - manifest["completed_rounds"], refresh=False
                    )
                    bar.update(1)

            if count == 1:
                _worker(
                    str(output),
                    0,
                    0,
                    rounds,
                    seed,
                    asdict(mcts),
                    report,
                    belief_model,
                    device,
                )
            elif count:
                environment = worker_environment()
                # Workers do Python search; NumPy's numerical thread pools would
                # only oversubscribe the CPU during startup and small operations.
                start = 0
                for index in range(count):
                    size = rounds // count + int(index < rounds % count)
                    job = dict(
                        output=str(output),
                        index=index,
                        start=start,
                        rounds=size,
                        seed=seed,
                        mcts=asdict(mcts),
                        belief_model=belief_model,
                        device=device,
                    )
                    log = (output / f"worker-{index}.log").open("w")
                    logs.append(log)
                    processes.append(
                        subprocess.Popen(
                            [sys.executable, "-m", "skat.collection", json.dumps(job)],
                            stdout=subprocess.PIPE,
                            stderr=log,
                            text=True,
                            env=environment,
                        )
                    )
                    start += size

                def monitor(index, process):
                    for line in process.stdout:
                        event = json.loads(line)
                        report(event["round"], event["decisions"])
                    if process.wait():
                        raise RuntimeError(
                            f"MCTS worker {index} failed; see {output / f'worker-{index}.log'}"
                        )

                readers = ThreadPoolExecutor(max_workers=count)
                futures = [
                    readers.submit(monitor, i, p) for i, p in enumerate(processes)
                ]
                for future in as_completed(futures):
                    future.result()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _stop(processes)
        raise
    finally:
        if readers is not None:
            readers.shutdown(wait=True)
        for process in processes:
            process.stdout.close()
        for log in logs:
            log.close()
        for directory in output.glob(".worker-*"):
            _publish(directory, output)
        for shard in output.glob("*.npz"):
            with np.load(shard, allow_pickle=False) as data:
                manifest["decisions"] += len(data["groups"])
        save()
    return manifest


if __name__ == "__main__":

    def emit(deal, decisions):
        print(json.dumps(dict(round=deal, decisions=decisions)), flush=True)

    _worker(**json.loads(sys.argv[1]), report=emit)
