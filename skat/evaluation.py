"""Common-deal evaluation; raw outcomes and deal-clustered paired comparisons."""

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from random import Random

import numpy as np

from skat.config import MCTSConfig
from skat.game import Round
from skat.models.observation import OBSERVATION_SCHEMA
from skat.rules import RULES_SCHEMA
from skat.utils.reproducibility import stage_seed


def wilson(wins, total):
    if not total:
        return None
    z = 1.96
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [center - half, center + half]


def summarize(rows):
    played = [r for r in rows if r["won"] is not None]
    wins = sum(r["won"] for r in played)

    def role(solo):
        group = [r for r in played if (r["soloist"] == r["seat"]) == solo]
        return dict(wins=sum(r["won"] for r in group), played=len(group))

    elapsed = sum(r["seconds"] for r in rows)
    return dict(
        wins=wins,
        played=len(played),
        passed=len(rows) - len(played),
        win_rate=wins / len(played) if played else None,
        wilson_interval=wilson(wins, len(played)),
        soloist=role(True),
        defence=role(False),
        official_points=sum(r["official_points"] for r in played),
        seconds=elapsed,
        rounds_per_second=len(rows) / elapsed if elapsed else None,
    )


def paired_comparison(first, second, seed=1):
    def index(rows):
        return {
            (r["panel"], r["seat"], r["seed"]): r for r in rows if r["won"] is not None
        }

    a, b = index(first), index(second)
    keys = sorted(a.keys() & b.keys())
    groups = {}
    better = worse = 0
    for key in keys:
        delta = int(b[key]["won"]) - int(a[key]["won"])
        better += int(delta > 0)
        worse += int(delta < 0)
        groups.setdefault(key[2], []).append(delta)
    if not keys:
        return dict(paired=0)
    # Bootstrap whole deals; their six seat/panel outcomes are correlated.
    sums = np.asarray([sum(v) for v in groups.values()], dtype=float)
    sizes = np.asarray([len(v) for v in groups.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(2000, len(groups)))
    deltas = sums[draws].sum(1) / sizes[draws].sum(1)
    n = better + worse
    p_value = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2**n)
        if n
        else 1.0
    )
    return dict(
        paired=len(keys),
        deal_groups=len(groups),
        delta=(better - worse) / len(keys),
        deal_bootstrap_interval=np.quantile(deltas, [0.025, 0.975]).tolist(),
        improved=better,
        regressed=worse,
        descriptive_mcnemar_p=p_value,
    )


def evaluate(
    agent_names,
    policy,
    output,
    *,
    deals=1000,
    seed=1,
    device="cpu",
    mcts=None,
    deal_offset=0,
    search_pool=None,
):
    from skat.__main__ import get_player

    if type(deals) is not int or deals < 1:
        raise ValueError("positive deal count required")
    if type(deal_offset) is not int or deal_offset < 0:
        raise ValueError("deal offset must be a nonnegative integer")
    if not policy:
        raise ValueError(
            "evaluation requires --policy for the competent opponent panel"
        )
    path = Path(output)
    if path.exists():
        raise ValueError("use a new results path; existing comparisons are preserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    mcts = mcts or MCTSConfig()
    import torch

    torch.set_num_threads(1)
    rows = []
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(str(source.relative_to(Path(__file__).parent)).encode())
        source_hash.update(source.read_bytes())
    manifest = dict(
        rules_schema=RULES_SCHEMA,
        observation_schema=OBSERVATION_SCHEMA,
        source_sha256=source_hash.hexdigest(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        seed=seed,
        deals=deals,
        deal_offset=deal_offset,
        agents=list(agent_names),
        mcts=asdict(mcts),
        search_workers=search_pool.workers if search_pool is not None else 1,
        mcts_rng_scheme="independent-world-v1",
        device=device,
        policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
    )

    def write():
        summaries = {
            name: summarize([r for r in rows if r["agent"] == name])
            for name in agent_names
        }
        comparisons = {
            f"{a} -> {b}": paired_comparison(
                [r for r in rows if r["agent"] == a],
                [r for r in rows if r["agent"] == b],
                seed,
            )
            for i, a in enumerate(agent_names)
            for b in agent_names[i + 1 :]
        }
        panels = {
            panel: {
                name: summarize(
                    [r for r in rows if r["agent"] == name and r["panel"] == panel]
                )
                for name in agent_names
            }
            for panel in ("random", "policy")
        }
        document = dict(
            manifest=manifest,
            summary=summaries,
            panels=panels,
            comparisons=comparisons,
            outcomes=rows,
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(path)
        return document

    try:
        for name in agent_names:
            for panel in ("random", "policy"):
                for seat in range(3):
                    agents = [
                        get_player(
                            name if s == seat else panel,
                            seed=seed + s,
                            policy=policy,
                            device=device,
                            mcts=mcts,
                            search_pool=search_pool,
                        )
                        for s in range(3)
                    ]
                    for i in range(deals):
                        deal_seed = stage_seed(seed, f"eval-deal-{i + deal_offset}")
                        for s, agent in enumerate(agents):
                            agent.rng = Random(stage_seed(deal_seed, f"seat-{s}"))
                        started = time.perf_counter()
                        r = Round(
                            dealer=(i + deal_offset) % 3,
                            agents=agents,
                            seed=deal_seed,
                            start=False,
                        )
                        won = r.start()
                        elapsed = time.perf_counter() - started
                        candidate_win = (
                            None
                            if won is None
                            else (won if r.solo_player_id == seat else not won)
                        )
                        rows.append(
                            dict(
                                agent=name,
                                panel=panel,
                                seat=seat,
                                seed=deal_seed,
                                soloist=None if won is None else r.solo_player_id,
                                won=candidate_win,
                                official_points=r.result.score
                                if r.solo_player_id == seat
                                else 0,
                                result=asdict(r.result),
                                seconds=elapsed,
                            )
                        )
                        if (i + 1) % 100 == 0:
                            print(
                                f"{name}/{panel}/seat{seat}: {i + 1}/{deals}",
                                flush=True,
                            )
                    write()
        report = write()
    finally:
        if rows:
            write()
    for name, summary in report["summary"].items():
        print(f"{name}: {summary}", flush=True)
    return report
