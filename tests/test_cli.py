import json
import subprocess
import sys

import pytest

from skat.__main__ import get_args, main
from skat.config import load_config
from skat.evaluation import paired_comparison, wilson


def test_play_without_torch_or_sgf(tmp_path):
    # Block optional imports, even when the current interpreter has them installed.
    code = """
import sys
class BlockOptional:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'tensorboard', 'pysgf', 'flask'):
            raise ImportError('optional import blocked: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from skat.__main__ import main
main(['play', '--rounds', '3'])
main(['play', '--rounds', '1', '--agents', 'mcts', 'random', 'random'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "3 rounds" in result.stdout


def test_cli_separates_training_from_play():
    args = get_args(["play"])
    assert not hasattr(args, "epochs") and not hasattr(args, "batch_size")
    with pytest.raises(SystemExit):
        get_args(["play", "--epsilon", "0.9"])
    with pytest.raises(SystemExit):
        main(["play", "--agents", "policy", "random", "random"])


def test_config_validation_and_defaults(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[mcts]\nsimulations=80\ndeterminizations=4\n[train]\nepochs=2\n")
    mcts, training = load_config(path)
    assert mcts.simulations == 80 and training.epochs == 2
    path.write_text("[mcts]\nunknown=1\n")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("[mcts]\nsimulations=1\ndeterminizations=2\n")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("shared", [False, True])
def test_play_loads_checkpoint_per_seat_with_shared_fallback(
    tmp_path, monkeypatch, shared
):
    torch = pytest.importorskip("torch")
    import skat.__main__ as cli
    from skat.agents.distill.checkpoint import save_policy
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    paths = [tmp_path / f"seat-{seat}.pt" for seat in range(2)]
    for seat, path in enumerate(paths):
        save_policy(
            path,
            CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32),
            {"seat_model": seat},
        )
    players = []
    original = cli.get_player

    def player(name, **kwargs):
        agent = original(name, **kwargs)
        players.append(agent)
        return agent

    monkeypatch.setattr(cli, "get_player", player)
    args = [
        "play",
        "--rounds",
        "2",
        "--agents",
        "policy",
        "policy",
        "random",
        "--policy0",
        str(paths[0]),
    ]
    args += ["--policy" if shared else "--policy1", str(paths[1])]
    cli.main(args)
    assert [p.metadata["seat_model"] for p in players[:2]] == [0, 1]
    # A model at seat 0 must not silently fill a missing model at seat 1.
    with pytest.raises(SystemExit, match="checkpoint"):
        cli.main(
            [
                "play",
                "--rounds",
                "0",
                "--agents",
                "policy",
                "policy",
                "random",
                "--policy0",
                str(paths[0]),
            ]
        )


def test_paired_results_and_missing_deals():
    first = [dict(panel="random", seat=0, seed=i, won=i % 2 == 0) for i in range(8)]
    second = [dict(r, won=True) for r in first]
    report = paired_comparison(first, second)
    assert report["delta"] == 0.5 and report["improved"] == 4
    assert report["paired"] == 8 and report["deal_groups"] == 8
    assert report["deal_bootstrap_interval"][0] > 0
    assert (
        paired_comparison(first, [dict(r, seed=r["seed"] + 100) for r in second])[
            "paired"
        ]
        == 0
    )
    assert wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]


def test_small_evaluation_retains_six_outcomes_per_deal(tmp_path):
    torch = pytest.importorskip("torch")
    from skat.agents.distill.checkpoint import save_policy
    from skat.config import MCTSConfig
    from skat.evaluation import evaluate
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    checkpoint = tmp_path / "test.pt"
    save_policy(
        checkpoint,
        CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32),
    )
    path = tmp_path / "results.json"
    report = evaluate(
        ["random", "mcts"], checkpoint, path, deals=2, mcts=MCTSConfig(12, 2)
    )
    assert len(report["outcomes"]) == 24
    assert len(json.loads(path.read_text())["outcomes"]) == 24
    assert report["comparisons"]["random -> mcts"]["paired"] <= 12
    assert (
        report["manifest"]["opponent_policy_sha256"]
        == report["manifest"]["policy_sha256"]
    )
    with pytest.raises(ValueError):
        evaluate(["random"], checkpoint, path, deals=2)


def test_evaluation_uses_fixed_opponent_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import hashlib

    import skat.__main__ as cli
    from skat.agents.distill.checkpoint import save_policy
    from skat.evaluation import evaluate
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    candidate, opponent = tmp_path / "candidate.pt", tmp_path / "opponent.pt"
    for seed, path in enumerate((candidate, opponent)):
        torch.manual_seed(seed)
        save_policy(
            path, CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
        )
    calls = []
    original = cli.get_player

    def player(name, **kwargs):
        calls.append((name, kwargs["policy"]))
        return original(name, **kwargs)

    monkeypatch.setattr(cli, "get_player", player)
    report = evaluate(
        ["policy"],
        candidate,
        tmp_path / "results.json",
        deals=1,
        opponent_policy=opponent,
    )
    # For each seat the candidate gets its own model; two policy opponents get
    # the fixed reference. Random opponents never load a checkpoint.
    policy_calls = [path for name, path in calls if name == "policy"]
    assert policy_calls.count(candidate) == 6
    assert policy_calls.count(opponent) == 6
    manifest = report["manifest"]
    assert (
        manifest["policy_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    )
    assert (
        manifest["opponent_policy_sha256"]
        == hashlib.sha256(opponent.read_bytes()).hexdigest()
    )
    args = cli.get_args(
        ["evaluate", "--out", "out.json", "--opponent-policy", str(opponent)]
    )
    assert args.opponent_policy == str(opponent)
    reference = cli.get_args(
        ["evaluate", "--out", "out.json", "--reference-policy", str(opponent)]
    )
    assert reference.opponent_policy == args.opponent_policy


def test_collect_cli_writes_samples_for_an_initially_empty_recorder(tmp_path):
    from skat.agents.distill.samples import load_shards

    settings = tmp_path / "quick.toml"
    settings.write_text("[mcts]\nsimulations=12\ndeterminizations=2\n")
    data = tmp_path / "data"
    main(
        [
            "collect",
            "--out",
            str(data),
            "--rounds",
            "3",
            "--seed",
            "9",
            "--config",
            str(settings),
        ]
    )
    samples = load_shards(data)
    assert len(samples) > 0 and len(set(samples.groups)) >= 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "fast"])
def test_training_config_rejects_invalid_learning_rate(value):
    from skat.config import TrainConfig

    with pytest.raises(ValueError):
        TrainConfig(learning_rate=value)


def test_evaluation_chunks_preserve_common_deals_and_seats(tmp_path):
    torch = pytest.importorskip("torch")
    from skat.agents.distill.checkpoint import save_policy
    from skat.evaluation import evaluate
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    checkpoint = tmp_path / "test.pt"
    save_policy(
        checkpoint,
        CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32),
    )
    whole = evaluate(["random"], checkpoint, tmp_path / "whole.json", deals=2, seed=12)
    parts = [
        evaluate(
            ["random"],
            checkpoint,
            tmp_path / f"part-{i}.json",
            deals=1,
            seed=12,
            deal_offset=i,
        )
        for i in range(2)
    ]

    def outcomes(report):
        return {
            (r["panel"], r["seat"], r["seed"]): (r["won"], r["result"])
            for r in report["outcomes"]
        }

    combined = {key: value for part in parts for key, value in outcomes(part).items()}
    assert outcomes(whole) == combined
