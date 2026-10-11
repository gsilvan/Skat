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
    with pytest.raises(ValueError):
        evaluate(["random"], checkpoint, path, deals=2)


def test_collect_cli_writes_samples_for_an_initially_empty_recorder(tmp_path):
    pytest.importorskip("torch")
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
