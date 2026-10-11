import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skat.agents.distill.checkpoint import load_policy, save_policy
from skat.agents.distill.loss import policy_loss
from skat.agents.distill.samples import ShardWriter, load_shards
from skat.agents.distill.targets import visit_targets
from skat.agents.distill.train import split_by_group, train
from skat.agents.mcts import MCTSAgent, RecordingMCTSAgent
from skat.agents.policy import PolicyAgent
from skat.agents.random import RandomAgent
from skat.card import Card
from skat.config import TrainConfig
from skat.game import Round, Tournament
from skat.models.card_transformer import CardTransformer
from skat.models.observation import INPUT_SIZE
from skat.rules import GameDeclaration


def test_illegal_logits_have_zero_gradient_and_do_not_change_loss():
    logits = torch.randn(2, 32, requires_grad=True)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[:, [1, 3]] = True
    target = visit_targets(torch.ones(2, 32), mask)
    first = policy_loss(logits, target, mask)
    altered = logits.detach().clone().masked_fill(~mask, 1e8).requires_grad_()
    assert first.item() == policy_loss(altered, target, mask).item()
    first.backward()
    assert not logits.grad[~mask].any()
    assert logits.grad[mask].abs().sum() > 0


def test_visit_targets_reject_empty_rows():
    with pytest.raises(ValueError):
        visit_targets(torch.zeros(1, 32), torch.ones(1, 32, dtype=torch.bool))


def rows(writer, count=2):
    moves = [Card(0, 0), Card(0, 1)]
    for _ in range(count):
        writer.add(
            np.zeros(INPUT_SIZE), {moves[0]: [2, 1], moves[1]: [1, 0]}, moves, moves[0]
        )


def test_group_split_keeps_a_round_crossing_shards_together(tmp_path):
    with ShardWriter(tmp_path, shard_size=2) as writer:
        writer.new_round()
        rows(writer, 3)
        writer.new_round()
        rows(writer, 3)
    data = load_shards(tmp_path)
    assert data.groups.tolist() == [0, 0, 0, 1, 1, 1]
    training, validation = split_by_group(data.groups)
    assert set(data.groups[training]).isdisjoint(data.groups[validation])
    assert len(training) == len(validation) == 3
    with pytest.raises(ValueError):
        ShardWriter(tmp_path)


def test_incompatible_shard_is_rejected(tmp_path):
    with ShardWriter(tmp_path) as writer:
        writer.new_round()
        rows(writer)
    path = next(tmp_path.glob("*.npz"))
    with np.load(path) as f:
        content = dict(f)
    content["observation_schema"] = "old-hidden-skat"
    np.savez(path, **content)
    with pytest.raises(RuntimeError):
        load_shards(tmp_path)


def test_checkpoint_roundtrip_and_strict_rejection(tmp_path):
    torch.set_num_threads(1)
    model = CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    model.eval()
    x = torch.randn(2, INPUT_SIZE)
    path = tmp_path / "policy.pt"
    save_policy(path, model, {"sample": 42})
    restored, metadata = load_policy(path)
    assert metadata["sample"] == 42
    assert not restored.training
    assert torch.equal(model(x), restored(x))
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError):
        load_policy(path)
    save_policy(path, model)
    saved = torch.load(path, weights_only=True)
    saved["rules_schema"] = "old-rules"
    torch.save(saved, path)
    with pytest.raises(ValueError):
        load_policy(path)


def test_transformer_features_match_real_observation():
    torch.set_num_threads(1)
    r = Round(
        start=False,
        skip_bidding=True,
        solo_player_id=0,
        hand_game=True,
        declare_game=GameDeclaration("grand", hand=True, ouvert=True),
        seed=7,
    )
    r.prepare()
    model = CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    x = r.get_state_t(1)
    tokens, context = model.features_of(x)
    assert tokens.shape == (1, 32, 17) and context.shape == (1, 25)
    assert tokens[0, :, 13:16].sum() == 10  # visible soloist hand
    assert tokens[0, :, 16].sum() == 0  # unknown skat
    assert model(x).shape == (1, 32)
    model.eval()
    assert torch.equal(model(x), model(x))


def test_recording_does_not_change_play(tmp_path):
    with ShardWriter(tmp_path, shard_size=5) as writer:

        def play(record):
            agents = [
                RecordingMCTSAgent(writer, simulations=20, determinizations=2, seed=s)
                if record
                else MCTSAgent(20, 2, seed=s)
                for s in range(3)
            ]
            r = Round(
                start=False, skip_bidding=True, solo_player_id=0, agents=agents, seed=4
            )
            r.start()
            return r.result, [
                (t.winner, tuple(t.buffer)) for t in r.trick_history.buffer
            ]

        assert play(False) == play(True)
    data = load_shards(tmp_path)
    assert len(np.unique(data.groups)) == 1
    assert (data.masks.sum(1) > 1).all()


def test_collect_train_serve_pipeline(tmp_path):
    data_dir = tmp_path / "data"
    with ShardWriter(data_dir, shard_size=7) as writer:
        agents = [
            RecordingMCTSAgent(writer, simulations=20, determinizations=2, seed=s)
            for s in range(3)
        ]
        Tournament(4, agents=agents, seed=9).start()
    output = tmp_path / "policy.pt"
    config = TrainConfig(epochs=1, batch_size=32, d_model=16, nhead=2, num_layers=1)
    report = train(data_dir, output, config=config)
    assert report["train_rows"] > 0 and report["val_rows"] > 0
    agents = [PolicyAgent(output), RandomAgent(1), RandomAgent(2)]
    t = Tournament(3, agents=agents, seed=10)
    t.start()
    assert len(t.round_records) == 3
    with pytest.raises(FileNotFoundError):
        PolicyAgent(tmp_path / "missing.pt")


@pytest.mark.parametrize("groups", [np.array([0.5]), np.array([-1])])
def test_shards_reject_invalid_round_identifiers(tmp_path, groups):
    from skat.card import Card

    with ShardWriter(tmp_path) as writer:
        writer.new_round()
        card = Card(0, 0)
        writer.add(np.zeros(INPUT_SIZE), {card: (1, 1.0)}, [card], card)
    path = next(tmp_path.glob("*.npz"))
    with np.load(path) as shard:
        data = {key: shard[key] for key in shard.files}
    data["groups"] = groups
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        load_shards(tmp_path)
