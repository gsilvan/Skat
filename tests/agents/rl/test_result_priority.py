import pytest
import torch
from torch import nn

from skat.agents.rl.dqn import DQN
from skat.agents.rl.result_q import secured_victory


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(240, 32)

    def forward(self, states):
        return self.layer(states)


def make_dqn(tmp_path):
    return DQN(
        device="cpu",
        model=TinyNet,
        result_priority=True,
        batch_size=1,
        learn_start=1,
        seed=1,
        model_path=str(tmp_path / "model.pt"),
        log_dir=str(tmp_path / "logs"),
    )


def observation(points, solo=True, null=False):
    s = torch.zeros(1, 240)
    s[0, 38], s[0, 42], s[0, 37] = float(solo), points / 120, float(null)
    return s


@pytest.mark.parametrize(
    "points,solo,expected",
    [(60, True, 0), (61, True, 1), (59, False, 0), (60, False, 1)],
)
def test_bonus_cannot_buy_a_lower_win_prediction(tmp_path, points, solo, expected):
    dqn = make_dqn(tmp_path)
    with torch.no_grad():
        for net in (dqn.policy_net.win, dqn.policy_net.bonus):
            net.layer.weight.zero_()
            net.layer.bias.zero_()
        dqn.policy_net.win.layer.bias[0] = 1
        dqn.policy_net.bonus.layer.bias[1] = 1000
    mask = torch.zeros(32, dtype=torch.bool)
    mask[:2] = True
    assert (
        dqn.select_action(observation(points, solo), mask, explore=False).argmax()
        == expected
    )
    assert (
        dqn.select_action(
            observation(points, solo), mask, explore=False, allow_bonus=False
        ).argmax()
        == 0
    )
    assert (
        dqn.select_action(
            observation(points, solo, null=True), mask, explore=False
        ).argmax()
        == 0
    )
    dqn.close()


def test_bonus_gate_reads_real_egocentric_observations():
    from unittest.mock import patch

    from skat.game import Round
    from skat.games.suit import SuitGame

    round_ = Round(start=False, solo_player_id=0, declare_game=SuitGame(3))
    with patch.object(
        Round,
        "observer_points",
        side_effect=lambda seat: (61, 59) if seat == 0 else (59, 61),
    ):
        assert [
            secured_victory(round_.get_state_t(seat)).item() for seat in range(3)
        ] == [True, False, False]
    with patch.object(Round, "observer_points", return_value=(60, 60)):
        assert [
            secured_victory(round_.get_state_t(seat)).item() for seat in range(3)
        ] == [False, True, True]


def test_primary_bootstrap_stops_at_secured_results(tmp_path):
    dqn = make_dqn(tmp_path)
    with torch.no_grad():
        dqn.target_net.win.layer.weight.zero_()
        dqn.target_net.win.layer.bias.fill_(100)
    states = torch.cat([observation(61), observation(50), observation(40)])
    states[2, 43] = 60 / 120
    masks = torch.ones(3, 32, dtype=torch.bool)
    assert dqn.bootstrap(states, masks).tolist() == [0.0, 100.0, 0.0]
    dqn.close()


def test_terminal_only_batch_and_bonus_gradients_do_not_change_win_net(tmp_path):
    dqn = make_dqn(tmp_path)
    s = observation(61)
    action = torch.zeros(1, 32, dtype=torch.long)
    action[0, 0] = 1
    with torch.no_grad():
        for net in (dqn.policy_net.win, dqn.policy_net.bonus):
            net.layer.weight.zero_()
            net.layer.bias.zero_()
    dqn.replay_buffer.push(
        state=s,
        action=action,
        next_state=None,
        next_mask=None,
        reward=torch.tensor([[0.0, 0.5]]),
    )
    before = {k: v.clone() for k, v in dqn.policy_net.win.state_dict().items()}
    dqn.optimize_model()
    assert all(
        torch.equal(before[k], v) for k, v in dqn.policy_net.win.state_dict().items()
    )
    assert dqn.policy_net.bonus.layer.bias[0] > 0
    dqn.save()
    restored = make_dqn(tmp_path)
    assert all(
        torch.equal(v, restored.policy_net.state_dict()[k])
        for k, v in dqn.policy_net.state_dict().items()
    )
    restored.close()
    dqn.close()
