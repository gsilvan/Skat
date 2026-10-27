from types import SimpleNamespace

import numpy as np
import pytest

from skat.__main__ import get_player
from skat.agents.bidding import covering_game
from skat.agents.declaring import press_for_game
from skat.agents.neural_declaring.features import declaration_state
from skat.agents.neural_declaring.inference import NeuralDeclarer, candidates
from skat.agents.random import RandomAgent
from skat.game import Round
from skat.games.null import Null
from skat.hand import FULL_HAND


def view(bid=0):
    return SimpleNamespace(
        hand=tuple(FULL_HAND[:12]),
        skat=tuple(FULL_HAND[10:12]),
        highest_bid=bid,
        seat_id=0,
        dealer=2,
        hand_game=False,
    )


class NullScorer:
    def probabilities(self, features):
        values = np.stack(features)
        return values[:, 104] + values[:, 31] * 0.1


def test_features_distinguish_kept_pressed_and_original_skat():
    game = Null()
    first = declaration_state(
        FULL_HAND[:10], FULL_HAND[10:12], FULL_HAND[10:12], game, 0, 2
    )
    second = declaration_state(
        (*FULL_HAND[:9], FULL_HAND[10]),
        (FULL_HAND[9], FULL_HAND[11]),
        FULL_HAND[10:12],
        game,
        0,
        2,
    )
    assert first.shape == (105,)
    assert first[:32].sum() == 10 and first[32:64].sum() == 2
    assert first[64:96].sum() == 2
    assert not np.array_equal(first, second)
    with pytest.raises(ValueError):
        declaration_state(
            FULL_HAND[:10], FULL_HAND[10:12], FULL_HAND[20:22], game, 0, 2
        )


def test_joint_choice_filters_games_below_won_bid():
    low = view()
    assert len(list(candidates(low))) == 396
    choice = NeuralDeclarer(NullScorer()).choose_after_pickup(low)
    assert isinstance(choice.game, Null)
    high = view(48)
    options = list(candidates(high))
    assert options and all(not isinstance(option.game, Null) for option, _ in options)
    choice = NeuralDeclarer(NullScorer()).choose_after_pickup(high)
    assert choice.game.spielwert(high.hand) >= high.highest_bid


def test_heuristic_baseline_and_round_lifecycle():
    baseline = RandomAgent()
    baseline.set_state(view(18))
    game = covering_game(set(baseline.state.hand), 18)
    assert baseline.press_skat() == tuple(press_for_game(baseline.state.hand, game))
    assert baseline.declare_game().declaration == game.declaration
    with pytest.raises(ValueError, match="prior press"):
        baseline.declare_game()
    assert get_player("random").declaration_label == "heuristic"
    with pytest.raises(ValueError, match="declaration-model"):
        get_player("random", declaration="neural")
    agents = [RandomAgent(i) for i in range(3)]
    agents[0].configure_declaration("neural", NullScorer())
    round_ = Round(
        seed=42, agents=agents, skip_bidding=True, solo_player_id=0, start=False
    )
    round_.prepare()
    assert round_.game.spielwert(round_.original_solo_cards) >= round_.highest_bid


@pytest.mark.parametrize("neural", [False, True])
def test_manual_discard_override_reconciles_declaration_with_original_skat(neural):
    class RecordingScorer:
        def __init__(self):
            self.features = []

        def probabilities(self, features):
            values = np.stack(features)
            self.features.append(values.copy())
            # Prefer Grand when the diamond seven is kept, otherwise Null.
            return values[:, 103] * values[:, 0] + values[:, 104] * (1 - values[:, 0])

    scorer = RecordingScorer()
    agent = RandomAgent(seed=42)
    if neural:
        agent.configure_declaration("neural", scorer)
    original = view(18)
    agent.set_state(original)
    suggested = agent.press_skat()
    cached = agent._declaration_choice
    agent.accept_pressed(original, suggested[::-1])
    assert agent._declaration_choice is cached
    actual = (FULL_HAND[0], FULL_HAND[5])
    if set(actual) == set(suggested):
        actual = (FULL_HAND[1], FULL_HAND[5])
    assert set(actual) != set(suggested)
    agent.accept_pressed(original, actual)
    assert set(agent._declaration_choice.pressed) == set(actual)
    assert original.hand == tuple(FULL_HAND[:12])
    assert original.skat == tuple(FULL_HAND[10:12])
    if neural:
        assert len(scorer.features) == 2
        fixed = scorer.features[-1]
        expected_kept = np.zeros(32)
        expected_kept[[c.np_index for c in original.hand if c not in actual]] = 1
        expected_pressed = np.zeros(32)
        expected_pressed[[c.np_index for c in actual]] = 1
        expected_skat = np.zeros(32)
        expected_skat[[c.np_index for c in original.skat]] = 1
        assert np.all(fixed[:, :32] == expected_kept)
        assert np.all(fixed[:, 32:64] == expected_pressed)
        assert np.all(fixed[:, 64:96] == expected_skat)
        assert agent._declaration_choice.game.declaration.kind == (
            "null" if FULL_HAND[0] in actual else "grand"
        )
    assert agent.declare_game().spielwert(original.hand) >= original.highest_bid


def test_fixed_discard_candidates_validate_pair_and_preserve_covering_games():
    original = view(48)
    pair = (FULL_HAND[1], FULL_HAND[3])
    options = list(candidates(original, pressed=pair))
    assert options and len(options) <= 6
    assert all(
        choice.pressed == pair and choice.game.spielwert(original.hand) >= 48
        for choice, _ in options
    )
    for pair in [(FULL_HAND[0],), (FULL_HAND[0],) * 2, (FULL_HAND[0], FULL_HAND[20])]:
        with pytest.raises(ValueError, match="two distinct"):
            list(candidates(original, pressed=pair))


def test_checkpoint_is_separate_and_batched(tmp_path):
    torch = pytest.importorskip("torch")
    from skat.agents.neural_declaring.checkpoint import (
        load_declaration,
        save_declaration,
    )
    from skat.agents.neural_declaring.inference import DeclarationScorer
    from skat.models.declaring import DeclarationNet

    torch.set_num_threads(1)
    path = tmp_path / "declaration.pt"
    save_declaration(path, DeclarationNet(d_model=16, nhead=2, num_layers=1))
    model, temperature, _ = load_declaration(path)
    assert temperature == 1
    scorer = DeclarationScorer(path)
    features = [state for _, state in list(candidates(view()))[:3]]
    assert scorer.probabilities(features).shape == (3,)
    assert model(torch.from_numpy(np.stack(features))).shape == (3,)


def test_import_samples_legal_actions_without_using_recorded_outcome():
    from iss.declaration import sampled_choices

    record = SimpleNamespace(
        original=(
            tuple(FULL_HAND[:10]),
            tuple(FULL_HAND[10:20]),
            tuple(FULL_HAND[20:30]),
            tuple(FULL_HAND[30:32]),
        ),
        soloist=0,
        bid=24,
        identifier="synthetic-deal",
    )
    choices = sampled_choices(record, 4, 7)
    cards = set(record.original[0]) | set(record.original[3])

    def signature(values):
        return [
            (tuple(card.np_index for card in choice.pressed), choice.game.declaration)
            for choice in values
        ]

    assert signature(choices) == signature(sampled_choices(record, 4, 7))
    budgeted = sampled_choices(record, 4, 7, candidate_budget=2)
    assert len(budgeted) == 2
    assert signature(budgeted) == signature(sampled_choices(record, 4, 7, 2))
    assert choices
    assert all(
        len(set(choice.pressed)) == 2
        and set(choice.pressed) <= cards
        and choice.game.spielwert(cards) >= record.bid
        for choice in choices
    )
    assert all(not isinstance(choice.game, Null) for choice in choices)
