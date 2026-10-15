from pathlib import Path
from random import Random

import numpy as np
import pytest

from iss.belief import BeliefWriter, examples_from_record
from iss.protocol import parse_record
from skat.agents.belief.features import INPUT_SIZE, KNOWN, belief_input, locations
from skat.agents.search.determinize import SKAT_SLOT, WeightedDealer, deal_unseen
from skat.agents.search.parallel import SearchPool
from skat.card import Card
from skat.game import Round
from skat.rules import GameDeclaration


def test_weighted_worlds_respect_global_capacity_and_exact_mass():
    cards = [Card(0, rank) for rank in range(4)]
    capacities = {1: 2, 2: 1, SKAT_SLOT: 1}
    voids = {2: frozenset([cards[3]])}
    weights = {
        card.np_index: {1: 9.0 if card == cards[0] else 1.0, 2: 1.0, SKAT_SLOT: 1.0}
        for card in cards
    }
    dealer = WeightedDealer(cards, capacities, voids, weights)
    assert np.isclose(
        sum(
            np.exp(dealer.log_probability(deal))
            for deal in (
                {1: [a, b], 2: [c], SKAT_SLOT: [d]}
                for a in cards
                for b in cards
                if a.np_index < b.np_index
                for c in cards
                if c not in (a, b)
                for d in cards
                if d not in (a, b, c)
            )
        ),
        1.0,
    )
    rng = Random(42)
    sampled = [dealer.sample(rng) for _ in range(200)]
    assert all(cards[3] not in world[2] for world in sampled)
    assert sum(cards[0] in world[1] for world in sampled) > 140
    uniform = WeightedDealer(
        cards,
        capacities,
        voids,
        {card.np_index: {slot: 1.0 for slot in capacities} for card in cards},
    )
    assert np.isclose(uniform.log_probability(sampled[0]), -np.log(9))


def test_weighted_sampler_rejects_impossible_worlds():
    cards = [Card(0, 0), Card(0, 1)]
    weights = {card.np_index: {1: 1.0, SKAT_SLOT: 1.0} for card in cards}
    with pytest.raises(ValueError, match="no complete world"):
        WeightedDealer(cards, {1: 2, SKAT_SLOT: 0}, {1: frozenset(cards)}, weights)
    with pytest.raises(ValueError, match="no complete world"):
        deal_unseen(cards, {1: 2, SKAT_SLOT: 0}, {1: frozenset(cards)}, Random(1))


def test_iss_labels_exclude_visible_cards_and_model_trains(tmp_path):
    pytest.importorskip("torch")
    from skat.agents.belief.inference import BeliefSampler
    from skat.agents.belief.train import train
    from skat.agents.mcts import MCTSAgent

    lines = Path("iss/tests/fixtures/card-games.sgf").read_text().splitlines()
    records = [parse_record(lines[index]) for index in (12, 13, 14)]
    writer = BeliefWriter(tmp_path / "corpus", shard_rows=3)
    writer.directory.mkdir()
    for split, record in zip(("train", "val", "test"), records):
        writer.add(split, examples_from_record(record))
    writer.finish({"fixture": True})
    report = train(
        writer.directory,
        tmp_path / "belief.pt",
        epochs=1,
        batch_size=2,
        d_model=16,
        num_layers=1,
        calibration_size=2,
    )
    assert np.isfinite(report["test_card_nll"])
    sampler = BeliefSampler(tmp_path / "belief.pt")
    round_ = Round(
        start=False,
        solo_player_id=0,
        skip_bidding=True,
        hand_game=True,
        declare_game=GameDeclaration("grand", hand=True),
    )
    round_.prepare()
    agent = MCTSAgent(simulations=4, determinizations=2, sampler=sampler)
    agent.set_state(round_.view(round_.next_player))
    world = agent.determinize(Random(4))
    assert sum(map(len, world.hands)) == 30
    moves = round_.player[round_.next_player].valid_moves(round_.game.trick)
    assert agent.search(moves) in moves
    agent.rng.seed(19)
    serial = agent.tally(moves)
    with SearchPool(2, capacity=2) as pool:
        agent.search_pool = pool
        agent.rng.seed(19)
        assert agent.tally(moves) == serial
    for declaration in (
        GameDeclaration("grand", hand=True, ouvert=True),
        GameDeclaration("null", hand=True),
        GameDeclaration("suit", 2),
    ):
        extra = Round(
            start=False,
            solo_player_id=0,
            skip_bidding=True,
            hand_game=declaration.hand,
            declare_game=declaration,
        )
        extra.prepare()
        agent.set_state(extra.view(1))
        world = agent.determinize(Random(11))
        assert sum(map(len, world.hands)) == 30
        assert set(world.hands[1]) == set(extra.player[1].hand)


def test_iss_features_match_unseen_cards_without_future_information():
    record = parse_record(
        Path("iss/tests/fixtures/card-games.sgf").read_text().splitlines()[12]
    )
    from iss.protocol import ReplayRound

    round_ = ReplayRound(
        dealer=2,
        start=False,
        skip_bidding=True,
        solo_player_id=record.soloist,
        declare_game=record.contract,
        hand_game=True,
        replay=True,
        initial_cards=[*record.hands, record.skat],
    )
    round_.prepare()
    view = round_.view(round_.next_player)
    features = belief_input(view)
    labels = locations(view, [player.hand for player in round_.player], round_.skat)
    assert features.shape == (INPUT_SIZE,)
    assert sum(labels == KNOWN) == len(view.hand) + len(view.skat)
    future_card = record.actions[0][1]
    assert features[-64:].sum() == 0
    if future_card in view.hand:
        assert labels[future_card.np_index] == KNOWN
    else:
        assert labels[future_card.np_index] != KNOWN
