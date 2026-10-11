from random import Random

import pytest

from skat.agents.mcts import MCTSAgent
from skat.agents.random import RandomAgent
from skat.agents.search.determinize import SKAT_SLOT, deal_unseen, infer_voids
from skat.card import Card
from skat.game import Round
from skat.rules import GameDeclaration


def test_capacity_prior_and_voids_are_respected():
    cards = [Card(0, i) for i in range(6)]
    voids = {1: frozenset(cards[:2])}
    for seed in range(30):
        deal = deal_unseen(cards, {1: 2, 2: 3, SKAT_SLOT: 1}, voids, Random(seed))
        assert [len(deal[s]) for s in (1, 2, SKAT_SLOT)] == [2, 3, 1]
        assert not set(deal[1]) & voids[1]
        assert set(c for hand in deal.values() for c in hand) == set(cards)
    with pytest.raises(ValueError):
        deal_unseen(cards, {1: 6}, {1: frozenset(cards)}, Random(1))


def test_uniform_feasible_world_sampling_counts_completions():
    # Of three valid two-card hands for seat 1, {a,b} is forbidden by the void.
    a, b, c, d = [Card(0, i) for i in range(4)]
    counts = {}
    rng = Random(5)
    for _ in range(3000):
        deal = deal_unseen([a, b, c, d], {1: 2, 2: 2}, {2: frozenset([d])}, rng)
        key = tuple(sorted(x.np_index for x in deal[1]))
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 3
    assert all(900 <= n <= 1100 for n in counts.values())


def make_round(d, seat=0, seed=4):
    agents = [RandomAgent(seed=i) for i in range(3)]
    agents[seat] = MCTSAgent(30, 3, seed=seed)
    r = Round(
        start=False,
        solo_player_id=0,
        skip_bidding=True,
        hand_game=d.hand,
        declare_game=d,
        agents=agents,
        seed=seed,
    )
    r.prepare()
    while r.next_player != seat:
        r.step()
    return r, agents[seat]


@pytest.mark.parametrize(
    "d",
    [
        GameDeclaration("suit", 3),
        GameDeclaration("grand", hand=True),
        GameDeclaration("null", hand=True),
        GameDeclaration("grand", hand=True, ouvert=True),
    ],
)
def test_search_rollouts_match_engine_for_every_game(d):
    r, agent = make_round(d)
    world = agent.determinize()
    r.player[0].hand[:] = world.hands[0]
    r.player[1].hand[:] = world.hands[1]
    r.player[2].hand[:] = world.hands[2]
    # Soloist knows the picked skat in non-Hand cases; sample replaces hidden
    # opponent holdings only. For Hand, reconstruct the sampled skat too.
    played = {c for t in (*r.trick_history.buffer, r.game.trick) for _, c in t.buffer}
    used = {c for hand in world.hands for c in hand} | played
    from skat.hand import FULL_HAND, Hand

    r.skat = [c for c in FULL_HAND if c not in used]
    for seat, p in enumerate(r.player):
        history_cards = [
            c for t in r.trick_history.buffer if t.winner == seat for _, c in t.buffer
        ]
        p.trick_stack = Hand(history_cards + (r.skat if seat == 0 else []))
    r.original_solo_cards = tuple(
        world.hands[0]
        + r.skat
        + [
            c
            for t in (*r.trick_history.buffer, r.game.trick)
            for seat, c in t.buffer
            if seat == 0
        ]
    )
    while not world.is_terminal:
        card = min(world.legal_moves(), key=lambda c: c.np_index)
        r.play_card(r.next_player, card)
        world.apply(card)
    assert r.result == world.result()


def test_hidden_changes_do_not_change_seeded_defender_search():
    r, agent = make_round(GameDeclaration("suit", 3, hand=True), seat=1)
    before = agent.state.observation.copy()
    agent.rng.seed(123)
    tally = agent.tally(
        sorted(r.player[1].valid_moves(r.game.trick), key=lambda c: c.np_index)
    )
    a, b = r.player[0].hand[0], r.skat[0]
    r.player[0].hand[0], r.skat[0] = b, a
    r._refresh()
    assert (before == agent.state.observation).all()
    agent.rng.seed(123)
    assert tally == agent.tally(
        sorted(r.player[1].valid_moves(r.game.trick), key=lambda c: c.np_index)
    )


def test_mcts_is_reproducible_and_never_removes_a_card():
    r, agent = make_round(GameDeclaration("suit", 3))
    moves = r.player[0].valid_moves(r.game.trick)
    before = list(r.player[0].hand)
    agent.rng.seed(17)
    first = agent.search(moves)
    agent.rng.seed(17)
    assert first == agent.search(moves)
    assert before == list(r.player[0].hand)
    assert sum(stats[0] for stats in agent.tally(list(moves)).values()) == 30


def test_void_inference_uses_game_follow_suit_rules():
    game = GameDeclaration("grand").make_game()
    trick = game.trick
    trick.append(0, Card(0, 7))
    trick.append(1, Card(0, 3))  # printed diamonds Jack is trump, fails diamonds
    voids = infer_voids([trick], trick)
    assert Card(0, 0) in voids[1]
    assert Card(0, 3) not in voids[1]
