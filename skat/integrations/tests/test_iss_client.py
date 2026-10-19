"""Check live player-view translation against the engine and protocol envelopes."""

import os

import numpy as np

from skat.agents.random import RandomAgent
from skat.game import Round
from skat.hand import FULL_HAND
from skat.integrations.iss import choose, format_card, parse_snapshot
from skat.integrations.iss_client import ISSClient, ISSWire
from skat.integrations.iss_state import LiveGame
from skat.rules import GameDeclaration


def deal_string(cards, viewer):
    blocks = []
    for seat in range(3):
        blocks.append(
            ".".join(format_card(c) for c in cards[seat])
            if seat == viewer
            else ".".join(["??"] * 10)
        )
    blocks.append("??.??")
    return "|".join(blocks)


def test_player_observation_matches_engine_during_cardplay():
    cards = tuple(tuple(FULL_HAND[i : i + 10]) for i in (0, 10, 20)) + (
        tuple(FULL_HAND[30:]),
    )
    round_ = Round(
        dealer=2,
        skip_bidding=True,
        solo_player_id=0,
        declare_game=GameDeclaration("suit", 1, hand=True),
        hand_game=True,
        initial_cards=cards,
        start=False,
    )
    round_.prepare()
    states = [LiveGame(seat) for seat in range(3)]
    for state in states:
        state.apply(4, deal_string(cards, state.seat))
        state.phase, state.solo, state.to_move = 3, 0, 0
        state.apply(0, "HH")
    while not round_.is_finished:
        for state in states:
            _, _, view, _ = parse_snapshot(state.snapshot())
            np.testing.assert_array_equal(
                view.observation, round_.get_state(state.seat)
            )
        seat = round_.next_player
        card = sorted(
            round_.player[seat].valid_moves(round_.game.trick),
            key=lambda c: c.np_index,
        )[0]
        round_.play_card(seat, card)
        for state in states:
            state.apply(seat, format_card(card))


def test_python_client_responds_to_iss_server_envelopes():
    agent = RandomAgent(seed=1)
    client = ISSClient(agent, table="test")
    sent = []
    client.send = sent.append
    client.handle("create test bot 3 -1")
    assert sent == ["table test bot ready"]
    client.handle("table test bot start 1 other 240 bot 240 third 240")
    assert client.game.seat == 1
    cards = tuple(tuple(FULL_HAND[i : i + 10]) for i in (0, 10, 20)) + (
        tuple(FULL_HAND[30:]),
    )
    client.handle("table test bot play w " + deal_string(cards, 1) + " 240 240 240")
    assert sent[-1].startswith("table test bot play ")
    client.handle("table test bot end (;GM[Skat])")
    assert sent[-1] == "table test bot ready"


def test_pickup_declaration_and_iss_ten_conversion():
    cards = tuple(tuple(FULL_HAND[i : i + 10]) for i in (0, 10, 20)) + (
        tuple(FULL_HAND[30:]),
    )
    state = LiveGame(0)
    state.apply(4, deal_string(cards, 0))
    state.phase, state.solo, state.to_move = 3, 0, 0
    agent = RandomAgent(seed=1)
    assert choose(agent, state.snapshot()) == "s"
    state.apply(0, "s")
    state.apply(4, ".".join(format_card(c) for c in cards[3]))
    move = choose(agent, state.snapshot())
    assert len(move.split(".")) == 3
    state.apply(0, move)
    assert len(state.hand) == 10
    assert len(state.skat) == 2


def test_pickup_observation_matches_engine():
    cards = tuple(tuple(FULL_HAND[i : i + 10]) for i in (0, 10, 20)) + (
        tuple(FULL_HAND[30:]),
    )
    round_ = Round(
        dealer=2,
        skip_bidding=True,
        solo_player_id=0,
        declare_game=GameDeclaration("suit", 0),
        hand_game=False,
        initial_cards=cards,
        start=False,
    )
    round_.prepare()
    states = [LiveGame(seat) for seat in range(3)]
    for state in states:
        state.apply(4, deal_string(cards, state.seat))
        state.phase, state.solo, state.to_move = 3, 0, 0
        state.apply(0, "s")
        state.apply(
            4,
            ".".join(format_card(c) for c in cards[3]) if state.seat == 0 else "??.??",
        )
        discards = (
            ".".join(format_card(c) for c in round_.skat)
            if state.seat == 0
            else "??.??"
        )
        state.apply(0, "D." + discards)
        _, _, view, _ = parse_snapshot(state.snapshot())
        np.testing.assert_array_equal(view.observation, round_.get_state(state.seat))


def test_bidding_transitions_and_wire_buffering():
    state = LiveGame(1)
    cards = tuple(tuple(FULL_HAND[i : i + 10]) for i in (0, 10, 20)) + (
        tuple(FULL_HAND[30:]),
    )
    state.apply(4, deal_string(cards, 1))
    assert (state.phase, state.to_move) == (1, 1)
    state.apply(1, "18")
    assert (state.phase, state.to_move) == (2, 0)
    state.apply(0, "y")
    state.apply(1, "p")
    assert (state.phase, state.to_move) == (1, 2)
    state.apply(2, "p")
    assert (state.phase, state.to_move, state.solo) == (3, 0, 0)

    incoming_read, incoming_write = os.pipe()
    outgoing_read, outgoing_write = os.pipe()

    class PipeSocket:
        def fileno(self):
            return incoming_read

        def recv(self, size):
            return os.read(incoming_read, size)

        def sendall(self, data):
            os.write(outgoing_write, data)

    try:
        wire = ISSWire(PipeSocket())
        os.write(incoming_write, b"password:\nWelcome bot version 1\n")
        assert wire.readline(timeout=1) == "password:"
        assert wire.readline(timeout=1) == "Welcome bot version 1"
        wire.send("time")
        assert os.read(outgoing_read, 10) == b"time\n"
    finally:
        for fd in (incoming_read, incoming_write, outgoing_read, outgoing_write):
            os.close(fd)
