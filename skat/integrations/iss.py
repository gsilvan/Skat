"""Translate an ISS player-view snapshot into the engine's agent interface."""

from dataclasses import replace

import numpy as np

from skat.card import Card
from skat.game import PlayerView
from skat.hand import Hand
from skat.rules import GameDeclaration

ISS_RANKS = "789TJQKA"
ENGINE_RANKS = "789JQKXA"
ISS_SUITS = "DHSC"
GAME_CODES = "DHSCGN"


def cards_from_mask(mask):
    if not 0 <= mask < 1 << 32:
        raise ValueError("invalid ISS card mask")
    return tuple(
        Card(
            suit, ENGINE_RANKS.index("X" if ISS_RANKS[rank] == "T" else ISS_RANKS[rank])
        )
        for suit in range(4)
        for rank in range(8)
        if mask & (1 << (8 * suit + rank))
    )


def parse_card(token):
    if len(token) != 2 or token[0] not in ISS_SUITS or token[1] not in ISS_RANKS:
        raise ValueError(f"invalid ISS card: {token}")
    rank = "X" if token[1] == "T" else token[1]
    return Card(ISS_SUITS.index(token[0]), ENGINE_RANKS.index(rank))


def format_card(card):
    rank = "T" if card.rank == "X" else card.rank
    return ISS_SUITS[card.np_index // 8] + rank


def format_game(game):
    d = game.declaration
    code = GAME_CODES[d.suit if d.kind == "suit" else 4 if d.kind == "grand" else 5]
    if d.ouvert:
        code += "O"
    if d.hand and (not d.ouvert or d.kind == "null"):
        code += "H"
    if d.schwarz_announced and not d.ouvert:
        code += "Z"
    elif d.schneider_announced and not d.ouvert:
        code += "S"
    return code


def _ints(token, count):
    values = tuple(int(value) for value in token.split(","))
    if len(values) != count:
        raise ValueError("invalid ISS vector width")
    return values


def parse_snapshot(line):
    """The official Java player sends only fields visible in its SimpleState."""
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 15:
        raise ValueError("invalid ISS snapshot width")
    phase, seat, bid, next_bid, hand_mask, skat_mask, solo, game_type, flags = map(
        int, fields[:9]
    )
    sizes = _ints(fields[9], 3)
    points = _ints(fields[10], 3)
    tricks = _ints(fields[11], 3)
    exposed_masks = _ints(fields[12], 3)
    plays = []
    if fields[13]:
        for item in fields[13].split(","):
            player, card = item.split(":", 1)
            plays.append((int(player), parse_card(card)))
    legal = tuple(fields[14].split(",")) if fields[14] else ()
    if seat not in range(3) or phase not in (1, 2, 3, 5, 6):
        raise ValueError("invalid ISS phase or seat")
    if not 0 <= bid < 500 or any(not 0 <= n <= 12 for n in sizes):
        raise ValueError("invalid ISS bid or hand size")
    hand = cards_from_mask(hand_mask)
    skat = cards_from_mask(skat_mask) if solo == seat and phase >= 5 else ()
    exposed = tuple(cards_from_mask(mask) for mask in exposed_masks)
    if len(hand) != sizes[seat]:
        raise ValueError("ISS own hand size disagrees with mask")
    game = None
    history = []
    if phase == 6:
        if game_type not in range(6) or solo not in range(3):
            raise ValueError("invalid ISS contract")
        kind = "suit" if game_type < 4 else "grand" if game_type == 4 else "null"
        declaration = GameDeclaration(
            kind,
            game_type if game_type < 4 else None,
            bool(flags & 1),
            bool(flags & 2),
            bool(flags & 4),
            bool(flags & 8),
        )
        game = declaration.make_game()
        for player, card in plays:
            if player not in range(3):
                raise ValueError("invalid ISS card player")
            game.trick.append(player, card)
            if game.trick.is_full:
                history.append(game.trick)
                game.new_trick()
        if sum(tricks) != len(history):
            raise ValueError("ISS trick count disagrees with card history")
    view = PlayerView(
        seat_id=seat,
        dealer=2,  # ISS seats 0/1/2 are fore/middle/rear hand.
        hand=hand,
        hand_sizes=sizes,
        solo_player_id=solo,
        highest_bid=bid,
        hand_game=bool(flags & 1) if phase == 6 else None,
        skat=skat,
        game=game,
        trick_history=tuple(history),
        points=points,
        trick_counts=tricks,
        exposed_hands=exposed,
        observation=None,
    )
    if game is not None:
        view = replace(view, observation=observation(view))
    return phase, next_bid, view, legal


def observation(view):
    """Exactly the public observation layout used by Round.get_state."""
    seats = tuple((view.seat_id + i) % 3 for i in range(3))
    d = view.game.declaration
    solo = view.solo_player_id
    solo_points = view.points[solo] + (
        sum(c.value for c in view.skat) if solo == view.seat_id else 0
    )
    defence_points = sum(view.points) - view.points[solo]
    party_points = (
        (solo_points, defence_points)
        if solo == view.seat_id
        else (defence_points, solo_points)
    )
    leader = (
        view.game.trick.buffer[0].player_id
        if view.game.trick.buffer
        else view.trick_history[-1].winner
        if view.trick_history
        else 0
    )
    result = np.concatenate(
        (
            Hand(view.hand).as_vector,
            view.game.to_numpy(),
            [int(view.seat_id == solo)],
            [int(seat == solo) for seat in seats],
            np.asarray(party_points) / 120,
            *(
                sum(
                    (trick.to_numpy(seat) for trick in view.trick_history), np.zeros(32)
                )
                for seat in seats
            ),
            *(view.game.trick.to_numpy(seat) for seat in seats),
            [int(view.seat_id == (leader + offset) % 3) for offset in range(3)],
            [view.game.trick.value / 120],
            [
                d.hand,
                d.ouvert,
                d.schneider_announced,
                d.schwarz_announced,
                view.highest_bid / 264,
            ],
            *(Hand(view.exposed_hands[seat]).as_vector for seat in seats),
            [view.trick_counts[seat] / 10 for seat in seats],
            Hand(view.skat).as_vector,
        ),
        dtype=np.float32,
    )
    if result.shape != (376,):
        raise ValueError("ISS observation layout mismatch")
    return result


ISS_BIDS = (
    18,
    20,
    22,
    23,
    24,
    27,
    30,
    33,
    35,
    36,
    40,
    44,
    45,
    46,
    48,
    50,
    54,
    55,
    59,
    60,
    63,
    66,
    70,
    72,
    77,
    80,
    81,
    84,
    88,
    90,
    96,
    99,
    100,
    108,
    110,
    117,
    120,
    121,
    126,
    130,
    132,
    135,
    140,
    143,
    144,
    150,
    153,
    154,
    156,
    160,
    162,
    165,
    168,
    170,
    171,
    176,
    180,
    187,
    189,
    190,
    192,
    198,
    204,
    216,
    240,
    264,
)


def choose(agent, line):
    phase, next_bid, view, legal = parse_snapshot(line)
    agent.set_state(view)
    if phase == 1:
        result = str(next_bid) if next_bid > 0 and next_bid <= agent.max_bid() else "p"
    elif phase == 2:
        result = "y" if agent.bid(view.highest_bid, view.highest_bid) else "p"
    elif phase == 3:
        if agent.pickup_skat():
            result = "s"
        else:
            agent.set_state(replace(view, hand_game=True))
            result = format_game(agent.declare_game())
    elif phase == 5:
        pressed = agent.press_skat()
        result = ".".join(
            (
                format_game(agent.declare_game()),
                *(format_card(card) for card in pressed),
            )
        )
    else:
        valid = tuple(parse_card(token) for token in legal)
        if not valid:
            raise ValueError("ISS did not provide legal card moves")
        result = format_card(agent.choose_card(valid))
    if legal and phase == 6 and result not in legal:
        raise ValueError(f"agent selected illegal ISS card: {result}")
    return result
