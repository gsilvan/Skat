"""Publish only information visible at the local table."""

from dataclasses import asdict

from .agents import manual_seat


def card_data(card):
    return {"id": card.np_index, "suit": card.suit, "rank": card.rank}


def trick_data(trick):
    return {
        "cards": [
            {"seat": seat, "card": card_data(card)} for seat, card in trick.buffer
        ],
        "winner": trick.winner,
        "value": trick.value,
    }


def snapshot(session, *, reveal=False):
    round_ = session.round
    if round_ is None:
        return {}
    finished = round_.result is not None
    public_points = [
        sum(t.value for t in round_.trick_history.buffer if t.winner == seat)
        for seat in range(3)
    ]
    solo = round_.solo_player_id if round_.solo_player_id in range(3) else None
    humans = {i for i, seat in enumerate(session.config["seats"]) if manual_seat(seat)}
    # Cache both views in the worker. Request threads never read the live Round.
    visible = humans | round_.exposed
    if reveal or finished:
        visible = set(range(3))
    players = [
        {
            "seat": seat,
            "label": config["label"],
            "kind": config["kind"],
            "hand": [card_data(card) for card in player.hand]
            if seat in visible
            else [],
            "handCount": len(player.hand),
            "points": player.trick_stack_value if finished else public_points[seat],
            "tricks": round_.trick_counts[seat],
        }
        for seat, (player, config) in enumerate(
            zip(round_.player, session.config["seats"], strict=True)
        )
    ]
    skat_visible = reveal or finished or (solo in humans and round_.skat_known)
    return {
        "dealCode": round_.deal_code if reveal or finished else None,
        "players": players,
        "dealer": round_.dealer,
        "phase": round_.phase.name.lower(),
        "nextPlayer": round_.next_player
        if round_.phase.name == "PLAYING" and not finished
        else None,
        "soloist": solo,
        "highestBid": round_.highest_bid,
        "declaration": asdict(round_.game.declaration) if round_.game else None,
        "currentTrick": trick_data(round_.game.trick) if round_.game else None,
        "history": [trick_data(trick) for trick in round_.trick_history.buffer],
        "skat": [card_data(card) for card in round_.skat] if skat_visible else [],
        "result": asdict(round_.result) if finished else None,
    }
