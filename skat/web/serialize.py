from __future__ import annotations

from typing import Any

from skat.card import Card
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame

from .agents import ObservedAgent, card_id


def serialize_card(card: Card) -> dict[str, Any]:
    return {
        "id": card_id(card),
        "suit": card.suit,
        "rank": "10" if card.rank == "X" else card.rank,
        "value": card.value,
    }


def game_name(game) -> str | None:
    if game is None:
        return None
    if isinstance(game, SuitGame):
        return ("Diamonds", "Hearts", "Spades", "Clubs")[game.suit]
    if isinstance(game, Grand):
        return "Grand"
    if isinstance(game, Null):
        return "Null"
    return type(game).__name__


def snapshot(session) -> dict[str, Any]:
    round_ = session.round
    base: dict[str, Any] = {
        "id": session.id,
        "revision": session.revision,
        "status": session.status,
        "error": session.error,
        "paused": session.paused,
        "speedMs": session.speed_ms,
        "reveal": session.reveal,
        "pending": session.pending,
        "activeAction": session.active_action,
        "lastAction": session.last_action,
        "result": session.result,
        "config": session.public_config,
    }
    if round_ is None:
        return base

    human_seats = {
        index
        for index, player in enumerate(round_.player)
        if session.seat_kinds[index] == "human"
    }
    players = []
    for index, player in enumerate(round_.player):
        visible = session.reveal or index in human_seats
        strategy = player.strategy
        delegate = (
            strategy.delegate if isinstance(strategy, ObservedAgent) else strategy
        )
        players.append(
            {
                "seat": index,
                "label": session.seat_labels[index],
                "kind": session.seat_kinds[index],
                "hand": [serialize_card(card) for card in player.hand]
                if visible
                else [],
                "handCount": len(player.hand),
                "points": player.trick_stack_value,
                "soloist": index == round_.solo_player_id,
                "biddingProbability": getattr(
                    delegate, "last_bidding_probability", None
                ),
                "biddingGame": getattr(delegate, "last_bidding_game", None),
                "biddingScores": getattr(delegate, "last_bidding_scores", None),
            }
        )

    current_trick = []
    if round_.game is not None:
        current_trick = [
            {"seat": turn.player_id, "card": serialize_card(turn.card)}
            for turn in round_.game.trick.buffer
        ]
    history = [
        {
            "winner": trick.winner,
            "value": trick.value,
            "cards": [
                {"seat": turn.player_id, "card": serialize_card(turn.card)}
                for turn in trick.buffer
            ],
        }
        for trick in round_.trick_history.buffer
    ]

    next_player = None
    if round_.phase.name == "PLAYING" and not round_.is_finished:
        next_player = round_.next_player
    base.update(
        {
            "phase": round_.phase.name.lower(),
            "dealer": round_.dealer,
            "frontHand": round_.front_hand,
            "nextPlayer": next_player,
            "soloist": None if round_.solo_player_id == -42 else round_.solo_player_id,
            "highestBid": round_.highest_bid,
            "handGame": round_.hand_game,
            "game": game_name(round_.game),
            "gameValue": round_.game_value,
            "overbid": round_.overbid,
            "pointsSoloist": round_.points_soloist,
            "pointsDefenders": round_.points_defenders,
            "players": players,
            "currentTrick": current_trick,
            "history": history,
            "skat": [serialize_card(card) for card in round_.skat]
            if session.reveal or session.status == "finished"
            else [],
        }
    )
    return base
