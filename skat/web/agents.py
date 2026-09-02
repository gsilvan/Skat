from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skat.agents import Agent
from skat.agents.bidding import VALID_BIDS, BiddingStyle
from skat.card import RANKS, SUITS, Card
from skat.games import Game
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame

if TYPE_CHECKING:
    from skat.player import Player
    from skat.web.session import GameSession


def card_id(card: Card) -> int:
    return card.np_index


def card_from_id(value: int) -> Card:
    if value not in range(32):
        raise ValueError(f"card id must be between 0 and 31, got {value}")
    return Card(value // len(RANKS), value % len(RANKS))


class WebAgent(Agent):
    """Human agent whose blocking decisions are supplied by the browser."""

    def __init__(self, session: GameSession, seat: int, seed: int | None = None):
        super().__init__(bidding=BiddingStyle.CONSERVATIVE, seed=seed)
        self.session = session
        self.seat = seat

    def bid(self, current_bid: int, offer: int | None = None) -> int:
        if offer is not None:
            return int(
                self.session.ask_human(
                    self.seat,
                    {
                        "kind": "bid",
                        "mode": "hear",
                        "offer": offer,
                        "options": [0, offer],
                    },
                )
            )
        next_bid = next((bid for bid in VALID_BIDS if bid > current_bid), None)
        options = [0] if next_bid is None else [0, next_bid]
        return int(
            self.session.ask_human(
                self.seat,
                {
                    "kind": "bid",
                    "mode": "say",
                    "current": current_bid,
                    "options": options,
                },
            )
        )

    def pickup_skat(self, state: Any) -> bool:
        return bool(
            self.session.ask_human(
                self.seat,
                {
                    "kind": "pickup",
                    "options": [True, False],
                },
            )
        )

    def press_skat(self) -> list[Card]:
        if self.state is None:
            raise RuntimeError("web agent has no player state")
        selected = self.session.ask_human(
            self.seat,
            {
                "kind": "press",
                "count": 2,
                "options": [card_id(card) for card in self.state.hand],
            },
        )
        cards = [card_from_id(int(value)) for value in selected]
        for card in cards:
            self.state.hand.remove(card)
        return cards

    def declare_game(self, state: Any) -> Game:
        choice = self.session.ask_human(
            self.seat,
            {
                "kind": "declare",
                "options": ["diamonds", "hearts", "spades", "clubs", "grand", "null"],
            },
        )
        games: dict[str, Game] = {
            "diamonds": SuitGame(SUITS.index("♦")),
            "hearts": SuitGame(SUITS.index("♥")),
            "spades": SuitGame(SUITS.index("♠")),
            "clubs": SuitGame(SUITS.index("♣")),
            "grand": Grand(),
            "null": Null(),
        }
        return games[str(choice)]

    def choose_card(self, valid_moves: set[Card]) -> Card:
        if self.state is None:
            raise RuntimeError("web agent has no player state")
        value = int(
            self.session.ask_human(
                self.seat,
                {
                    "kind": "play",
                    "options": [
                        card_id(card)
                        for card in sorted(valid_moves, key=lambda card: card.np_index)
                    ],
                },
            )
        )
        choice = card_from_id(value)
        self.state.hand.remove(choice)
        return choice


class ObservedAgent(Agent):
    """Delegate that exposes an automated agent's decisions to a session."""

    def __init__(self, delegate: Agent, session: GameSession, seat: int) -> None:
        super().__init__(bidding=delegate.bidding)
        self.delegate = delegate
        self.session = session
        self.seat = seat

    def set_state(self, state: Player) -> None:
        self.state = state
        self.delegate.set_state(state)

    def new_round_event(self) -> None:
        self.delegate.new_round_event()

    def trick_done_event(self, is_terminal: bool) -> None:
        self.delegate.trick_done_event(is_terminal)

    def _run(self, kind: str, call, **details):
        self.session.before_agent_action(self.seat, kind, details)
        result = call()
        shown = card_id(result) if isinstance(result, Card) else result
        if isinstance(result, list) and all(isinstance(item, Card) for item in result):
            shown = [card_id(item) for item in result]
        if isinstance(result, Game):
            shown = type(result).__name__
        self.session.after_agent_action(self.seat, kind, shown)
        return result

    def bid(self, current_bid: int, offer: int | None = None) -> int:
        return self._run(
            "bid",
            lambda: self.delegate.bid(current_bid, offer),
            current=current_bid,
            offer=offer,
        )

    def pickup_skat(self, state: Any) -> bool:
        return self._run("pickup", lambda: self.delegate.pickup_skat(state))

    def press_skat(self) -> list[Card]:
        return self._run("press", self.delegate.press_skat)

    def declare_game(self, state: Any) -> Game:
        return self._run("declare", lambda: self.delegate.declare_game(state))

    def choose_card(self, valid_moves: set[Card]) -> Card:
        return self._run(
            "play",
            lambda: self.delegate.choose_card(valid_moves),
            options=[card_id(card) for card in valid_moves],
        )

    def close(self) -> None:
        if hasattr(self.delegate, "close"):
            self.delegate.close()  # type: ignore[attr-defined]
