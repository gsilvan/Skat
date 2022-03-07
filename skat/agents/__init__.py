from abc import ABC, abstractmethod
from typing import Optional

from skat.card import Card
from skat.games import Game
from skat.player import Player


class Agent(ABC):
    state: Optional[Player]
    """Defines a minimal Agent Object"""

    VALID_BIDS = [
        0,
        18,
        20,
        22,
        24,
        27,
        30,
        33,
        36,
        40,
        44,
        45,
        48,
        50,
        54,
        55,
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
        200,
        204,
        207,
        209,
        210,
        216,
        220,
        225,
        228,
        230,
        231,
        234,
        240,
        242,
        243,
        250,
        252,
        253,
        260,
        261,
        264,
    ]

    @abstractmethod
    def choose_card(self, valid_moves: set[Card]) -> Card:
        raise NotImplementedError

    @abstractmethod
    def pickup_skat(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def bid(self, current_bid, offer=False) -> int:
        raise NotImplementedError

    @abstractmethod
    def declare_game(self, state) -> Game:
        raise NotImplementedError

    @abstractmethod
    def press_skat(self) -> list[Card]:
        raise NotImplementedError

    def set_state(self, state: Player) -> None:
        self.state = state
