import random

from skat.card import RANKS, SUITS, Card


class Deck:
    """
    A Deck contains 32 Cards() and provides methods for card-shuffling and
    card-distribution (dealing).
    """

    def __init__(self) -> None:
        """Initialize an empty deck and seed the random function"""
        self.deck: list[Card] = list()

    def __len__(self) -> int:
        """
        The length of Deck is the quantity of Cards currently held by
        self.deck. TODO: grammar-check
        """
        return len(self.deck)

    def __hash__(self) -> int:
        """
        The hash of a Deck is the hash of the string representation of the
        Deck. We can't use self.deck because Lists are not immutable and thus
        not hashable.
        """
        return hash(str(self.deck))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __str__(self) -> str:
        """String representation of cards in self.deck"""
        return f"{self.deck}"

    def initialize_cards(self) -> None:
        """
        Initialize and fill the 32 different cards from Card() into the deck.
        The list is initialized in a sorted fashion.
        """
        self.deck = list()
        for i, _ in enumerate(SUITS):
            for j, _ in enumerate(RANKS):
                self.deck.append(Card(i, j))

    def shuffle(self, seed=None) -> None:
        """
        Shuffle the deck randomly.
        """
        random.seed(a=seed)
        random.shuffle(self.deck)
        random.seed(None)

    def deal_cards(self, quantity=10) -> list[Card]:
        """
        Pop cards from deck and return it to the caller. This function is not
        respecting the well-known offline dealing pattern
        three–Skat–four–three. As this pattern is a countermeasure of human
        cheating we don't need to implement it for not-human dealers.
        If one sees the need of implementing this, call this function with
        the quantity argument.
        :param quantity: The quantity of cards to be dealt. default = 10.
        :return: A list of Cards
        """
        result = list()
        for _ in range(quantity):
            result.append(self.deck.pop(0))
        return result
