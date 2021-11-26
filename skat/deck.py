import random

from skat.card import Card, RANKS, SUITS


class Deck:
    """
    A Deck contains 32 Cards() and provides methods for card-shuffling and
    card-distribution (dealing).
    """

    def __init__(self):
        """Initialize an empty deck and seed the random function"""
        random.seed(0)  # use a fixed seed for now
        self.deck: list[Card] = list()

    def initialize_cards(self) -> None:
        """
        Initialize and fill the 32 different cards from Card() into the deck.
        The list is initialized in a sorted fashion.
        """
        self.deck = list()
        for i, _ in enumerate(SUITS):
            for j, _ in enumerate(RANKS):
                self.deck.append(Card(i, j))

    def shuffle(self) -> None:
        """
        Shuffle the deck randomly.
        """
        random.shuffle(self.deck)

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

    def __str__(self) -> str:
        return f'{self.deck}'
