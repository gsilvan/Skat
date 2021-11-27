SUITS = ['♦', '♥', '♠', '♣']  # sorted by suit ascending
RANKS = ['7', '8', '9', 'J', 'Q', 'K', 'X', 'A']  # sorted by rank ascending


class Card:
    """
    Definition of the 32-card pack for Skat:
    Suits = {♦, ♥, ♠, ♣}
    Ranks = {7, 8, 9, J, Q, K, X, A}
    """

    def __init__(self, suit, rank) -> None:
        self.suit = SUITS[suit]
        self.rank = RANKS[rank]
        self.value = [0, 0, 0, 2, 3, 4, 10, 11][rank]

    def __lt__(self, other) -> bool:
        """
        As this function is used by built-in sort() function, this function
        compares two cards against each other with respect of the suit order
        in first place.

        For example: ♦Q is lower than ♦A, but ♦A is lower than ♥9.
        """
        rank_comparison = RANKS.index(self.rank) < RANKS.index(other.rank)
        suit_comparison = SUITS.index(self.suit) < SUITS.index(other.suit)
        if self.suit == other.suit:
            return rank_comparison
        else:
            return suit_comparison

    def __gt__(self, other) -> bool:
        """
        The greater than function is just comparing ranks against each other,
        without respecting suit order.
        """
        return RANKS.index(self.rank) > RANKS.index(other.rank)

    def __eq__(self, other) -> bool:
        """
        Two cards are equal if they have same suit and rank.
        """
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

    def __str__(self) -> str:
        return "┌───────┐\n" \
               f"| {self.rank}     |\n" \
               "|       |\n" \
               f"|   {self.suit}   |\n" \
               "|       |\n" \
               f"|     {self.rank} |\n" \
               "└───────┘"

    def __repr__(self) -> str:
        return f"{self.suit}{self.rank}"
