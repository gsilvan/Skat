from skat.hand import Hand


class Player:
    """Engine-owned seat. Agents receive detached PlayerView snapshots."""

    def __init__(self, agent, seat_id):
        self.strategy = agent
        self.seat_id = seat_id
        self.hand = Hand()
        self.trick_stack = Hand()

    @property
    def trick_stack_value(self):
        return self.trick_stack.value

    def valid_moves(self, trick):
        cards = set(self.hand)
        return cards & trick.forced_cards or cards

    def receive_cards(self, cards):
        self.hand.extend(cards)

    def take_trick(self, trick):
        self.trick_stack.extend(card for _, card in trick.buffer)

    def bid(self, current_bid, offer=None):
        return self.strategy.bid(current_bid, offer)

    def pickup_skat(self):
        return self.strategy.pickup_skat()

    def press_skat(self):
        return self.strategy.press_skat()

    def declare_game(self):
        return self.strategy.declare_game()

    def play_card(self, trick):
        return self.strategy.choose_card(self.valid_moves(trick))
