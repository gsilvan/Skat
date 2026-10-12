from skat.agents import Agent
from skat.rules import Claim, Concede, GameDeclaration, OpenHand


def read_choice(prompt, choices):
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer
        print("Choose one of: " + ", ".join(choices))


class CommandLineAgent(Agent):
    @property
    def bidding_label(self):
        return "human"

    def choose_card(self, valid_moves):
        print(f"Seat {self.state.seat_id}: {list(self.state.hand)}")
        for seat, cards in enumerate(self.state.exposed_hands):
            if cards:
                print(f"Open seat {seat}: {list(cards)}")
        moves = sorted(valid_moves, key=lambda c: c.np_index)
        choices = {str(i): c for i, c in enumerate(moves)}
        choices["concede"] = Concede()
        if self.state.seat_id == self.state.solo_player_id:
            d = self.state.game.declaration
            levels = (
                [0]
                if d.kind == "null"
                else range(2 if d.schwarz_announced else int(d.schneider_announced), 3)
            )
            choices.update({f"claim{level}": Claim(level) for level in levels})
        if self.state.seat_id != self.state.solo_player_id:
            choices["open"] = OpenHand()
        print("Cards: " + ", ".join(f"{i}={c!r}" for i, c in enumerate(moves)))
        return choices[read_choice("Card/action: ", choices)]

    def pickup_skat(self, state=None):
        return read_choice("Pick up skat? [y/n]: ", ("y", "n")) == "y"

    def bid(self, current_bid, offer=None):
        print(f"Your hand: {list(self.state.hand)}")
        if offer is not None:
            return (
                offer if read_choice(f"Hold {offer}? [y/n]: ", ("y", "n")) == "y" else 0
            )
        legal = [v for v in self.VALID_BIDS if v == 0 or v > current_bid]
        return int(
            read_choice(f"Bid above {current_bid} (0=pass): ", [str(v) for v in legal])
        )

    def press_skat(self):
        remaining = list(self.state.hand)
        pressed = []
        for _ in range(2):
            print(", ".join(f"{i}={c!r}" for i, c in enumerate(remaining)))
            index = int(
                read_choice(
                    "Press card index: ", [str(i) for i in range(len(remaining))]
                )
            )
            pressed.append(remaining.pop(index))
        return pressed

    def declare_game(self, state=None):
        kind = read_choice(
            "Game [diamonds/hearts/spades/clubs/grand/null]: ",
            ("diamonds", "hearts", "spades", "clubs", "grand", "null"),
        )
        suit = (
            ("diamonds", "hearts", "spades", "clubs").index(kind)
            if kind not in ("grand", "null")
            else None
        )
        ouvert = False
        schneider = schwarz = False
        if kind == "null" or self.state.hand_game:
            ouvert = read_choice("Ouvert? [y/n]: ", ("y", "n")) == "y"
        if kind != "null" and self.state.hand_game and not ouvert:
            level = read_choice(
                "Announcement [simple/schneider/schwarz]: ",
                ("simple", "schneider", "schwarz"),
            )
            schneider, schwarz = level != "simple", level == "schwarz"
        return GameDeclaration(
            "suit" if suit is not None else kind,
            suit,
            bool(self.state.hand_game),
            ouvert,
            schneider,
            schwarz,
        )

    def confirm_concession(self):
        return read_choice("Accept soloist concession? [y/n]: ", ("y", "n")) == "y"

    def confirm_claim(self, level):
        print(f"Soloist cards: {self.state.exposed_hands[self.state.solo_player_id]}")
        return (
            read_choice(f"Accept soloist claim (level {level})? [y/n]: ", ("y", "n"))
            == "y"
        )
