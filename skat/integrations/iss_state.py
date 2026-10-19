"""Player-view ISS game replay for the live, line-oriented server protocol.

This tracks only cards revealed to the connected player. Server echoes are the
authoritative moves; the agent's proposed move does not change local state.
"""

from dataclasses import dataclass, field

from skat.integrations.iss import ISS_BIDS, format_card, parse_card
from skat.rules import GameDeclaration


def _declaration(token):
    if not token or token[0] not in "DHSCGN":
        raise ValueError(f"unsupported ISS declaration: {token}")
    code = token[0]
    flags = token[1:]
    if any(flag not in "OHSZ" for flag in flags):
        raise ValueError(f"unsupported ISS declaration: {token}")
    hand = "H" in flags or ("O" in flags and code != "N")
    return GameDeclaration(
        "suit" if code in "DHSC" else "grand" if code == "G" else "null",
        "DHSC".index(code) if code in "DHSC" else None,
        hand,
        "O" in flags,
        "S" in flags,
        "Z" in flags,
    )


@dataclass
class LiveGame:
    seat: int
    phase: int = 0
    to_move: int = 4
    bidder: int = 1
    asked: int = 0
    bid: int = 0
    solo: int = -1
    hand: set = field(default_factory=set)
    skat: set = field(default_factory=set)
    exposed: list = field(default_factory=lambda: [set(), set(), set()])
    played: list = field(default_factory=list)
    points: list = field(default_factory=lambda: [0, 0, 0])
    tricks: list = field(default_factory=lambda: [0, 0, 0])
    game: object = None
    own_initial: set = field(default_factory=set)

    def __post_init__(self):
        if self.seat not in range(3):
            raise ValueError("ISS seat must be fore, middle or rear hand")

    def next_bid(self):
        return next((value for value in ISS_BIDS if value > self.bid), -1)

    def legal_cards(self):
        if self.phase != 6 or self.to_move != self.seat:
            return ()
        forced = self.game.trick.forced_cards
        return tuple(
            sorted((self.hand & forced) or self.hand, key=lambda c: c.np_index)
        )

    def snapshot(self):
        """Encode the same player-view fields as the official SimpleState."""

        def mask(cards):
            return sum(
                1
                << (
                    8 * "DHSC".index(format_card(c)[0])
                    + "789TJQKA".index(format_card(c)[1])
                )
                for c in cards
            )

        sizes = [
            10 - sum(player == seat for player, _ in self.played) for seat in range(3)
        ]
        if self.phase == 5 and self.solo == self.seat:
            sizes[self.seat] = len(self.hand) + len(self.skat)
        else:
            sizes[self.seat] = len(self.hand)
        declaration = self.game.declaration if self.game is not None else None
        game_type = (
            declaration.suit
            if declaration and declaration.kind == "suit"
            else 4
            if declaration and declaration.kind == "grand"
            else 5
            if declaration
            else -1
        )
        flags = 0
        if declaration:
            flags = (
                int(declaration.hand)
                | int(declaration.ouvert) << 1
                | int(declaration.schneider_announced) << 2
                | int(declaration.schwarz_announced) << 3
            )

        def vector(values):
            return ",".join(map(str, values))

        return "\t".join(
            map(
                str,
                (
                    self.phase,
                    self.seat,
                    self.bid,
                    self.next_bid(),
                    mask(self.hand | self.skat) if self.phase == 5 else mask(self.hand),
                    mask(self.skat),
                    self.solo,
                    game_type,
                    flags,
                    vector(sizes),
                    vector(self.points),
                    vector(self.tricks),
                    vector(mask(cards) for cards in self.exposed),
                    ",".join(
                        f"{seat}:{format_card(card)}" for seat, card in self.played
                    ),
                    ",".join(format_card(card) for card in self.legal_cards()),
                ),
            )
        )

    def apply(self, player, action):
        """Apply one server-echoed action. Unsupported special moves stop play."""
        if self.phase == 6 and player == 4 and action.startswith(("TI.", "LE.")):
            self.phase, self.to_move = 7, -1
            return
        if self.phase == 6 and action.startswith("SC."):
            if player != self.solo:
                raise ValueError("only soloist may show cards")
            self.exposed[player] = {parse_card(card) for card in action.split(".")[1:]}
            if not self.exposed[player]:
                raise ValueError("ISS Show Cards had no revealed cards")
            return
        if self.phase == 6 and action == "RE":
            return  # Concession needs the server's subsequent end/play event.
        if player != self.to_move:
            raise ValueError(f"ISS out-of-turn move: {player}, expected {self.to_move}")
        if self.phase == 0:
            if player != 4:
                raise ValueError("ISS deal must come from world")
            blocks = action.split("|")
            if len(blocks) != 4:
                raise ValueError("invalid ISS deal")
            self.hand = {parse_card(card) for card in blocks[self.seat].split(".")}
            if len(self.hand) != 10:
                raise ValueError("ISS deal has no complete own hand")
            self.own_initial = self.hand.copy()
            self.phase, self.to_move = 1, 1
            return
        if self.phase == 1:
            if action == "p":
                if self.bidder == 0:
                    self.phase, self.to_move = 7, -1
                elif self.bidder == 1:
                    self.bidder, self.asked, self.to_move = 2, 0, 2
                elif self.bid < 18:
                    self.bidder, self.asked, self.to_move = 0, 4, 0
                else:
                    self.phase, self.solo, self.to_move = 3, self.asked, self.asked
            else:
                value = int(action)
                if value <= self.bid or value >= 500:
                    raise ValueError("invalid ISS bid")
                self.bid = value
                if player == 0:
                    self.phase, self.solo, self.to_move = 3, 0, 0
                else:
                    self.phase, self.to_move = 2, self.asked
            return
        if self.phase == 2:
            if action == "y":
                self.phase, self.to_move = 1, self.bidder
            elif action == "p":
                if self.bidder == 1:
                    self.phase, self.bidder, self.asked, self.to_move = 1, 2, 1, 2
                else:
                    self.phase, self.solo, self.to_move = 3, self.bidder, self.bidder
            else:
                raise ValueError("invalid ISS bid answer")
            return
        if self.phase == 3:
            if action == "s":
                self.phase, self.to_move = 4, 4
            else:
                parts = action.split(".")
                decl = _declaration(parts[0])
                if not decl.hand:
                    raise ValueError("ISS declaration after no pickup must be Hand")
                self.game = decl.make_game()
                if decl.ouvert:
                    self.exposed[self.solo] = (
                        self.hand.copy()
                        if self.solo == self.seat
                        else {parse_card(c) for c in parts[1:]}
                    )
                    if len(self.exposed[self.solo]) != 10:
                        raise ValueError("invalid ISS ouvert exposure")
                self.phase, self.to_move = 6, 0
            return
        if self.phase == 4:
            parts = action.split(".")
            if len(parts) != 2:
                raise ValueError("invalid ISS skat reveal")
            if self.solo == self.seat:
                self.skat = {parse_card(card) for card in parts}
                if len(self.skat) != 2:
                    raise ValueError("invalid ISS skat")
            self.phase, self.to_move = 5, self.solo
            return
        if self.phase == 5:
            parts = action.split(".")
            if len(parts) < 3:
                raise ValueError("ISS needs combined declaration and discards")
            decl = _declaration(parts[0])
            if decl.hand:
                raise ValueError("ISS pickup declaration cannot be Hand")
            self.game = decl.make_game()
            if self.solo == self.seat:
                pressed = {parse_card(card) for card in parts[1:3]}
                if len(pressed) != 2 or not pressed <= self.hand | self.skat:
                    raise ValueError("invalid own ISS discard")
                self.hand = (self.hand | self.skat) - pressed
                self.skat = pressed
            if decl.ouvert:
                self.exposed[self.solo] = (
                    self.hand.copy()
                    if self.solo == self.seat
                    else {parse_card(c) for c in parts[3:]}
                )
                if len(self.exposed[self.solo]) != 10:
                    raise ValueError("invalid ISS ouvert exposure")
            self.phase, self.to_move = 6, 0
            return
        if self.phase == 6:
            card = parse_card(action)  # Reject SC/RE/claims until mapped explicitly.
            if player == self.seat:
                if card not in self.legal_cards():
                    raise ValueError("illegal own ISS card")
                self.hand.remove(card)
            if card in self.exposed[player]:
                self.exposed[player].remove(card)
            self.played.append((player, card))
            self.game.trick.append(player, card)
            if self.game.trick.is_full:
                winner = self.game.trick.winner
                self.points[winner] += self.game.trick.value
                self.tricks[winner] += 1
                self.game.new_trick()
                self.to_move = winner
                if sum(self.tricks) == 10 or (
                    self.game.declaration.kind == "null" and self.tricks[self.solo]
                ):
                    self.phase, self.to_move = 7, -1
            else:
                self.to_move = (player + 1) % 3
            return
        raise ValueError("ISS game already finished")
