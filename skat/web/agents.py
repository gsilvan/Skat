"""Browser decisions and transparent instrumentation of existing agents."""

from copy import deepcopy
from dataclasses import asdict

from skat.agents import Agent
from skat.card import Card
from skat.hand import FULL_HAND
from skat.rules import Claim, Concede, GameDeclaration, OpenHand


def manual_seat(config):
    return config["kind"] == "human" or config.get("execution") == "manual"


def shown(value):
    if isinstance(value, Concede):
        return "concede"
    if isinstance(value, OpenHand):
        return "open"
    if isinstance(value, Claim):
        return f"claim{value.level}"
    if isinstance(value, Card):
        return value.np_index
    if isinstance(value, (list, tuple)):
        return [shown(item) for item in value]
    if isinstance(value, GameDeclaration):
        return asdict(value)
    if hasattr(value, "declaration"):
        return asdict(value.declaration)
    return value


class WebAgent(Agent):
    def __init__(self, session, seat, seed):
        super().__init__(seed=seed)
        self.session, self.seat = session, seat

    def ask(self, kind, options, **details):
        return self.session.ask(self.seat, kind, options, **details)

    def bid(self, current_bid, offer=None):
        options = (
            [0, offer]
            if offer is not None
            else [bid for bid in self.VALID_BIDS if bid == 0 or bid > current_bid]
        )
        return self.ask("bid", options, current=current_bid, offer=offer)

    def pickup_skat(self, state=None):
        return self.ask("pickup", [True, False])

    def press_skat(self):
        ids = self.ask("press", [card.np_index for card in self.state.hand])
        return [FULL_HAND[index] for index in ids]

    def declare_game(self, state=None):
        hand = bool(self.state.hand_game)
        declarations = []
        for kind, suit in [("suit", i) for i in range(4)] + [
            ("grand", None),
            ("null", None),
        ]:
            flags = [(False, False, False)]
            if kind == "null":
                flags.append((True, False, False))
            elif hand:
                flags.extend(
                    [(False, True, False), (False, True, True), (True, True, True)]
                )
            for ouvert, schneider, schwarz in flags:
                declarations.append(
                    asdict(
                        GameDeclaration(kind, suit, hand, ouvert, schneider, schwarz)
                    )
                )
        return GameDeclaration(**self.ask("declare", declarations))

    def choose_card(self, valid_moves):
        options = [
            card.np_index for card in sorted(valid_moves, key=lambda c: c.np_index)
        ]
        options.append("concede")
        if self.state.seat_id == self.state.solo_player_id:
            d = self.state.game.declaration
            levels = (
                [0]
                if d.kind == "null"
                else range(2 if d.schwarz_announced else int(d.schneider_announced), 3)
            )
            options.extend(f"claim{level}" for level in levels)
        else:
            options.append("open")
        choice = self.ask("play", options)
        if type(choice) is int:
            return FULL_HAND[choice]
        if choice == "concede":
            return Concede()
        if choice == "open":
            return OpenHand()
        return Claim(int(choice[-1]))

    def confirm_concession(self):
        return self.ask("confirm_concession", [True, False])

    def confirm_claim(self, level):
        return self.ask("confirm_claim", [True, False], level=level)


class AssistedWebAgent(WebAgent):
    """Suggest with an existing agent; execute only the browser's answer."""

    def __init__(self, delegate, session, seat, seed):
        super().__init__(session, seat, seed)
        self.delegate = delegate

    def set_state(self, state):
        super().set_state(state)
        self.delegate.set_state(state)

    def new_round_event(self):
        super().new_round_event()
        self.delegate.new_round_event()

    def trick_done_event(self, is_terminal):
        self.delegate.trick_done_event(is_terminal)

    def ask(self, kind, options, **details):
        self.session.before_decision(self.seat, kind, manual=True)
        press_view = deepcopy(self.state) if kind == "press" else None
        if kind == "bid":
            result = self.delegate.bid(details["current"], details["offer"])
        elif kind == "play":
            result = self.delegate.choose_card(
                {FULL_HAND[index] for index in options if type(index) is int}
            )
        elif kind == "confirm_claim":
            result = self.delegate.confirm_claim(details["level"])
        else:
            method = {
                "pickup": self.delegate.pickup_skat,
                "press": self.delegate.press_skat,
                "declare": self.delegate.declare_game,
                "confirm_concession": self.delegate.confirm_concession,
            }[kind]
            result = method()
        answer = super().ask(kind, options, suggestion=shown(result), **details)
        if press_view is not None:
            self.delegate.accept_pressed(
                press_view, [FULL_HAND[index] for index in answer]
            )
        return answer


class ObservedAgent(Agent):
    """Delegate every decision and lifecycle hook without changing strategies."""

    def __init__(self, delegate, session, seat):
        super().__init__()
        self.delegate, self.session, self.seat = delegate, session, seat

    def set_state(self, state):
        self.state = state
        self.delegate.set_state(state)

    def new_round_event(self):
        self.delegate.new_round_event()

    def trick_done_event(self, is_terminal):
        self.delegate.trick_done_event(is_terminal)

    def call(self, kind, function, *args):
        self.session.before_decision(self.seat, kind)
        result = function(*args)
        self.session.after_decision(self.seat, kind, shown(result))
        return result

    def bid(self, current_bid, offer=None):
        return self.call("bid", self.delegate.bid, current_bid, offer)

    def pickup_skat(self, state=None):
        return self.call("pickup", self.delegate.pickup_skat)

    def press_skat(self):
        return self.call("press", self.delegate.press_skat)

    def declare_game(self, state=None):
        return self.call("declare", self.delegate.declare_game)

    def choose_card(self, valid_moves):
        return self.call("play", self.delegate.choose_card, valid_moves)

    def confirm_concession(self):
        return self.call("confirm_concession", self.delegate.confirm_concession)

    def confirm_claim(self, level):
        return self.call("confirm_claim", self.delegate.confirm_claim, level)
