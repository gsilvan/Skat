"""Official declarations and scoring, shared by the engine and search.

ISkO November 2022, sections 2.3–2.6 and 3.6. No agent or ML dependencies.
"""

from dataclasses import dataclass
from math import ceil
from typing import Literal

RULES_SCHEMA = "skat-isko-2022-v1"


@dataclass(frozen=True)
class GameDeclaration:
    kind: Literal["suit", "grand", "null"]
    suit: int | None = None
    hand: bool = False
    ouvert: bool = False
    schneider_announced: bool = False
    schwarz_announced: bool = False

    def __post_init__(self):
        if self.kind not in ("suit", "grand", "null"):
            raise ValueError("unknown game kind")
        for flag in (
            self.hand,
            self.ouvert,
            self.schneider_announced,
            self.schwarz_announced,
        ):
            if type(flag) is not bool:
                raise ValueError("declaration flags must be boolean")
        if (
            self.kind == "suit"
            and (type(self.suit) is not int or self.suit not in range(4))
        ) or (self.kind != "suit" and self.suit is not None):
            raise ValueError("only suit games require a suit from 0 to 3")
        if self.kind == "null":
            if self.schneider_announced or self.schwarz_announced:
                raise ValueError("Null has no Schneider/Schwarz announcements")
        else:
            if (
                self.ouvert or self.schneider_announced or self.schwarz_announced
            ) and not self.hand:
                raise ValueError("point-game announcements and ouvert require Hand")
            if self.ouvert:
                object.__setattr__(self, "schwarz_announced", True)
            if self.schwarz_announced:
                object.__setattr__(self, "schneider_announced", True)

    @property
    def base_value(self):
        if self.kind == "null":
            return {
                (False, False): 23,
                (True, False): 35,
                (False, True): 46,
                (True, True): 59,
            }[self.hand, self.ouvert]
        return (9, 10, 11, 12)[self.suit] if self.kind == "suit" else 24

    @property
    def minimum_stages(self):
        return (
            1
            + int(self.hand)
            + 2 * int(self.schneider_announced)
            + 2 * int(self.schwarz_announced)
            + int(self.ouvert)
        )

    def make_game(self):
        from skat.games.grand import Grand
        from skat.games.null import Null
        from skat.games.suit import SuitGame

        if self.kind == "suit":
            game = SuitGame(self.suit)
        elif self.kind == "grand":
            game = Grand()
        else:
            game = Null(self.hand, self.ouvert)
        game.declaration = self
        return game


@dataclass(frozen=True)
class RoundResult:
    soloist: int | None
    soloist_won: bool | None
    reason: str
    points: tuple[int, int] = (0, 0)
    tricks: tuple[int, int] = (0, 0)
    matadors: int = 0
    stages: int = 0
    game_value: int = 0
    score: int = 0  # signed official score for the soloist
    declaration: GameDeclaration | None = None
    scored_declaration: GameDeclaration | None = None

    @property
    def overbid(self):
        return self.reason == "overbid"

    @property
    def winner_party(self):
        if self.soloist_won is None:
            return None
        return "soloist" if self.soloist_won else "defenders"


def evaluate_game(
    declaration,
    matadors,
    points,
    tricks,
    highest_bid=0,
    *,
    soloist=0,
    forced_winner=None,
    awarded_level=None,
    reason=None,
    null_overbid_value=None,
    null_overbid_declaration=None,
):
    """Evaluate a finished game, or an explicitly agreed early result.

    Early results use only awarded/proven grades, never infer Schwarz from
    unplayed tricks. Null overbids must supply the cheapest legal point-game
    substitute (ISkO 3.6.2); the engine/search computes it from the original cards.
    """
    solo_points, defence_points = points
    solo_tricks, defence_tricks = tricks
    if declaration.kind == "null":
        won = solo_tricks == 0
        if forced_winner is not None:
            won = forced_winner and solo_tricks == 0
        stages = 0
        value = declaration.base_value
    else:
        if awarded_level is None:
            schneider = min(points) <= 30
            schwarz = min(tricks) == 0
        else:
            schneider = awarded_level >= 1 or max(points) >= 90
            schwarz = awarded_level >= 2 or (sum(tricks) == 10 and min(tricks) == 0)
        stages = (
            1
            + int(declaration.hand)
            + int(schneider or declaration.schneider_announced)
            + int(declaration.schneider_announced)
            + int(schwarz or declaration.schwarz_announced)
            + int(declaration.schwarz_announced)
            + int(declaration.ouvert)
        )
        won = (
            solo_points >= 61
            and (not declaration.schneider_announced or defence_points <= 30)
            and (not declaration.schwarz_announced or defence_tricks == 0)
        )
        value = (matadors + stages) * declaration.base_value
        if forced_winner is not None:
            # A concession cannot reverse an outcome already established by
            # the announced objective or publicly earned points (ISkO 4.1.3).
            failed = (
                defence_points >= 60
                or (declaration.schneider_announced and defence_points > 30)
                or (declaration.schwarz_announced and defence_tricks > 0)
            )
            secured = (
                solo_points >= (90 if declaration.schneider_announced else 61)
                and not declaration.schwarz_announced
                and value >= highest_bid
            )
            won = False if failed else (True if secured else forced_winner)
    overbid = value < highest_bid
    scored = declaration
    if overbid:
        won = False
        if declaration.kind == "null":
            if null_overbid_value is None or null_overbid_declaration is None:
                raise ValueError("Null overbid requires a point-game substitute")
            value = null_overbid_value
            scored = null_overbid_declaration
        else:
            value = max(
                value,
                ceil(highest_bid / declaration.base_value) * declaration.base_value,
            )
    loss_reason = (
        "announced-level"
        if (declaration.schneider_announced or declaration.schwarz_announced)
        and solo_points >= 61
        else "cards"
    )
    return RoundResult(
        soloist,
        won,
        "overbid" if overbid else (reason or ("won" if won else loss_reason)),
        tuple(points),
        tuple(tricks),
        matadors,
        stages,
        value,
        value if won else -2 * value,
        declaration,
        scored,
    )


def null_overbid_substitute(cards, highest_bid):
    from skat.games.grand import Grand
    from skat.games.suit import SuitGame

    games = [SuitGame(s) for s in range(4)] + [Grand()]

    def cost(game):
        return max(game.spielwert(cards), ceil(highest_bid / game.value) * game.value)

    game = min(games, key=cost)
    return cost(game), game.declaration


@dataclass(frozen=True)
class Concede:
    """Offer to concede; the engine obtains any required consent."""


@dataclass(frozen=True)
class Claim:
    """Reveal soloist cards and request an early result (0/simple, 1/2 grades)."""

    level: int = 0

    def __post_init__(self):
        if self.level not in (0, 1, 2):
            raise ValueError("claim level must be 0, 1 or 2")


@dataclass(frozen=True)
class OpenHand:
    """Defender opens their cards to shorten play (ISkO 4.4.5)."""


def can_soloist_take_trick(game, hands, to_move, soloist):
    """Exact existential check over *all* legal continuations, not agent play.

    A defender may shorten by opening cards only when no continuation can
    give the soloist another trick. Memoized, intended for endgame claims.
    """
    from copy import copy
    from functools import lru_cache

    from skat.hand import FULL_HAND

    initial_hands = tuple(tuple(sorted(c.np_index for c in hand)) for hand in hands)
    initial_trick = tuple((seat, c.np_index) for seat, c in game.trick.buffer)

    @lru_cache(maxsize=None)
    def possible(remaining, buffer, seat):
        if not any(remaining):
            return False
        trick = copy(game.trick)
        trick.buffer = []
        for player, index in buffer:
            trick.append(player, FULL_HAND[index])
        forced = trick.forced_cards
        legal = [i for i in remaining[seat] if FULL_HAND[i] in forced] or remaining[
            seat
        ]
        for index in legal:
            updated = list(remaining)
            updated[seat] = tuple(i for i in remaining[seat] if i != index)
            played = buffer + ((seat, index),)
            if len(played) == 3:
                completed = copy(trick)
                completed.buffer = list(trick.buffer)
                completed.append(seat, FULL_HAND[index])
                winner = completed.winner
                if winner == soloist or possible(tuple(updated), (), winner):
                    return True
            elif possible(tuple(updated), played, (seat + 1) % 3):
                return True
        return False

    return possible(initial_hands, initial_trick, to_move)
