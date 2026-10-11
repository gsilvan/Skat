"""Strict world-view ISS records and rule-checked card-play replay.

Protocol reference: https://skatgame.net/iss/clients.tgz,
skatgui/doc/skat-game-format and src/common/SimpleState.java (2019).
ISS fixes 0/1/2 to fore/middle/rear hand; it permits non-ladder bids < 500.
Special server events are deliberately unsupported, never skipped.
"""

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np

import iss
from skat.game import Round
from skat.rules import GameDeclaration


class RejectedRecord(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def reject(reason):
    raise RejectedRecord(reason)


def cards(token):
    try:
        result = tuple(iss.card[c] for c in token.replace("|", ".").split("."))
    except KeyError:
        reject("unknown_card")
    if len(set(result)) != len(result):
        reject("duplicate_card")
    return result


def declaration(token):
    match = re.fullmatch(r"([GCSHDN])(O?H?S?Z?)", token)
    if not match:
        reject("unsupported_declaration")
    kind, flags = match.groups()
    try:
        return GameDeclaration(
            "null" if kind == "N" else "grand" if kind == "G" else "suit",
            "DHSC".index(kind) if kind in "DHSC" else None,
            "H" in flags or ("O" in flags and kind != "N"),
            "O" in flags,
            "S" in flags,
            "Z" in flags,
        )
    except ValueError:
        reject("invalid_declaration_flags")


@dataclass(frozen=True)
class Record:
    identifier: str
    session: str
    date: str
    players: tuple[str, ...]
    original: tuple
    hands: tuple
    skat: tuple
    soloist: int
    bid: int
    contract: GameDeclaration
    actions: tuple
    won: bool
    points: int
    tricks: int
    score: int
    matadors: int
    overbid: bool

    @property
    def deal_hash(self):
        # Reordering cards within hands or rotating seats cannot escape grouping.
        blocks = [tuple(sorted(c.np_index for c in h)) for h in self.original[:3]]
        rotations = [blocks[i:] + blocks[:i] for i in range(3)]
        value = repr((min(rotations), sorted(c.np_index for c in self.original[3])))
        return hashlib.sha256(value.encode()).hexdigest()

    @property
    def metadata(self):
        return dict(
            id=self.identifier,
            session=self.session,
            date=self.date,
            players=self.players,
            actor_types=["unknown"] * 3,
            soloist=self.soloist,
            bid=self.bid,
            declaration=asdict(self.contract),
            won=self.won,
            points=self.points,
            score=self.score,
        )


def parse_record(line):
    iss.require_sgf_support()
    from pysgf import ParseError

    try:
        parsed = iss.SGF(line)
        if line[parsed.ix :].strip():
            reject("multiple_or_trailing_sgf")
        root = parsed.root

        def prop(name):
            values = root.get_list_property(name)
            if values is None or len(values) != 1:
                reject("missing_or_multiple_property")
            # pysgf preserves property-value escapes rather than decoding them.
            return re.sub(
                r"\\(\r\n|\r|\n|.)",
                lambda m: "" if m[1] in ("\n", "\r", "\r\n") else m[1],
                values[0],
            )

        if prop("GM") != "Skat" or "Skat Server" not in prop("PC"):
            reject("wrong_source")
        identifier, session = prop("ID"), prop("SE")
        if not identifier or not session:
            reject("missing_identifier")
        date = datetime.strptime(prop("DT"), "%Y-%m-%d/%H:%M:%S/UTC").isoformat()
        players = tuple(prop(f"P{i}") for i in range(3))
        result = prop("R").split()
        if len(result) < 3 or result[1] not in ("win", "loss"):
            reject("not_completed_game")
        fields = dict(t.split(":", 1) for t in result if ":" in t)
        soloist = int(fields["d"])
        if soloist not in range(3):
            reject("invalid_soloist")
        points, tricks, score, matadors = (int(fields[k]) for k in ("p", "t", "v", "m"))
        if any(fields.get(k, "-1") != "-1" for k in ("l", "to")) or any(
            fields.get(f"p{i}", "0") != "0" for i in range(3)
        ):
            reject("server_penalty")
        if fields.get("r", "0") != "0":
            reject("server_resignation")
        tokens = prop("MV").replace("|", ".").split()
    except RejectedRecord:
        raise
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, ParseError):
        reject("malformed_sgf_or_metadata")
    if len(tokens) < 4 or len(tokens) % 2 or tokens[0] != "w":
        reject("malformed_moves")
    deal = cards(tokens[1])
    if len(deal) != 32:
        reject("incomplete_deal")
    original = (deal[:10], deal[10:20], deal[20:30], deal[30:])
    moves = list(zip(tokens[2::2], tokens[3::2]))
    bidder, asked, bid, phase, declarer = 1, 0, 0, "bid", None
    index = 0
    # Follow the ISS BID/ANSWER transitions, including FH's final opening bid.
    while phase in ("bid", "answer"):
        if index == len(moves):
            reject("incomplete_bidding")
        seat, move = moves[index]
        index += 1
        if seat != str(bidder if phase == "bid" else asked):
            reject("bidding_turn")
        if phase == "bid":
            if move == "p":
                if bidder == 0:
                    reject("passed_game")
                if bidder == 1:
                    bidder, asked = 2, 0
                elif bid < 18:
                    bidder, asked = 0, None
                else:
                    declarer, phase = asked, "declare"
            elif move.isdecimal() and bid < int(move) < 500:
                bid = int(move)
                if bidder == 0:
                    declarer, phase = 0, "declare"
                else:
                    phase = "answer"
            else:
                reject("invalid_bid")
        elif move == "y":
            phase = "bid"
        elif move == "p":
            if bidder == 1:
                bidder, asked, phase = 2, 1, "bid"
            else:
                declarer, phase = bidder, "declare"
        else:
            reject("invalid_bid_response")
    if declarer != soloist:
        reject("soloist_mismatch")

    def take(expected):
        nonlocal index
        if index == len(moves) or moves[index][0] != expected:
            reject("declaration_turn")
        move = moves[index][1]
        index += 1
        return move

    token = take(str(soloist))
    pickup = token == "s"
    if pickup:
        if set(cards(take("w"))) != set(original[3]):
            reject("skat_mismatch")
        token = take(str(soloist))
    parts = token.split(".")
    contract = declaration(parts[0])
    if contract.hand == pickup:
        reject("pickup_hand_mismatch")
    hands = [list(h) for h in original[:3]]
    skat = original[3]
    if pickup:
        if len(parts) < 3:
            reject("missing_discards")
        skat = cards(".".join(parts[1:3]))
        hands[soloist].extend(original[3])
        for c in skat:
            if c not in hands[soloist]:
                reject("invalid_discard")
            hands[soloist].remove(c)
    revealed = parts[3:] if pickup else parts[1:]
    if revealed:
        if not contract.ouvert or set(cards(".".join(revealed))) != set(hands[soloist]):
            reject("ouvert_cards_mismatch")
    actions = []
    for seat, move in moves[index:]:
        if seat not in ("0", "1", "2") or move not in iss.card:
            reject("unsupported_event")
        actions.append((int(seat), iss.card[move]))
    return Record(
        f"iss:{identifier}",
        f"iss:{session}",
        date,
        players,
        original,
        tuple(tuple(h) for h in hands),
        tuple(skat),
        soloist,
        bid,
        contract,
        tuple(actions),
        result[1] == "win",
        points,
        tricks,
        score,
        matadors,
        "overbid" in result,
    )


class ReplayRound(Round):
    def _refresh(self):
        # No live agents decide anything. Avoid three detached/deepcopied views
        # on every action; observations are captured only for the acting seat.
        pass


def replay(record, capture=True):
    round_ = ReplayRound(
        dealer=2,
        start=False,
        skip_bidding=True,
        solo_player_id=record.soloist,
        declare_game=record.contract,
        hand_game=True,
        replay=True,
        initial_cards=[*record.hands, record.skat],
    )
    round_.highest_bid = record.bid
    round_.prepare()
    rows = []
    decisions = 0
    for seat, chosen in record.actions:
        if round_.is_finished or round_.next_player != seat:
            reject("card_turn_or_after_terminal")
        legal = round_.player[seat].valid_moves(round_.game.trick)
        if chosen not in legal:
            reject("illegal_card")
        if len(legal) > 1:
            decisions += 1
        if capture and len(legal) > 1:
            mask = np.zeros(32, dtype=bool)
            mask[[c.np_index for c in legal]] = True
            rows.append((round_.get_state(seat), mask, chosen.np_index, seat))
        round_.play_card(seat, chosen)
    if not round_.is_finished or round_.result is None:
        reject("incomplete_card_play")
    result = round_.result
    if (result.soloist_won, result.points[0], result.tricks[0], result.score) != (
        record.won,
        record.points,
        record.tricks,
        record.score,
    ) or result.matadors != abs(record.matadors):
        reject("result_or_scoring_mismatch")
    return rows if capture else decisions
