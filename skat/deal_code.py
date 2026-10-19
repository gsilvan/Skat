"""Version 1: reversible 53-bit IDs for unordered 10/10/10/2 deals.

The card order below is a wire-format constant, independent of model schemas.
Combination ranks follow lexicographic order (as in itertools.combinations).
"""

from dataclasses import dataclass
from math import comb
from zlib import crc32

from skat.card import RANKS, SUITS, Card

CARDS = tuple(
    Card(SUITS.index(suit), RANKS.index(rank))
    for suit in ("♦", "♥", "♠", "♣")
    for rank in ("7", "8", "9", "J", "Q", "K", "X", "A")
)
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
DEAL_COUNT = comb(32, 10) * comb(22, 10) * comb(12, 10)
ID_COUNT = 3 * DEAL_COUNT


def _rank(indices, n):
    rank, previous = 0, -1
    for i, chosen in enumerate(indices):
        for skipped in range(previous + 1, chosen):
            rank += comb(n - skipped - 1, len(indices) - i - 1)
        previous = chosen
    return rank


def _unrank(rank, n, k):
    indices, start = [], 0
    for i in range(k):
        for chosen in range(start, n):
            count = comb(n - chosen - 1, k - i - 1)
            if rank < count:
                indices.append(chosen)
                start = chosen + 1
                break
            rank -= count
    return indices


@dataclass(frozen=True)
class Deal:
    hands: tuple[tuple[Card, ...], ...]
    skat: tuple[Card, ...]
    dealer: int

    def __post_init__(self):
        if type(self.dealer) is not int or self.dealer not in range(3):
            raise ValueError("Geber muss 0, 1 oder 2 sein")
        hands = tuple(tuple(sorted(hand, key=CARDS.index)) for hand in self.hands)
        skat = tuple(sorted(self.skat, key=CARDS.index))
        flat = [card for hand in hands for card in hand] + list(skat)
        if (
            len(hands) != 3
            or any(len(hand) != 10 for hand in hands)
            or len(skat) != 2
            or len(flat) != 32
            or set(flat) != set(CARDS)
        ):
            raise ValueError(
                "Austeilung muss alle Karten genau einmal als 10/10/10/2 enthalten"
            )
        object.__setattr__(self, "hands", hands)
        object.__setattr__(self, "skat", skat)

    @property
    def initial_cards(self):
        return (*self.hands, self.skat)

    @property
    def id(self) -> int:
        remaining, rank = list(CARDS), 0
        for hand in self.hands:
            indices = sorted(remaining.index(card) for card in hand)
            rank = rank * comb(len(remaining), 10) + _rank(indices, len(remaining))
            remaining = [card for card in remaining if card not in hand]
        return rank * 3 + self.dealer

    @classmethod
    def from_id(cls, value: int) -> "Deal":
        if type(value) is not int or not 0 <= value < ID_COUNT:
            raise ValueError("Austeilungs-ID außerhalb des gültigen Bereichs")
        rank, dealer = divmod(value, 3)
        rank, r2 = divmod(rank, comb(12, 10))
        r0, r1 = divmod(rank, comb(22, 10))
        remaining, hands = list(CARDS), []
        for rank in (r0, r1, r2):
            indices = _unrank(rank, len(remaining), 10)
            hand = tuple(remaining[i] for i in indices)
            hands.append(hand)
            remaining = [card for card in remaining if card not in hand]
        return cls(tuple(hands), tuple(remaining), dealer)


def _base32(value, length):
    return "".join(ALPHABET[(value >> (5 * i)) & 31] for i in reversed(range(length)))


def encode_deal(deal: Deal) -> str:
    payload = _base32(deal.id, 11)
    check = _base32(crc32(("SK1" + payload).encode("ascii")) & 1023, 2)
    return f"SK1-{payload[:4]}-{payload[4:8]}-{payload[8:]}-{check}"


def decode_deal(code: str) -> Deal:
    if not isinstance(code, str):
        raise ValueError("Austeilungscode muss Text sein")
    compact = code.strip().upper().replace("-", "")
    if not compact.startswith("SK1"):
        raise ValueError("Unbekannte Austeilungscode-Version (erwartet SK1)")
    body = compact[3:].translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
    if len(body) != 13 or any(char not in ALPHABET for char in body):
        raise ValueError("Ungültige Länge oder Zeichen im Austeilungscode")
    payload, check = body[:11], body[11:]
    if check != _base32(crc32(("SK1" + payload).encode("ascii")) & 1023, 2):
        raise ValueError("Prüfsumme des Austeilungscodes stimmt nicht")
    value = 0
    for char in payload:
        value = value * 32 + ALPHABET.index(char)
    return Deal.from_id(value)
