"""Short codec and original-deal lifecycle checks."""

from random import Random

import pytest

from skat.deal_code import CARDS, ID_COUNT, Deal, decode_deal, encode_deal
from skat.game import Round


def test_roundtrip_and_canonical_order():
    rng = Random(42)
    for i in range(300):
        cards = list(CARDS)
        rng.shuffle(cards)
        deal = Deal(
            tuple(tuple(cards[n : n + 10]) for n in (0, 10, 20)),
            tuple(cards[30:]),
            i % 3,
        )
        code = encode_deal(deal)
        assert decode_deal(code) == deal
        assert Deal.from_id(deal.id) == deal
        assert 0 <= deal.id < ID_COUNT < 2**53
        assert (
            encode_deal(
                Deal(
                    tuple(tuple(reversed(h)) for h in deal.hands),
                    deal.skat[::-1],
                    deal.dealer,
                )
            )
            == code
        )
        assert decode_deal(code.lower().replace("-", "")) == deal


def test_boundaries_and_invalid_ids():
    for value in (0, 1, 2, ID_COUNT - 1):
        assert Deal.from_id(value).id == value
    for value in (-1, ID_COUNT, True, 1.5):
        with pytest.raises(ValueError):
            Deal.from_id(value)
    base = Deal.from_id(0)
    with pytest.raises(ValueError):
        Deal((base.hands[0], base.hands[0], base.hands[2]), base.skat, 0)
    assert Deal(base.hands[::-1], base.skat, 0).id != base.id


def test_bad_codes_and_aliases():
    code = encode_deal(Deal.from_id(0))
    assert decode_deal(code.replace("0", "O")) == Deal.from_id(0)
    for bad in (
        None,
        "",
        code.replace("SK1", "SK2"),
        code[:-1],
        code[:-1] + ("0" if code[-1] != "0" else "1"),
        code.replace("0", "!"),
    ):
        with pytest.raises(ValueError):
            decode_deal(bad)


def test_original_survives_play_and_replay_is_unknown():
    deal = Deal.from_id(ID_COUNT - 1)
    round_ = Round(
        initial_cards=deal.initial_cards, dealer=deal.dealer, seed=42, start=False
    )
    assert round_.deal_id is None and round_.deal_code is None
    round_.start()
    assert round_.original_deal == deal
    assert decode_deal(round_.deal_code) == deal
    assert round_.hand_game is False
    assert set(round_.skat) != set(deal.skat)
    replay = Round(initial_cards=deal.initial_cards, replay=True, start=False)
    replay.deal()
    assert replay.deal_code is None


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (0, "SK1-0000-0000-000-3Y"),
        (1, "SK1-0000-0000-001-78"),
        (2, "SK1-0000-0000-002-AJ"),
        (8259883225513919, "SK1-7ARA-7PRN-ZXZ-HY"),
        (123456789012345, "SK1-03G9-230V-QVS-66"),
    ],
)
def test_version_one_reference_codes(value, code):
    assert encode_deal(Deal.from_id(value)) == code
    assert decode_deal(code).id == value


def test_fixed_card_partition_reference():
    cards = [CARDS[(i * 13) % 32] for i in range(32)]
    deal = Deal(
        tuple(tuple(cards[i : i + 10]) for i in (0, 10, 20)), tuple(cards[30:]), 1
    )
    assert deal.id == 620600309781925
    assert encode_deal(deal) == "SK1-0HMD-V2G6-5D5-XV"
    assert decode_deal("SK1-0HMD-V2G6-5D5-XV") == deal


def test_valid_checksum_does_not_allow_out_of_range_id():
    from zlib import crc32

    from skat.deal_code import ALPHABET

    # Eleven base32 characters can contain 55 bits, but only ID_COUNT IDs exist.
    for value in (ID_COUNT, 2**55 - 1):
        payload = "".join(
            ALPHABET[(value >> (5 * i)) & 31] for i in reversed(range(11))
        )
        crc = crc32(("SK1" + payload).encode("ascii")) & 1023
        check = ALPHABET[crc >> 5] + ALPHABET[crc & 31]
        with pytest.raises(ValueError, match="Bereich"):
            decode_deal("SK1" + payload + check)


def test_wire_card_order_is_explicit():
    assert tuple((c.suit, c.rank) for c in CARDS) == tuple(
        (suit, rank)
        for suit in ("♦", "♥", "♠", "♣")
        for rank in ("7", "8", "9", "J", "Q", "K", "X", "A")
    )
