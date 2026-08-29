"""Collect counterfactual win/loss labels from MCTS card play.

Each seeded deal is played three times, forcing every seat to be soloist once.
That supplies both winning and losing hands instead of observing only hands
that happened to win a normal auction.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import tqdm

from skat.agents.mcts import MCTSAgent
from skat.agents.neural_bidding.features import bidding_state
from skat.agents.neural_bidding.samples import BiddingShardWriter
from skat.deck import Deck
from skat.game import Round
from skat.utils.reproducibility import seed_everything


def derived_seed(root: int, deal: int, soloist: int, seat: int) -> int:
    digest = hashlib.sha256(
        f"skat-bidding-v1:{root}:{deal}:{soloist}:{seat}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def collect(
    out: str,
    deals: int,
    simulations: int,
    determinizations: int,
    seed: int,
    shard_size: int | None = None,
) -> dict[str, int]:
    if min(deals, simulations, determinizations) <= 0:
        raise ValueError("deals, simulations and determinizations must be positive")
    seed_everything(seed)
    options = {
        "directory": out,
        "tag": str(seed),
        "provenance": {
            "seed": seed,
            "simulations": simulations,
            "determinizations": determinizations,
        },
    }
    if shard_size is not None:
        options["shard_size"] = shard_size
    writer = BiddingShardWriter(**options)
    wins = 0
    try:
        for deal_index in tqdm.tqdm(range(deals), desc="bidding deals"):
            dealer = deal_index % 3
            cards = Deck.factory(seed=derived_seed(seed, deal_index, 3, 3)).to_list()
            for soloist in range(3):
                agents = [
                    MCTSAgent(
                        simulations=simulations,
                        determinizations=determinizations,
                        seed=derived_seed(seed, deal_index, soloist, seat),
                    )
                    for seat in range(3)
                ]
                round_ = Round(
                    dealer=dealer,
                    skip_bidding=True,
                    solo_player_id=soloist,
                    agents=agents,
                    seed=derived_seed(seed, deal_index, soloist, 3),
                    initial_cards=cards,
                    start=False,
                )
                won = bool(round_.start())
                wins += int(won)
                writer.add(
                    bidding_state(cards[soloist], soloist, dealer),
                    won=won,
                    group=deal_index,
                )
    finally:
        writer.close()
    return {
        "deals": deals,
        "samples": writer.written,
        "wins": wins,
        "losses": writer.written - wins,
        "shards": writer.shards,
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play every seat as soloist and record MCTS win/loss labels."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--deals", type=int, default=2_000)
    parser.add_argument("--simulations", type=int, default=600)
    parser.add_argument("--determinizations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    stats = collect(
        str(args.out),
        args.deals,
        args.simulations,
        args.determinizations,
        args.seed,
        args.shard_size,
    )
    print(
        f"recorded {stats['samples']} hands from {stats['deals']} deals: "
        f"{stats['wins']} wins / {stats['losses']} losses in "
        f"{stats['shards']} shard(s)"
    )


if __name__ == "__main__":
    main()
