"""
A shard is a contract between the teacher and the trainer.

A collection run is an hour of MCTS and is the expensive half of the whole
exercise, so what it writes has to survive a Ctrl-C, has to be readable by
several collectors at once, and has to refuse to be read by a trainer that
expects a different observation. The last one matters most: `Round.get_state`
and `QNet.INPUT_SIZE` are coupled by hand, and a corpus collected before
a layout change is silently wrong rather than obviously broken.
"""

import os
import tempfile
import unittest

import numpy as np

from skat.agents.distill.samples import ShardWriter, load_shards
from skat.hand import FULL_HAND

WIDTH = 8  # narrow observations keep the fixtures readable


def writer(directory: str, **kwargs) -> ShardWriter:
    kwargs.setdefault("input_size", WIDTH)
    kwargs.setdefault("tag", "test")
    return ShardWriter(directory, **kwargs)


def add_one(w: ShardWriter, *, seed: float = 1.0, slots=(0, 1, 2)) -> None:
    """One decision over `slots`, with visits descending and values matching."""
    moves = [FULL_HAND[slot] for slot in slots]
    totals = {move: [10 - i, seed + i] for i, move in enumerate(moves)}
    w.add(
        state=np.full(WIDTH, seed, dtype=np.float32),
        totals=totals,
        moves=moves,
        choice=moves[0],
    )


class RoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_a_shard_round_trips(self) -> None:
        w = writer(self.directory.name)
        w.new_round()
        add_one(w, seed=3.0, slots=(4, 5))
        w.close()

        samples = load_shards(self.directory.name, input_size=WIDTH)
        self.assertEqual(1, len(samples))
        np.testing.assert_allclose(np.full(WIDTH, 3.0), samples.states[0].numpy())
        self.assertEqual([4, 5], samples.masks[0].nonzero().flatten().tolist())
        self.assertEqual(10.0, samples.visits[0][4].item())
        self.assertEqual(9.0, samples.visits[0][5].item())
        self.assertEqual(3.0, samples.values[0][4].item())
        self.assertEqual(4, samples.choices[0].item())
        self.assertEqual(0, samples.groups[0].item())

    def test_different_reward_objectives_cannot_be_mixed(self):
        from skat.rewards import REWARD_OBJECTIVE

        for tag, provenance in (
            ("old", {}),
            ("new", {"reward_objective": REWARD_OBJECTIVE}),
        ):
            w = writer(self.directory.name, tag=tag, provenance=provenance)
            w.new_round()
            add_one(w)
            w.close()
        with self.assertRaisesRegex(RuntimeError, "Mixed reward objectives"):
            load_shards(self.directory.name, input_size=WIDTH)

    def test_reward_objective_round_trips(self):
        from skat.rewards import REWARD_OBJECTIVE

        w = writer(
            self.directory.name, provenance={"reward_objective": REWARD_OBJECTIVE}
        )
        w.new_round()
        add_one(w)
        w.close()
        self.assertEqual(
            REWARD_OBJECTIVE,
            load_shards(self.directory.name, input_size=WIDTH).reward_objective,
        )

    def test_independent_bonus_sums_round_trip(self):
        from skat.agents.mcts import MoveTotals
        from skat.rewards import MCTS_TIEBREAK_OBJECTIVE

        w = writer(
            self.directory.name,
            provenance={"reward_objective": MCTS_TIEBREAK_OBJECTIVE},
        )
        w.new_round()
        moves = list(FULL_HAND[:2])
        totals = MoveTotals(moves)
        totals[moves[0]], totals[moves[1]] = [10, 5.0], [10, 5.0]
        totals.bonus_sums[moves[0]], totals.bonus_sums[moves[1]] = 1.0, 8.0
        w.add(np.zeros(WIDTH), totals, moves, moves[1])
        w.close()
        samples = load_shards(self.directory.name, input_size=WIDTH)
        self.assertEqual([1.0, 8.0], samples.bonuses[0, :2].tolist())
        from skat.agents.distill.train import build_targets

        self.assertEqual(1, int(build_targets(samples, "value", 0.05).argmax()))

    def test_nothing_lands_on_a_card_that_was_not_a_move(self) -> None:
        w = writer(self.directory.name)
        w.new_round()
        add_one(w, slots=(4, 5))
        w.close()
        samples = load_shards(self.directory.name, input_size=WIDTH)
        illegal = ~samples.masks[0]
        self.assertEqual(0.0, samples.visits[0][illegal].sum().item())
        self.assertEqual(0.0, samples.values[0][illegal].sum().item())

    def test_every_sample_of_one_round_shares_a_group(self) -> None:
        w = writer(self.directory.name)
        for _ in range(2):
            w.new_round()
            add_one(w)
            add_one(w)
        w.close()
        samples = load_shards(self.directory.name, input_size=WIDTH)
        self.assertEqual([0, 0, 1, 1], samples.groups.tolist())

    def test_a_position_with_no_visits_is_dropped(self) -> None:
        # It would divide to nan in `visit_targets`; refuse it at the source.
        w = writer(self.directory.name)
        w.new_round()
        move = FULL_HAND[0]
        w.add(
            state=np.zeros(WIDTH, dtype=np.float32),
            totals={move: [0, 0.0]},
            moves=[move],
            choice=move,
        )
        self.assertEqual(0, len(w))
        w.close()
        self.assertEqual([], os.listdir(self.directory.name))

    def test_an_observation_of_the_wrong_width_is_refused(self) -> None:
        w = writer(self.directory.name)
        w.new_round()
        move = FULL_HAND[0]
        with self.assertRaises(ValueError) as caught:
            w.add(
                state=np.zeros(WIDTH + 1, dtype=np.float32),
                totals={move: [1, 1.0]},
                moves=[move],
                choice=move,
            )
        self.assertIn(str(WIDTH), str(caught.exception))


class ShardingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_it_rolls_over_at_the_shard_size(self) -> None:
        w = writer(self.directory.name, shard_size=2)
        w.new_round()
        for _ in range(4):
            add_one(w)
        self.assertEqual(2, len(os.listdir(self.directory.name)))
        self.assertEqual(4, len(w))

    def test_close_flushes_the_tail(self) -> None:
        w = writer(self.directory.name, shard_size=2)
        w.new_round()
        for _ in range(3):
            add_one(w)
        self.assertEqual(1, len(os.listdir(self.directory.name)))
        w.close()
        self.assertEqual(2, len(os.listdir(self.directory.name)))
        self.assertEqual(3, len(load_shards(self.directory.name, input_size=WIDTH)))

    def test_closing_an_empty_writer_writes_nothing(self) -> None:
        writer(self.directory.name).close()
        self.assertEqual([], os.listdir(self.directory.name))

    def test_two_writers_in_one_directory_do_not_collide(self) -> None:
        first = writer(self.directory.name, tag="a")
        second = writer(self.directory.name, tag="b")
        for w in (first, second):
            w.new_round()
            add_one(w)
            w.close()
        self.assertEqual(2, len(os.listdir(self.directory.name)))
        self.assertEqual(2, len(load_shards(self.directory.name, input_size=WIDTH)))

    def test_group_ids_from_separate_shards_stay_separate(self) -> None:
        # Both writers count their rounds from zero. Concatenated naively they
        # would merge, and a split on group boundaries would leak.
        for tag in ("a", "b"):
            w = writer(self.directory.name, tag=tag)
            w.new_round()
            add_one(w)
            w.close()
        samples = load_shards(self.directory.name, input_size=WIDTH)
        self.assertEqual(2, len(set(samples.groups.tolist())))

    def test_full_width_seed_in_provenance_is_lossless(self):
        # SHA-256 stage seeds are uint32; signed int32 used to raise here for
        # values above 2**31 (the standard seed=42 run hit exactly that path).
        w = writer(self.directory.name, provenance={"seed": 2_522_612_516})
        w.new_round()
        add_one(w)
        w.close()
        path = next(
            os.path.join(self.directory.name, name)
            for name in os.listdir(self.directory.name)
        )
        with np.load(path) as shard:
            self.assertEqual(2_522_612_516, int(shard["seed"]))


class RefusalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_a_shard_from_another_state_layout_is_refused(self) -> None:
        w = writer(self.directory.name)
        w.new_round()
        add_one(w)
        w.close()
        with self.assertRaises(RuntimeError) as caught:
            load_shards(self.directory.name, input_size=240)
        message = str(caught.exception)
        self.assertIn(str(WIDTH), message)
        self.assertIn("240", message)

    def test_an_empty_directory_says_so(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_shards(self.directory.name, input_size=WIDTH)

    def test_a_shard_from_a_future_version_is_refused(self) -> None:
        np.savez_compressed(
            os.path.join(self.directory.name, "mcts-x-0-0000.npz"),
            states=np.zeros((1, WIDTH), dtype=np.float32),
            visits=np.zeros((1, 32), dtype=np.int32),
            values=np.zeros((1, 32), dtype=np.float32),
            masks=np.zeros((1, 32), dtype=bool),
            groups=np.zeros(1, dtype=np.int32),
            choices=np.zeros(1, dtype=np.int8),
            input_size=np.int32(WIDTH),
            shard_version=np.int32(99),
        )
        with self.assertRaises(RuntimeError) as caught:
            load_shards(self.directory.name, input_size=WIDTH)
        self.assertIn("v99", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
