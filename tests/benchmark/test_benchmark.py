import json
import tempfile
import unittest
from pathlib import Path

from skat.agents.random import RandomAgent
from skat.benchmark.compare import compare_runs, mcnemar_exact
from skat.benchmark.config import BenchmarkConfig, policy_variants
from skat.benchmark.evaluation import evaluate_candidate, rank_results
from skat.benchmark.runner import Journal, stage_seed


class ConfigTest(unittest.TestCase):
    def test_primary_and_control_variants_are_explicit(self):
        self.assertEqual(
            ["policy-mlp", "policy-transformer"],
            [variant.name for variant in policy_variants()],
        )
        controls = policy_variants(include_controls=True)
        self.assertEqual(6, len(controls))
        self.assertEqual(6, len({variant.name for variant in controls}))

    def test_stage_seeds_are_stable_and_namespaced(self):
        self.assertEqual(stage_seed(7, "training"), stage_seed(7, "training"))
        self.assertNotEqual(stage_seed(7, "training"), stage_seed(7, "evaluation"))


class EvaluationTest(unittest.TestCase):
    @staticmethod
    def factory(seed, _seat):
        return RandomAgent(seed=seed)

    def test_seeded_evaluation_repeats_and_rotates_all_seats(self):
        first = evaluate_candidate("random", self.factory, rounds=3, seed=1234)
        second = evaluate_candidate("random", self.factory, rounds=3, seed=1234)
        self.assertEqual(first, second)
        self.assertEqual([0, 1, 2], [row["seat"] for row in first["by_seat"]])
        self.assertEqual(9, first["scheduled_rounds"])
        self.assertEqual(first["wins"] / first["rounds"], first["win_rate"])

    def test_ranking_uses_win_rate_then_card_points(self):
        rows = [
            {"name": "a", "win_rate": 0.5, "card_points_per_round": 50},
            {"name": "b", "win_rate": 0.6, "card_points_per_round": 40},
        ]
        ranked = rank_results(rows)
        self.assertEqual(["b", "a"], [row["name"] for row in ranked])
        self.assertEqual([1, 2], [row["rank"] for row in ranked])


class JournalTest(unittest.TestCase):
    def test_completed_stage_is_skipped_on_resume(self):
        config = BenchmarkConfig.from_profile("smoke")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            journal = Journal(path, config, resume=False)
            self.assertEqual(
                {"value": 1}, journal.run_stage("one", lambda: {"value": 1})
            )

            resumed = Journal(path, config, resume=True)

            def should_not_run():
                self.fail("completed stage was run twice")

            self.assertEqual({"value": 1}, resumed.run_stage("one", should_not_run))


class ComparisonTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(1.0, mcnemar_exact(0, 0))
        self.assertAlmostEqual(0.0625, mcnemar_exact(5, 0))

    def test_runs_are_paired_by_panel_seat_and_deal_seed(self):
        def row(outcomes):
            return {
                "name": "agent",
                "by_seat": [
                    {
                        "opponent": "random",
                        "seat": 0,
                        "outcomes": outcomes,
                        "deals": [{"seed": seed} for seed in (10, 11, 12)],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            (old / "results.json").write_text(
                json.dumps({"ranking": [row([False, True, False])]}),
                encoding="utf-8",
            )
            (new / "results.json").write_text(
                json.dumps({"ranking": [row([True, False, True])]}),
                encoding="utf-8",
            )
            comparison = compare_runs(old, new)[0]

        self.assertEqual(3, comparison["paired_deals"])
        self.assertEqual(2, comparison["improved"])
        self.assertEqual(1, comparison["regressed"])
        self.assertAlmostEqual(1 / 3, comparison["delta"])


if __name__ == "__main__":
    unittest.main()
