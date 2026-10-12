"""Choose bidding risk on validation deals, then freeze it for new test deals."""

from pathlib import Path

from skat.agents.distill.train import save_training_report
from skat.agents.search.parallel import SearchPool
from skat.config import MCTSConfig
from skat.evaluation import evaluate, paired_points_comparison

THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def benchmark_bidding(
    output,
    *,
    agent,
    policy,
    reference_policy,
    bidding_model,
    mcts=None,
    workers=None,
    device="cpu",
    validation_deals=500,
    deals=2000,
):
    if validation_deals < 1 or deals < 1:
        raise ValueError("positive validation/test deal counts required")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    mcts = mcts or MCTSConfig()
    report = dict(
        agent=agent,
        validation_seed=42,
        test_seed=43,
        validation=[],
        state="running",
        objective="own official points per dealt round, including passed rounds",
    )
    report_path = directory / "report.json"
    save_training_report(report_path, report)
    with SearchPool(workers, capacity=mcts.determinizations) as pool:

        def run(name, count, seed, threshold=None):
            return evaluate(
                [agent],
                policy,
                directory / f"{name}.json",
                deals=count,
                seed=seed,
                device=device,
                mcts=mcts,
                opponent_policy=reference_policy,
                search_pool=pool,
                bidding="conservative" if threshold is None else "neural",
                bidding_model=bidding_model if threshold is not None else None,
                bidding_threshold=threshold if threshold is not None else 2 / 3,
            )

        try:
            baseline = run("validation-conservative", validation_deals, 42)
            report["validation_baseline"] = baseline["summary"][agent]
            for threshold in THRESHOLDS:
                result = run(
                    f"validation-{threshold:.2f}", validation_deals, 42, threshold
                )
                report["validation"].append(
                    dict(threshold=threshold, summary=result["summary"][agent])
                )
                save_training_report(report_path, report)
            selected = max(
                report["validation"],
                key=lambda x: (x["summary"]["points_per_deal"], -x["threshold"]),
            )["threshold"]
            report["selected_threshold"] = selected
            save_training_report(report_path, report)
            first = run("test-conservative", deals, 43)
            second = run("test-neural", deals, 43, selected)
            report["test_baseline"], report["test_neural"] = (
                first["summary"][agent],
                second["summary"][agent],
            )
            report["test_points_paired"] = paired_points_comparison(
                first["outcomes"], second["outcomes"], 43
            )
            report["panels"] = {
                panel: dict(
                    baseline=first["panels"][panel][agent],
                    neural=second["panels"][panel][agent],
                    points_paired=paired_points_comparison(
                        [r for r in first["outcomes"] if r["panel"] == panel],
                        [r for r in second["outcomes"] if r["panel"] == panel],
                        43,
                    ),
                )
                for panel in ("random", "policy")
            }
            report["state"] = "complete"
        except BaseException as error:
            report["state"], report["error"] = "failed", repr(error)
            raise
        finally:
            save_training_report(report_path, report)
    return report
