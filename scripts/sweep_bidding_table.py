"""Joint threshold sweep for three Random agents, with totals and 3D payoff plots."""

import argparse
import csv
import hashlib
import itertools
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.sweep_bidding_threshold import metrics, write_json
from skat.agents.neural_bidding.inference import NeuralBidder
from skat.agents.random import RandomAgent
from skat.game import Tournament
from skat.utils.processes import available_cpus

FINE_VALUES = tuple(i / 20 for i in range(21))
PREDICTOR = None


def grid_values(step):
    step = Decimal(str(step))
    ticks = step * 20
    if step <= 0 or step > 1 or ticks != int(ticks) or 20 % int(ticks):
        raise ValueError("step must divide 1 and be a multiple of 0.05")
    return tuple(i / 20 for i in range(0, 21, int(ticks)))


class PredictionCache:
    def __init__(self, path, device):
        self.predictor = NeuralBidder(path, device=device)
        self.values = {}

    def probability(self, cards, seat, dealer):
        key = (tuple(sorted(card.np_index for card in cards)), seat, dealer)
        if key not in self.values:
            self.values[key] = self.predictor.probability(cards, seat, dealer)
        return self.values[key]


class ThresholdBidder:
    def __init__(self, threshold):
        self.threshold = threshold

    def probability(self, cards, seat, dealer):
        return PREDICTOR.probability(cards, seat, dealer)


def initialize_worker(model, device):
    global PREDICTOR
    torch.set_num_threads(1)
    PREDICTOR = PredictionCache(model, device)


def case_name(thresholds):
    return (
        "baseline"
        if thresholds is None
        else "thresholds-" + "-".join(f"{t:.2f}" for t in thresholds)
    )


def run_case(thresholds, rounds, seed, directory):
    agents = [RandomAgent(seed=seed + seat) for seat in range(3)]
    if thresholds is not None:
        for agent, threshold in zip(agents, thresholds, strict=True):
            agent.configure_bidding("neural", ThresholdBidder(threshold))
    tournament = Tournament(rounds, agents, seed=seed, progress=False)
    started = time.perf_counter()
    try:
        tournament.start()
    finally:
        completed = len(tournament.round_records)
        result = {
            "thresholds": thresholds,
            "seed": seed,
            "completed_rounds": completed,
            "status": "complete" if completed == rounds else "interrupted",
            "bidders": [a.bidding_label for a in agents],
            "seats": [metrics(s, completed) for s in tournament.seat_scores]
            if completed
            else [],
            "seconds": time.perf_counter() - started,
            "rounds": tournament.round_records,
        }
        write_json(Path(directory) / (case_name(thresholds) + ".json"), result)
    return {k: v for k, v in result.items() if k != "rounds"}


def payoff_grid(cases, values, objective):
    lookup = {tuple(row["thresholds"]): row for row in cases}
    shape = (len(values),) * 3
    result = np.empty((*shape, 3), dtype=float)
    for indices in np.ndindex(shape):
        combo = tuple(values[i] for i in indices)
        result[indices] = [score[objective] for score in lookup[combo]["seats"]]
    return result


def regrets(payoffs):
    """Unilateral improvement at each profile; equal payoffs are irrelevant."""
    result = np.empty_like(payoffs)
    for seat in range(3):
        own = payoffs[..., seat]
        result[..., seat] = own.max(axis=seat, keepdims=True) - own
    return result


def choose_profile(payoffs, values):
    gains = regrets(payoffs)
    indices = min(
        np.ndindex(payoffs.shape[:3]),
        key=lambda i: (
            gains[i].max(),
            gains[i].sum(),
            -payoffs[i].min(),
            -payoffs[i].sum(),
            i,
        ),
    )
    equilibria = [
        {"thresholds": [values[j] for j in i], "utilities": payoffs[i].tolist()}
        for i in np.ndindex(payoffs.shape[:3])
        if np.all(gains[i] == 0)
    ]
    return {
        "thresholds": [values[i] for i in indices],
        "utilities": payoffs[indices].tolist(),
        "regrets": gains[indices].tolist(),
        "empirical_grid_equilibria": equilibria,
    }


def response_profiles(current, values=FINE_VALUES):
    return sorted(
        {
            tuple(current[:seat]) + (value,) + tuple(current[seat + 1 :])
            for seat in range(3)
            for value in values
        }
    )


def best_responses(current, lookup, objective, values=FINE_VALUES):
    own = [score[objective] for score in lookup[tuple(current)]["seats"]]
    answers = []
    for seat in range(3):
        options = [
            (
                value,
                lookup[tuple(current[:seat]) + (value,) + tuple(current[seat + 1 :])][
                    "seats"
                ][seat][objective],
            )
            for value in values
        ]
        maximum = max(score for _, score in options)
        # Stay on a tied best response rather than introducing artificial cycles.
        best = (
            current[seat]
            if own[seat] == maximum
            else min(value for value, score in options if score == maximum)
        )
        answers.append(
            {
                "seat": seat,
                "current_threshold": current[seat],
                "best_threshold": best,
                "current_utility": own[seat],
                "best_utility": maximum,
                "gain": maximum - own[seat],
            }
        )
    return answers


def plot_surfaces(cases, xvalues, yvalues, third, output, title, selected=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lookup = {tuple(row["thresholds"]): row for row in cases}
    x, y = np.meshgrid(xvalues, yvalues, indexing="ij")
    colors = ("#0891b2", "#ea580c", "#7c3aed")
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    for seat, color in enumerate(colors):
        z = np.asarray(
            [
                [lookup[(a, b, third)]["seats"][seat]["game_points"] for b in yvalues]
                for a in xvalues
            ]
        )
        ax.plot_surface(
            x, y, z, color=color, alpha=0.5, edgecolor=color, linewidth=0.4, shade=True
        )
        ax.scatter(x, y, z, color=color, s=14, depthshade=False)
        if selected is not None and selected[2] == third:
            score = lookup[tuple(selected)]["seats"][seat]["game_points"]
            ax.plot(
                [selected[0]],
                [selected[1]],
                [score],
                marker="*",
                markersize=14,
                color=color,
                markeredgecolor="black",
                linestyle="None",
                zorder=20,
            )
    ax.set(
        xlabel="Neural-Threshold Sitz 0",
        ylabel="Neural-Threshold Sitz 1",
        zlabel="Gesamtpunkte nach dem Lauf",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.view_init(elev=28, azim=-130)
    ax.set_box_aspect((1.2, 1.2, 0.9))
    ax.legend(
        handles=[
            Patch(facecolor=color, label=f"Sitz {seat}")
            for seat, color in enumerate(colors)
        ],
        loc="upper left",
    )
    fig.suptitle(
        title + f"\nSitz 2: Threshold {third:.2f} · Punkte sind gemessene Rasterwerte"
    )
    fig.savefig(Path(str(output) + ".png"), dpi=180)
    fig.savefig(Path(str(output) + ".pdf"))
    plt.close(fig)


def plot_outcomes(baseline, selected, output, rounds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    x = np.arange(3)
    for offset, row, label, color in (
        (-0.18, baseline, "Conservative", "#64748b"),
        (0.18, selected, "Neural", "#0891b2"),
    ):
        bars = axes[0].bar(
            x + offset,
            [s["game_points"] for s in row["seats"]],
            width=0.35,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, fontsize=9)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set(
        xticks=x,
        xticklabels=[f"Sitz {i}" for i in x],
        ylabel="Gesamtpunkte",
        title=f"Gesamtpunkte nach {rounds} Runden",
    )
    axes[0].legend()
    for index, row in enumerate((baseline, selected)):
        positions = x * 2 + index
        bottom = np.zeros(3)
        for key, label, color in (
            ("wins", "Siege", "#16a34a"),
            ("losses", "Niederlagen", "#dc2626"),
            ("passed", "Eingepasst", "#94a3b8"),
        ):
            counts = np.asarray([s[key] for s in row["seats"]])
            axes[1].bar(
                positions,
                counts,
                bottom=bottom,
                color=color,
                label=label if index == 0 else None,
            )
            for p, n, b in zip(positions, counts, bottom, strict=True):
                if n:
                    axes[1].text(
                        p, b + n / 2, str(n), ha="center", va="center", fontsize=8
                    )
            bottom += counts
    axes[1].set(
        xticks=np.arange(6),
        xticklabels=[
            f"Sitz {seat}\n{kind}"
            for seat in range(3)
            for kind in ("Conserv.", "Neural")
        ],
        ylabel="Anzahl Runden",
        ylim=(0, rounds),
        title="Siege + Niederlagen + eingepasst = alle Runden",
    )
    axes[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.suptitle(
        "Random / Random / Random · Neural-Schwellen "
        + " / ".join(f"{t:.2f}" for t in selected["thresholds"])
    )
    fig.savefig(Path(str(output) + ".png"), dpi=180)
    fig.savefig(Path(str(output) + ".pdf"))
    plt.close(fig)


def write_csv(cases, path):
    keys = (
        "game_points",
        "wins",
        "losses",
        "passed",
        "solo_wins",
        "solo_losses",
        "defence_wins",
        "defence_losses",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("threshold0", "threshold1", "threshold2", "seat", *keys)
        )
        writer.writeheader()
        for row in cases:
            for seat, score in enumerate(row["seats"]):
                writer.writerow(
                    dict(
                        zip(
                            ("threshold0", "threshold1", "threshold2"),
                            row["thresholds"],
                            strict=True,
                        ),
                        seat=seat,
                        **{key: score[key] for key in keys},
                    )
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bidding-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--step",
        type=float,
        default=0.25,
        help="Joint raster: default 0.25; 0.05 tests all 9,261 combinations",
    )
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--objective", choices=("points", "wins"), default="points")
    parser.add_argument(
        "--max-updates",
        type=int,
        default=8,
        help="Fine best-response updates; 0 selects from the joint raster only",
    )
    parser.add_argument("--validation-seed", type=int, default=43)
    args = parser.parse_args(argv)
    try:
        values = grid_values(args.step)
    except ValueError as error:
        parser.error(str(error))
    if args.rounds < 1 or args.workers < 1 or args.max_updates < 0:
        parser.error("positive rounds/workers and nonnegative max-updates required")
    if args.validation_seed == args.seed:
        parser.error("validation seed must differ from the search seed")
    torch.set_num_threads(1)
    NeuralBidder(args.bidding_model, device=args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    search = args.out / f"seed-{args.seed}"
    search.mkdir()
    validation = args.out / f"seed-{args.validation_seed}"
    validation.mkdir()
    field = "game_points" if args.objective == "points" else "wins"
    source = hashlib.sha256()
    for path in sorted(Path("skat").rglob("*.py")):
        source.update(path.as_posix().encode())
        source.update(path.read_bytes())
    report = {
        "status": "running",
        "stage": "joint raster",
        "manifest": {
            "measured_at": datetime.now(UTC).isoformat(),
            "agents": ["random"] * 3,
            "rounds": args.rounds,
            "seed": args.seed,
            "validation_seed": args.validation_seed,
            "grid_values": values,
            "fine_values": FINE_VALUES,
            "objective": field,
            "workers": args.workers,
            "device": args.device,
            "source_sha256": source.hexdigest(),
            "bidding_model": str(args.bidding_model),
            "bidding_model_sha256": hashlib.sha256(
                args.bidding_model.read_bytes()
            ).hexdigest(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "search_method": (
                "exhaustive joint 0.05 raster"
                if len(values) == 21
                else "joint coarse raster, then unilateral full 0.05 responses; not an exhaustive fine joint raster"
            ),
        },
        "cases": [],
        "trajectory": [],
    }
    lookup = {}
    validation_lookup = {}

    def save():
        report["evaluated_combinations"] = len(lookup)
        write_json(args.out / "summary.json", report)

    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(args.bidding_model, args.device),
    ) as pool:

        def evaluate(combos, seed=args.seed):
            cache = lookup if seed == args.seed else validation_lookup
            directory = search if seed == args.seed else validation
            todo = [tuple(t) for t in sorted(set(combos)) if tuple(t) not in cache]
            futures = {
                pool.submit(run_case, t, args.rounds, seed, str(directory)): t
                for t in todo
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=report["stage"],
                unit="Kombination",
            ):
                row = future.result()
                cache[tuple(row["thresholds"])] = row
                if seed == args.seed:
                    report["cases"].append(row)
                save()

        try:
            baseline = pool.submit(
                run_case, None, args.rounds, args.seed, str(search)
            ).result()
            report["baseline"] = baseline
            save()
            evaluate(itertools.product(values, repeat=3))
            joint = payoff_grid(report["cases"], values, field)
            coarse = choose_profile(joint, values)
            report["coarse_selection"] = coarse
            current = tuple(coarse["thresholds"])
            for third in values:
                plot_surfaces(
                    report["cases"],
                    values,
                    values,
                    third,
                    args.out / f"surface-seat2-{third:.2f}",
                    f"Random / Random / Random · {args.rounds} Runden · Seed {args.seed}",
                    current,
                )
            print("Static 3D plots ready:", args.out, flush=True)
            save()
            visited = set()
            if args.max_updates:
                for iteration in range(args.max_updates + 1):
                    report["stage"] = f"fine responses {iteration}"
                    evaluate(response_profiles(current))
                    answers = best_responses(current, lookup, field)
                    report["trajectory"].append(
                        {"thresholds": current, "responses": answers}
                    )
                    if max(a["gain"] for a in answers) == 0:
                        report["refinement_status"] = (
                            "empirically stable on fine response grid"
                        )
                        break
                    if current in visited:
                        report["refinement_status"] = "best-response cycle"
                        break
                    visited.add(current)
                    if iteration == args.max_updates:
                        report["refinement_status"] = "update limit reached"
                        break
                    seat = max(range(3), key=lambda s: answers[s]["gain"])
                    changed = list(current)
                    changed[seat] = answers[seat]["best_threshold"]
                    current = tuple(changed)
                best = min(
                    report["trajectory"],
                    key=lambda row: (
                        max(a["gain"] for a in row["responses"]),
                        sum(a["gain"] for a in row["responses"]),
                    ),
                )
                current = tuple(best["thresholds"])
                report["selected_responses"] = best["responses"]
            else:
                report["refinement_status"] = "coarse raster only"
            selected = lookup[current]
            report["selected"] = selected
            plot_outcomes(baseline, selected, args.out / "outcomes", args.rounds)
            # Render a measured 3D slice through the selected fine profile.
            xs = tuple(sorted(set(values) | {current[0]}))
            ys = tuple(sorted(set(values) | {current[1]}))
            report["stage"] = "selected 3D slice"
            evaluate((a, b, current[2]) for a in xs for b in ys)
            plot_surfaces(
                report["cases"],
                xs,
                ys,
                current[2],
                args.out / "selected-surface",
                f"Random / Random / Random · {args.rounds} Runden · Seed {args.seed}",
                current,
            )
            report["stage"] = "independent validation"
            evaluate(response_profiles(current), args.validation_seed)
            validation_baseline = pool.submit(
                run_case, None, args.rounds, args.validation_seed, str(validation)
            ).result()
            report["validation"] = {
                "selected": validation_lookup[current],
                "baseline": validation_baseline,
                "responses": best_responses(current, validation_lookup, field),
            }
            plot_outcomes(
                validation_baseline,
                validation_lookup[current],
                args.out / "validation-outcomes",
                args.rounds,
            )
            write_csv(report["cases"], args.out / "search.csv")
            write_csv(list(validation_lookup.values()), args.out / "validation.csv")
            report["status"] = "complete"
            report["stage"] = "complete"
            save()
            write_analysis(report, args.out)
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = repr(error)
            save()
            raise


def write_analysis(report, directory):
    selected = report["selected"]
    baseline = report["baseline"]
    text = [
        "# Gemeinsamer Threshold-Sweep: Random / Random / Random",
        "",
        (
            f"{report['manifest']['rounds']} ausgeteilte Runden je Kombination, Seed {report['manifest']['seed']}. "
            "Alle drei Sitze reizen neural mit ihrer eigenen Schwelle. Die Baseline reizt an allen Sitzen konservativ."
        ),
        "",
        "## Punkteflächen der ausgewählten Kombination",
        "",
        "![3D-Gesamtpunkte](selected-surface.png)",
        "",
        (
            "Die x- und y-Achsen sind die Schwellen von Sitz 0 und 1. Die Höhe zeigt Gesamtpunkte. "
            "Die Schwelle von Sitz 2 steht im Titel und bleibt in dieser Ansicht fest. Sterne markieren die ausgewählte Kombination."
        ),
        "",
        "## Siege, Niederlagen und Gesamtpunkte",
        "",
        "![Ergebnisse](outcomes.png)",
        "",
        "| Sitz | Bidder / Threshold | Gesamtpunkte | Siege | Niederlagen | Eingepasst | Solo gewonnen / gespielt |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seat in range(3):
        for run, label in (
            (baseline, "conservative"),
            (selected, f"neural / {selected['thresholds'][seat]:.2f}"),
        ):
            score = run["seats"][seat]
            text.append(
                f"| {seat} | {label} | {score['game_points']:+d} | {score['wins']} | {score['losses']} | {score['passed']} | {score['solo_wins']} / {score['solo_rounds']} |"
            )
    text += [
        "",
        (
            "Siege zählen Parteisiege als Alleinspieler oder Gegenspieler. Eingepasste Runden sind keine Niederlagen. "
            "Offizielle Spielpunkte erhält nur der Alleinspieler; im Gegenspiel gibt es in dieser Wertung null Punkte."
        ),
        "",
        "## Beste Antworten und Stabilität",
        "",
        (
            "Ein Schnittpunkt der Punkteflächen bedeutet gleiche Punktzahlen. Ein Gleichgewicht erfordert dagegen, "
            "dass kein Sitz durch Änderung nur seiner eigenen Schwelle seinen Wert erhöhen kann."
        ),
        "",
        "Suchstatus: **"
        + report["refinement_status"]
        + "**. Zielgröße: **"
        + report["manifest"]["objective"]
        + "**.",
        "",
        "## Prüfung auf neuen Deals",
        "",
        "![Neue Deals](validation-outcomes.png)",
        "",
        (
            f"Auf Seed {report['manifest']['validation_seed']} wird die ausgewählte Kombination unverändert getestet "
            "und für jeden Sitz werden erneut alle 0.05-Schwellen gegen die festgehaltenen Gegner geprüft."
        ),
        "",
        "| Sitz | Gewählter Threshold | Beste Antwort auf neuen Deals | Verbesserung durch Abweichen |",
        "| --- | ---: | ---: | ---: |",
    ]
    if report.get("selected_responses"):
        details = [
            "Die Auswahl minimiert den größten einseitigen Verbesserungsgewinn unter den vollständig geprüften Suchprofilen. Sie maximiert nicht die Summe der drei Punktzahlen.",
            "",
            "| Sitz | Gewählter Threshold | Beste Antwort auf Suchdeals | Verbesserung durch Abweichen |",
            "| --- | ---: | ---: | ---: |",
        ]
        for response in report["selected_responses"]:
            details.append(
                f"| {response['seat']} | {response['current_threshold']:.2f} | {response['best_threshold']:.2f} | {response['gain']:+g} |"
            )
        details.append("")
        position = text.index("## Prüfung auf neuen Deals")
        text[position:position] = details
    for response in report["validation"]["responses"]:
        text.append(
            f"| {response['seat']} | {response['current_threshold']:.2f} | {response['best_threshold']:.2f} | {response['gain']:+g} |"
        )
    largest_gain = max(r["gain"] for r in report["validation"]["responses"])
    text += [
        "",
        f"Auf neuen Deals beträgt der größte gemessene einseitige Verbesserungsgewinn **{largest_gain:g}** ({report['manifest']['objective']}). "
        + (
            "Die ausgewählte Kombination ist damit auf diesen Deals kein empirisches Gleichgewicht."
            if largest_gain > 0
            else "Auf diesen Deals gibt es keine bessere einseitige Antwort im geprüften Raster."
        ),
    ]
    if report["manifest"]["objective"] == "game_points":
        text += [
            "",
            "Diese Suche optimiert Gesamtpunkte, nicht die absolute Anzahl der Parteisiege. "
            "Beide Ziele können verschiedene Schwellen bevorzugen. Mehr eingepasste Runden reduzieren die Zahl ausgespielter Runden; "
            "die Siege/Niederlagen und Solo-Ergebnisse sind deshalb vollständig mit angegeben. "
            "Für eine Suche nach möglichst vielen Parteisiegen kann derselbe Versuch mit `--objective wins` ausgeführt werden.",
        ]
    text += [
        "",
        (
            "Das Ergebnis ist eine empirische Rasteruntersuchung mit endlicher Stichprobe. "
            + (
                "Alle 9.261 feinen Kombinationen wurden gemessen. "
                if len(report["manifest"]["grid_values"]) == 21
                else "Das grobe gemeinsame Raster und die feinen einseitigen Antworten decken nicht alle 9.261 feinen Kombinationen ab. "
            )
            + "Ein beständiges Optimum oder ein Nash-Gleichgewicht im vollständigen Skat ist damit nicht bewiesen."
        ),
        "",
        "## Gemeinsame Rasteransichten",
        "",
    ]
    for third in report["manifest"]["grid_values"]:
        text += [
            f"### Threshold Sitz 2 = {third:.2f}",
            "",
            f"![Punkteflächen](surface-seat2-{third:.2f}.png)",
            "",
        ]
    (directory / "analysis.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
