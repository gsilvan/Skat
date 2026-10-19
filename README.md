# Skat

Eine Python-Engine für offizielles digitales Skat mit Kommandozeilenspieler,
Random-Agent, MCTS und Transformer-Policies. Policies lassen sich aus MCTS-
Selbstspiel oder aufgezeichneten ISS-Kartenzügen trainieren. Das Reizen ist
separat konfigurierbar und kann einen trainierten ISS-Handklassifikator nutzen.

## Agentenstärke gegen Random

Jede Variante spielt **1.000 Runden mit Seed 42** an Sitz 0 gegen zwei
`random`-Agenten mit `conservative`-Reizern an Sitz 1 und 2. Der Geber rotiert.
Alle Varianten erhalten dieselben Deals; die 1.000 Runden enthalten auch
eingepasste Blätter. Kartenmodell und Reizer werden getrennt angegeben.

<!-- agent-ratings:start -->
| Agent / Kartenmodell | Reizer | Ansager | Siege / gespielt | Als Alleinspieler | Im Gegenspiel | Eingepasst | Punkte / Runde | Δ Punkte zur Baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer · ISS 100k / 20 Epochen | neural (0.55) | heuristic | 648/949 (68.3%) | 321/333 (96.4%) | 327/616 (53.1%) | 51 | +13.42 | +14.39 |
| Transformer · ISS 100k / 20 Epochen | conservative | neural (1M/20 Ep.) | 640/946 (67.7%) | 314/330 (95.2%) | 326/616 (52.9%) | 54 | +11.01 | +11.97 |
| Transformer · ISS 100k / 20 Epochen | conservative | neural (1M/20 Ep.) + Hand-NN | 655/946 (69.2%) | 311/330 (94.2%) | 344/616 (55.8%) | 54 | +10.65 | +11.61 |
| MCTS · ISS-Handmodell 100k / 600 Simulationen / 20 Welten | conservative | heuristic | 656/946 (69.3%) | 312/330 (94.5%) | 344/616 (55.8%) | 54 | +10.48 | +11.45 |
| MCTS · ISS-Handmodell 300k / 600 Simulationen / 20 Welten | conservative | heuristic | 657/946 (69.5%) | 312/330 (94.5%) | 345/616 (56.0%) | 54 | +10.47 | +11.43 |
| MCTS · 600 Simulationen / 20 Welten | conservative | heuristic | 651/946 (68.8%) | 313/330 (94.8%) | 338/616 (54.9%) | 54 | +10.38 | +11.35 |
| Transformer · MCTS 2.000 / 10k / 100 Epochen | conservative | heuristic | 651/946 (68.8%) | 311/330 (94.2%) | 340/616 (55.2%) | 54 | +10.33 | +11.29 |
| Transformer · ISS 100k / 20 Epochen | conservative | heuristic + Hand-NN | 655/946 (69.2%) | 306/330 (92.7%) | 349/616 (56.7%) | 54 | +10.25 | +11.21 |
| Transformer · ISS 100k / 20 Epochen | conservative | neural (1M/40 Ep.) | 660/946 (69.8%) | 310/330 (93.9%) | 350/616 (56.8%) | 54 | +10.24 | +11.21 |
| Transformer · ISS 100k / 20 Epochen | conservative | heuristic | 648/946 (68.5%) | 302/330 (91.5%) | 346/616 (56.2%) | 54 | +10.16 | +11.12 |
| Transformer · MCTS 5k / 12 Epochen | conservative | heuristic | 618/946 (65.3%) | 307/330 (93.0%) | 311/616 (50.5%) | 54 | +10.04 | +11.00 |
| MCTS · 2.000 Simulationen / 20 Welten | conservative | heuristic | 643/946 (68.0%) | 312/330 (94.5%) | 331/616 (53.7%) | 54 | +9.97 | +10.94 |
| Transformer · ISS 100k / 20 Epochen | conservative | neural (300k) | 649/946 (68.6%) | 310/330 (93.9%) | 339/616 (55.0%) | 54 | +9.86 | +10.82 |
| MCTS · 600 Simulationen / 20 Welten | conservative | neural (1M/40 Ep.) | 651/946 (68.8%) | 311/330 (94.2%) | 340/616 (55.2%) | 54 | +9.76 | +10.72 |
| MCTS · 600 Simulationen / 20 Welten | conservative | neural (300k) | 641/946 (67.8%) | 317/330 (96.1%) | 324/616 (52.6%) | 54 | +9.76 | +10.72 |
| MCTS · 600 Simulationen / 20 Welten | conservative | neural (1M/20 Ep.) | 669/946 (70.7%) | 315/330 (95.5%) | 354/616 (57.5%) | 54 | +9.70 | +10.67 |
| Transformer · ISS 100k / 20 Epochen | conservative | neural (100k) | 657/946 (69.5%) | 307/330 (93.0%) | 350/616 (56.8%) | 54 | +9.55 | +10.52 |
| Transformer · MCTS 5k / 100 Epochen | conservative | heuristic | 652/946 (68.9%) | 305/330 (92.4%) | 347/616 (56.3%) | 54 | +9.38 | +10.34 |
| MCTS · 600 Simulationen / 20 Welten | conservative | neural (100k) | 649/946 (68.6%) | 308/330 (93.3%) | 341/616 (55.4%) | 54 | +9.24 | +10.21 |
| Random | neural (0.55) | heuristic | 470/949 (49.5%) | 242/333 (72.7%) | 228/616 (37.0%) | 51 | +1.81 | +2.77 |
| Random | conservative | neural (300k) | 436/946 (46.1%) | 217/330 (65.8%) | 219/616 (35.6%) | 54 | +0.17 | +1.14 |
| Random | conservative | neural (1M/40 Ep.) | 434/946 (45.9%) | 208/330 (63.0%) | 226/616 (36.7%) | 54 | -0.45 | +0.51 |
| Random | conservative | neural (1M/20 Ep.) | 443/946 (46.8%) | 203/330 (61.5%) | 240/616 (39.0%) | 54 | -0.62 | +0.35 |
| Random | conservative | neural (100k) | 449/946 (47.5%) | 211/330 (63.9%) | 238/616 (38.6%) | 54 | -0.87 | +0.09 |
| Random (Baseline) | conservative | heuristic | 440/946 (46.5%) | 215/330 (65.2%) | 225/616 (36.5%) | 54 | -0.96 | +0.00 |
<!-- agent-ratings:end -->

`5k` und `10k` bezeichnen die Anzahl der MCTS-Korpusrunden; `100k`, `300k`
und `1M` bezeichnen die ISS-Spiele im jeweiligen Trainingskorpus. Die
Epochenzahl beschreibt den
Trainingslauf;
gespielt wird jeweils mit dessen bestem Validierungscheckpoint.

Die Tabelle ist nach offiziellen Punkten pro ausgeteilter Runde sortiert.
Gegenspiel und eingepasste Blätter tragen dabei null Punkte bei; verlorene
Alleinspiele zählen negativ. Die Gewinnquoten beziehen sich auf gespielte
Runden und geben jeweils den Parteisieg des Kandidaten an. Eine hohe
Alleinspielerquote kann auch durch selektiveres Reizen entstehen; deshalb
stehen die Anzahl der Spiele und die Punkte daneben. Kleine Unterschiede in
diesem einen Lauf belegen keine allgemeine Rangfolge gegen andere Gegner.
Bei den beiden `Hand-NN`-Profilen verschiebt ein geändertes Spiel auch die
Zufallszahlen der Random-Gegner in späteren Runden. Ein zusätzlicher
[gepaarter Vergleich](docs/pickup-training.md) mit je Austeilung neu gesetzten
Zufallsströmen fand auf 2.000 Deals keine geänderten Siege, aber rund
+0,1 offizielle Punkte pro ausgeteilter Runde.

Einstellungen, Checkpoint-Hashes und ungerundete Ergebnisse stehen in
[docs/agent-ratings.json](docs/agent-ratings.json). Die Tabelle wird bei neuen
Agenten, Checkpoints und Änderungen an Spielstrategie oder Reizen mit diesem
festen Vergleichsverfahren aktualisiert. Alle aufgeführten Varianten neu messen:

```bash
# System-Python mit installiertem Torch; ein neues Ausgabeverzeichnis verwenden.
python -m scripts.rate_agents \
  --out benchmark_runs/readme-ratings-run-1 --device cpu

# Einzelvergleich nach demselben Verfahren, beispielsweise MCTS:
python -m skat play --agents mcts random random \
  --bidding0 conservative --bidding1 conservative --bidding2 conservative \
  --rounds 1000 --seed 42
```

Der Rating-Lauf benötigt die in [scripts/rate_agents.py](scripts/rate_agents.py)
aufgeführten lokalen Checkpoints. Er speichert vollständige Rundenergebnisse im
Ausgabeverzeichnis und aktualisiert die Tabelle sowie den Ergebnisbericht nach
Abschluss.

## Installation

Python >= 3.12.3 und [uv](https://docs.astral.sh/uv/):

```bash
uv sync                         # Engine + Tests, ohne Torch
uv sync --extra cpu             # CPU-Torch
uv sync --extra cuda            # Linux/Windows: CUDA 13.0
uv sync --extra rocm            # Linux: ROCm 7.2
uv sync --extra sgf             # zusätzlich rohe ISS-SGF-Dateien lesen
uv sync --extra cpu --extra sgf # CPU-Torch und SGF-Import
```

Die normale Installation braucht keine KI-Checkpoints. MCTS läuft ebenfalls ohne
Torch. Für Policy-Training und Inferenz genau eines der Extras `cpu`, `cuda`
oder `rocm` wählen. `cpu` installiert auf Linux/Windows CPU-Torch, `cuda` den
CUDA-13.0-Build auf Linux/Windows und `rocm` den ROCm-7.2-Build auf Linux.
`sgf` lässt sich mit jedem davon kombinieren.
`uv sync --extra cpu` verwendet die im Projekt gesperrte CPU-Version und ersetzt
kein systemweit installiertes ROCm-Torch. Wer bereits ein GPU-Torch im
System-Python hat, startet Training und Inferenz mit diesem `python` statt mit
`uv run`. `pysgf` wird nur zum Einlesen roher ISS-SGF-Dateien benötigt;
vorbereitete Korpora und die Spiel-Engine brauchen es nicht.

Ein dokumentierter Trainingslauf mit **2.000 MCTS-Simulationen, 10.000 Runden
und 100 Epochen** steht in [docs/teacher-training.md](docs/teacher-training.md),
mit Lehrerprüfung und Befehlen für alle vier Arbeitsabläufe.

Eine eigenständige Policy aus dem ISS-Archiv ist in
[docs/iss-training.md](docs/iss-training.md) beschrieben, einschließlich
automatischem Audit, Training, Evaluation und Partien.

Ein optionales ISS-Modell zur Vorhersage versteckter Hände und zur Gewichtung
der MCTS-Suchwelten steht in [docs/belief-sampling.md](docs/belief-sampling.md).
Ohne `--belief-model PATH` nutzt MCTS weiterhin den gleichverteilten Sampler.

Reizen lässt sich unabhängig vom Kartenagenten mit einem ISS-Handklassifikator
konfigurieren. Training und Vergleiche stehen in
[docs/bidding-training.md](docs/bidding-training.md).

Drücken und Ansage können unabhängig von der Kartenstrategie mit
`--declaration0/1/2 heuristic|neural` gewählt werden. Die Heuristik bleibt
Standard; für `neural` ist zusätzlich `--declaration-model PATH` erforderlich.
Das ISS-1M-Ansagemodell gewann mit derselben ISS-Kartenpolicy auf neuen,
gepaarten Deals 3.245 statt 3.195 Solospiele mit dem 300k-Modell und 3.026
mit der Heuristik (12.000 ausgeteilte Kandidatenrunden). Training und Bewertung stehen in
[docs/declaration-training.md](docs/declaration-training.md).
Ein längerer [40-Epochen-Versuch](docs/declaration-40epoch-experiment.md)
verbesserte die Vorhersage, aber auf neuen Deals nicht die Solosiege.

Vor der Skataufnahme kann ein eigener Hand-Klassifikator mit
`--pickup0/1/2 neural --pickup-model PATH` gewählt werden. Er vergleicht die
Gewinnchance von Handspiel und Skataufnahme anhand der ursprünglichen zehn
Karten und des Reizwerts. Standard bleibt Skataufnahme; Daten, Grenzen und
Bewertung stehen in [docs/pickup-training.md](docs/pickup-training.md).

## Vier Arbeitsabläufe

```bash
# Zufälliges legales Kartenspiel mit konservativem Reizen
uv run python -m skat play --rounds 33 --seed 7

# Mensch an Sitz 0, Suchagent und Random als Gegner
uv run python -m skat play --agents cli mcts random

# MCTS-Selbstspiel auf allen verfügbaren CPUs; neues Verzeichnis pro Lauf
uv run python -m skat collect --rounds 1000 --seed 1 --out data/run-1

# Transformer auf MCTS-Besuchszahlen; ISS-Korpora verwenden aufgezeichnete Karten
uv run --extra cpu python -m skat train \
  --data data/run-1 --out trained_models/policy.pt --seed 1

# Ein Modell dient allen drei Sitzen
uv run --extra cpu python -m skat play \
  --agents policy random random --policy trained_models/policy.pt

# Verschiedene Modelle an Sitz 0 und 1, MCTS an Sitz 2
uv run --extra cpu python -m skat play \
  --agents policy policy mcts \
  --policy0 trained_models/transformer-run-2-100epochs.pt \
  --policy1 trained_models/transformer-run-2.pt

# 1000 gemeinsame Deals, alle drei Sitze, Random- und Policy-Gegnergruppen
uv run --extra cpu python -m skat evaluate \
  --policy trained_models/policy.pt --deals 1000 --seed 7 \
  --out benchmark_runs/comparison.json
```

`collect` verteilt vollständige Runden auf separate Prozesse und nutzt
standardmäßig alle verfügbaren CPUs. Mit `--workers 8` begrenzt du die
Prozesszahl; `--workers 1` sammelt seriell. Seeds und Geberrotation sind
an den globalen Rundenindex gebunden, sodass die Workerzahl die Spiele
und Suchentscheidungen nicht verändert. Shards werden beim Laden in globaler
Rundenreihenfolge sortiert, damit auch die Trainingsdatenreihenfolge und der
Split unabhängig von der Workerzahl bleiben.

Auch `play` und `evaluate` nutzen bei MCTS standardmäßig alle verfügbaren CPUs:
Die unabhängigen Suchwelten eines Kartenentscheids laufen in separaten Prozessen.
Ein Prozesspool wird über die gesamte Partie und alle MCTS-Sitze wiederverwendet.
Mit `--workers 8` begrenzt du ihn, mit `--workers 1` suchst du seriell. Höchstens
ein Worker pro Determinisierung arbeitet gleichzeitig; die Voreinstellung sind
20 Suchwelten. Bei `collect` bleibt es bei parallelen Runden, ohne zusätzliche
Prozesspools innerhalb der Worker.

Suchwelten haben eigene Zufallsströme: Derselbe Seed liefert mit einem oder
mehreren Workern identische Besuchszahlen, Werte und Kartenentscheidungen.
Diese Aufteilung ändert die konkreten Seed-Ergebnisse gegenüber älteren
Commits; neue Korpus- und Evaluationsmanifeste halten sie fest.

`collection.json` im neuen Korpusverzeichnis protokolliert Einstellungen,
abgeschlossene Runden, Entscheidungen und Laufstatus. Bei Abbruch werden
verwertbare Shards erhalten; ein neuer Lauf benötigt ein neues Verzeichnis.

`play` und `collect` zeigen eine Fortschrittsleiste mit abgeschlossenen und
verbleibenden Runden sowie geschätzter Restzeit.

Bei `play` setzt `--policy PATH` ein gemeinsames Modell für alle Policy-Sitze.
`--policy0`, `--policy1` und `--policy2` überschreiben es für den jeweiligen Sitz.
Alle drei Sitze sind gleichwertig; die Parteien ergeben sich pro Runde aus dem
Reizen. Ein Modell ist somit weder auf einen Sitz noch auf eine Partei festgelegt.

`--help` zeigt nur die Parameter des jeweiligen Befehls. Alte Aufrufe mit
`-p0`, `-m0`, Reizstilen oder RL-Hyperparametern werden nicht weiter unterstützt.
Fehlende und inkompatible Modelle führen zu einem Fehler. Historische ISS-Pickle-
Dateien ohne vollständige Ansageinformationen müssen vor dem Replay mit dem
aktuellen Parser erneut aus SGF erzeugt werden.

Erweiterte Einstellungen kommen aus einer TOML-Datei:

```toml
[mcts]
simulations = 600
determinizations = 20

[train]
epochs = 12
batch_size = 256
learning_rate = 0.0003
dropout = 0.1
d_model = 128
nhead = 4
num_layers = 4
threads = 1
```

Die Datei wird mit `--config settings.toml` am jeweiligen Befehl angegeben.
Unbekannte Einstellungen werden abgelehnt; Defaults stehen zentral in
`skat/config.py`. Training und Policy-Spiel unterstützen `--device cpu`,
`--device cuda` oder `--device auto`.

## Regeln und Schnittstellen

Referenz ist die [Internationale Skatordnung November 2022](https://dskv.de/app/uploads/sites/43/2022/11/ISkO-2022.pdf).
Der [Regelabgleich](docs/skatordnung.md) beschreibt Implementierung und Grenzen.

- Farbspiele und Grand: Skataufnahme, Hand, zulässige Ansagen und Ouvert.
- Null: Null, Null Hand, Null ouvert und Null Hand ouvert.
- Sagen/Hören, vollständige Reizwertliste einschließlich der Nullwerte,
  Eingepasst, Bedienpflicht und korrekte Stichreihenfolge.
- Spitzen einschließlich Skat, erreichte Gewinnstufen, doppelte Verlustwertung
  und endgültige Überreizwertung. Ein erreichter Schneider kann ein höheres
  Gebot decken; ein gegnerischer Stich ohne Augen verhindert Schwarz.
- Aufgabe mit erforderlicher Zustimmung, Ansprüche auf vorzeitigen Gewinn und
  Abkürzung durch Aufdecken eines Gegenspielers. Letztere prüft alle legalen
  Fortsetzungen und kann bei großen Resthänden aufwendig sein.

```python
from skat.game import Round
from skat.rules import GameDeclaration

round_ = Round(
    skip_bidding=True,
    solo_player_id=0,
    declare_game=GameDeclaration("grand", hand=True, ouvert=True),
    seed=7,
)
print(round_.result)  # RoundResult: Partei, Grund, Augen, Stiche, Wert und Score
```

`prepare()` richtet eine Runde bis zum ersten Ausspiel ein; `play_card(seat,
card)` führt eine geprüfte Aktion aus. `concede()`, `claim()` und
`open_defender()` bilden Spielabkürzungen ab. Kartenentnahme erfolgt ausschließlich
durch die Engine. Ungültige Aktionen ändern den Spielzustand nicht.

Agenten bekommen `PlayerView`: eigene Karten, öffentliche Stiche, Ansage,
Handgrößen, offenliegende Karten und rechtmäßig bekannte Skatkarten. Die Ansicht
enthält keinen Zugriff auf die anderen Hände oder den echten verborgenen Spielwert.
Automatische Agenten reizen standardmäßig konservativ und verwenden dieselbe
Heuristik für Skataufnahme, Drücken und Ansage. Die Engine unterstützt alle
Spielarten; die automatische Heuristik wählt Farbspiel oder Grand mit
Skataufnahme. Menschen und die Python-API können sämtliche Varianten wählen.

Ein optionaler ISS-Klassifikator bewertet die Gewinnchance des ursprünglichen
Zehn-Karten-Blatts. Er steuert nur, ob ein automatischer Agent das Reizen
fortsetzt; die konservative Reizgrenze und die passende Spielansage bleiben
erhalten. So kann man den Reizer unabhängig vom Kartenagenten pro Sitz setzen:

```bash
uv run --extra cpu python -m skat play \
  --agents policy random random \
  --policy trained_models/policy.pt \
  --bidding0 neural \
  --bidding-model trained_models/bidding.pt \
  --bidding-threshold 0.55 \
  --rounds 1000 --seed 42
```

`--bidding0`, `--bidding1` und `--bidding2` nehmen `conservative` oder `neural`;
der Standard ist `conservative`. Der Klassifikator wird mit `--bidding-model`
geladen; `--bidding-threshold` setzt die Mindestgewinnchance für das Reizen.
Die Standardschwelle ist 2/3. Die Zusammenfassung zeigt den
Reizer pro Sitz samt Schwelle. Für `evaluate` gelten `--bidding neural`,
`--bidding-model` und `--bidding-threshold` nur für den Kandidaten; die beiden
Referenzagenten reizen konservativ. Details und Benchmarks stehen in
[docs/bidding-training.md](docs/bidding-training.md).

## Training und Vergleich

KI-Ziel ist der regelgerechte Parteisieg, ohne zusätzliche Bonusnetze. Offizielle
Spielpunkte werden unabhängig davon ausgewiesen. Die Beobachtung ist egocentrisch
und enthält Ansagen, Reizwert, Stichzahlen, offene Hände und bekannte Skatkarten.

Datensätze und Modelle verlangen exakt passende Regel-, Beobachtungs- und
Zielformate. Vor der Konsolidierung erzeugte Artefakte bleiben im alten Branch;
für diesen Stand müssen Daten neu gesammelt und Modelle neu trainiert werden.
Die [abgeschlossene Validierung](docs/validation.md) enthält die gemessenen
Ergebnisse dieses Standes einschließlich des verbleibenden Policy-/MCTS-Abstands.
Der Split trennt vollständige Runden, auch wenn eine Runde mehrere Shards berührt.
Shards und Checkpoints werden atomar geschrieben; Ctrl-C sichert gepufferte Daten.
Training speichert die vollständige Lernkurve nach jeder Epoche neben dem
Checkpoint als `.training.json`. Die KL-Metrik zeigt den Abstand zu den
Besuchsverteilungen nach Abzug ihrer Entropie; die Spielstärke prüft die Evaluation.

Evaluation speichert einzelne Ergebnisse, Seeds, Rollen, offizielle Punkte,
Laufzeiten und Modellhashes in JSON. Unterschiede werden auf gemeinsamen
Deal-/Sitz-/Gegnergruppen berechnet. Das Bootstrap-Intervall zieht vollständige
Deals, weil die sechs Ergebnisse pro Deal zusammenhängen. Der McNemar-Wert und
die Wilson-Intervalle sind zusätzliche deskriptive Kennzahlen. Kleine Testmodelle
oder Smoke-Runs sind kein Nachweis besserer Spielstärke.

Für den Vergleich verschiedener Checkpoints hält `evaluate --reference-policy
PATH` die zwei Policy-Gegner auf einem festen Modell. `--policy` wählt weiterhin
das Modell des Kandidaten; ohne `--reference-policy` dient es auch als Referenz.
Beide Modellhashes stehen im Ergebnismanifest. Verwende denselben Gegner,
Seed und dieselbe Dealzahl für alle Kandidaten.
Die Evaluation rotiert den Kandidaten durch alle drei Sitze, während die beiden
anderen Sitze Referenzagenten verwenden. `--opponent-policy` bleibt als Alias
für bisherige Aufrufe gültig.

## Prüfung

```bash
uv run pytest
uv run --extra cpu pytest
uv run ruff check --select F,E9,I .
uv run ruff format --check .
```

Die Tests enthalten nur einen winzigen Trainingsdurchlauf zur Funktionsprüfung.
Datensammlung, längeres Training und Spielstärkebenchmarks laufen ausschließlich
über die separaten CLI-Befehle. Ohne Torch-Extra überspringen die Tests die KI-Pipeline. SGF-Tests sind ohne Parser
optional. Die Engine bildet drei aktive Sitze ab. Physische Gebefehler,
Schiedsrichterverfahren, Ramsch, Kontra/Re und Bockrunden sind nicht enthalten.
