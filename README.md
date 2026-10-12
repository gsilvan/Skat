# Skat

Eine Python-Engine für offizielles digitales Skat mit Kommandozeilenspieler,
Random-Agent, MCTS und Transformer-Policies. Policies lassen sich aus MCTS-
Selbstspiel oder aufgezeichneten ISS-Kartenzügen trainieren. Das Reizen ist
separat konfigurierbar und kann einen trainierten ISS-Handklassifikator nutzen.

## Installation

Python >= 3.12.3 und [uv](https://docs.astral.sh/uv/):

```bash
uv sync                         # Engine + Tests, ohne Torch
uv sync --extra ai              # zusätzlich Policy-Training und Inferenz
uv sync --extra sgf             # zusätzlich rohe ISS-SGF-Dateien lesen
uv sync --extra ai --extra sgf  # beides
```

Die normale Installation braucht keine KI-Checkpoints. MCTS läuft ebenfalls ohne
Torch. Auf Linux/Windows installiert das AI-Extra standardmäßig CPU-Torch;
GPU-Training benötigt einen passenden CUDA/ROCm-Build im verwendeten Interpreter.

Ein dokumentierter Trainingslauf mit **2.000 MCTS-Simulationen, 10.000 Runden
und 100 Epochen** steht in [docs/teacher-training.md](docs/teacher-training.md),
mit Lehrerprüfung und Befehlen für alle vier Arbeitsabläufe.

Eine eigenständige Policy aus dem ISS-Archiv ist in
[docs/iss-training.md](docs/iss-training.md) beschrieben, einschließlich
automatischem Audit, Training, Evaluation und Partien.

Reizen lässt sich unabhängig vom Kartenagenten mit einem ISS-Handklassifikator
konfigurieren. Training und Vergleiche stehen in
[docs/bidding-training.md](docs/bidding-training.md).

## Vier Arbeitsabläufe

```bash
# Zufälliges legales Kartenspiel mit konservativem Reizen
uv run python -m skat play --rounds 33 --seed 7

# Mensch an Sitz 0, Suchagent und Random als Gegner
uv run python -m skat play --agents cli mcts random

# MCTS-Selbstspiel auf allen verfügbaren CPUs; neues Verzeichnis pro Lauf
uv run python -m skat collect --rounds 1000 --seed 1 --out data/run-1

# Transformer auf MCTS-Besuchszahlen; ISS-Korpora verwenden aufgezeichnete Karten
uv run --extra ai python -m skat train \
  --data data/run-1 --out trained_models/policy.pt --seed 1

# Ein Modell dient allen drei Sitzen
uv run --extra ai python -m skat play \
  --agents policy random random --policy trained_models/policy.pt

# Verschiedene Modelle an Sitz 0 und 1, MCTS an Sitz 2
uv run --extra ai python -m skat play \
  --agents policy policy mcts \
  --policy0 trained_models/transformer-run-2-100epochs.pt \
  --policy1 trained_models/transformer-run-2.pt

# 1000 gemeinsame Deals, alle drei Sitze, Random- und Policy-Gegnergruppen
uv run --extra ai python -m skat evaluate \
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
uv run --extra ai python -m skat play \
  --agents policy random random \
  --policy trained_models/policy.pt \
  --bidding0 neural \
  --bidding-model trained_models/bidding.pt \
  --bidding-threshold 0.55 \
  --rounds 1000 --seed 42
```

`--bidding0`, `--bidding1` und `--bidding2` nehmen `conservative` oder `neural`;
der Standard ist `conservative`. Der Klassifikator wird mit `--bidding-model`
geladen und mit `--bidding-threshold` kalibriert. Die Zusammenfassung zeigt den
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
uv run --extra ai pytest
uv run ruff check --select F,E9,I .
uv run ruff format --check .
```

Die Tests enthalten nur einen winzigen Trainingsdurchlauf zur Funktionsprüfung.
Datensammlung, längeres Training und Spielstärkebenchmarks laufen ausschließlich
über die separaten CLI-Befehle. Ohne AI-Extra überspringen die Tests die KI-Pipeline. SGF-Tests sind ohne Parser
optional. Die Engine bildet drei aktive Sitze ab. Physische Gebefehler,
Schiedsrichterverfahren, Ramsch, Kontra/Re und Bockrunden sind nicht enthalten.
