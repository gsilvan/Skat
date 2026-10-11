# Skat

Eine Python-Engine für offizielles digitales Skat, mit menschlichem Spieler,
Random-Agent, MCTS und einer aus MCTS trainierten Transformer-Policy.

Der Branch `consolidation` beginnt auf `master` und übernimmt ausgewählte
Korrekturen aus `vibe-coding`. Die Versuche mit DQN, Belief-Netzen, gelerntem
Reizen, alternativen Belohnungen und Weboberfläche bleiben dort erhalten.
Übernahmeentscheidungen stehen in [docs/consolidation.md](docs/consolidation.md).

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

## Vier Arbeitsabläufe

```bash
# Zufälliges legales Kartenspiel mit konservativem Reizen
uv run python -m skat play --rounds 33 --seed 7

# Mensch an Sitz 0, Suchagent und Random als Gegner
uv run python -m skat play --agents cli mcts random

# MCTS-Selbstspiel; jedes Experiment erhält ein neues Verzeichnis
uv run python -m skat collect --rounds 1000 --seed 1 --out data/run-1

# Genau eine Architektur und ein Ziel: Transformer auf MCTS-Besuchszahlen
uv run --extra ai python -m skat train \
  --data data/run-1 --out trained_models/policy.pt --seed 1

# Ein Modell dient allen drei Sitzen
uv run --extra ai python -m skat play \
  --agents policy random random --policy trained_models/policy.pt

# 1000 gemeinsame Deals, alle drei Sitze, Random- und Policy-Gegnergruppen
uv run --extra ai python -m skat evaluate \
  --policy trained_models/policy.pt --deals 1000 --seed 7 \
  --out benchmark_runs/comparison.json
```

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
Automatische Agenten verwenden dieselbe konservative Reiz-/Ansageheuristik;
MCTS und Policy unterscheiden sich ausschließlich im Kartenspiel. Die Engine
unterstützt alle Spielarten; die automatische Heuristik wählt Farbspiel oder Grand
mit Skataufnahme. Menschen und die Python-API können sämtliche Varianten wählen.

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

Evaluation speichert einzelne Ergebnisse, Seeds, Rollen, offizielle Punkte,
Laufzeiten und Modellhashes in JSON. Unterschiede werden auf gemeinsamen
Deal-/Sitz-/Gegnergruppen berechnet. Das Bootstrap-Intervall zieht vollständige
Deals, weil die sechs Ergebnisse pro Deal zusammenhängen. Der McNemar-Wert und
die Wilson-Intervalle sind zusätzliche deskriptive Kennzahlen. Kleine Testmodelle
oder Smoke-Runs sind kein Nachweis besserer Spielstärke.

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
