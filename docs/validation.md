# Prüfung der Konsolidierung

## Automatisierte Prüfungen

- Vollständige Installation: **235 Tests bestanden**, zusätzlich fünf SGF-Teilprüfungen; etwa acht Sekunden.
- Engine ohne Torch und SGF: **214 Tests bestanden**, zwölf optionale Prüfungen übersprungen; etwa drei Sekunden.
- Ruff (`F,E9,I`), Formatprüfung und `git diff --check`: sauber.
- Im Test enthalten ist nur ein Mini-Durchlauf mit vier Runden, einer Epoche
  und kleinem Netz. Die folgenden langen Läufe fanden separat statt.

## Neue Daten und GPU-Training

1.000 neue Selbstspielrunden in vier unabhängigen Sammlern mit Seeds
20261011–20261014, je 600 Simulationen / 20 Determinisierungen pro Entscheidung.
Daraus entstanden **21.241 Entscheidungen aus 949 gespielten Rundengruppen**.
Erzwungene Züge und eingepasste Runden liefern keine Trainingszeilen.

Zwölf Epochen, 128 Modellbreite, vier Transformerlagen, Batchgröße 256,
AdamW mit Lernrate 0,0003. Der Split trennt Rundengruppen: 17.011 Trainings-
und 4.230 Validierungszeilen. Training mit System-Python 3.14.7 / Torch 2.14.0
auf AMD Radeon RX 5600 XT über ROCm; PyTorch-Gerät `cuda`.
Das nach Validierungsverlust gewählte Modell stammt aus Epoche 11,
mit **34,9 % Übereinstimmung mit den tatsächlichen Lehrerentscheidungen**.

## Vergleich auf gemeinsamen Deals

Seed 20261012, 1.000 gemeinsame Deals, jeder Kandidat an allen drei Sitzen
gegen jeweils zwei Random- beziehungsweise Policy-Gegner: **18.000 Ergebnisse**.
Pro Tabellenzelle wurden 2.856 Runden gespielt und 144 eingepasst.
Alle Agenten verwenden dieselbe konservative Reiz-/Ansageheuristik.
Es gab keine Überreizverluste. Bei allen gespielten Runden sind 120 Augen
und zehn Stiche vollständig abgerechnet.

| Kandidat | Zwei Random-Gegner | Zwei Policy-Gegner |
|---|---:|---:|
| random | 46.0% | 40.6% |
| mcts | 67.9% | 65.0% |
| policy | 50.9% | 45.5% |

Die Werte zählen Parteisiege des Kandidaten, unabhängig vom offiziellen
Soloisten-Score. Gegen Random erreicht MCTS als Soloist 93,4 % und auf
Verteidigung 55,1 %; die Policy erreicht 69,0 % beziehungsweise 41,9 %.

Über beide Gegnergruppen verbessert sich MCTS gegenüber Random um
**23,1 Prozentpunkte** (95%-Intervall des Deal-Bootstraps: 21,8–24,5).
Die Policy verbessert sich um **4,9 Prozentpunkte** (3,7–6,2), bleibt aber
**18,2 Prozentpunkte hinter MCTS**. Die Policy gewinnt damit einen Teil des
Suchvorteils zurück; der Abstand zum Lehrer ist erheblich.

CPU-Evaluation auf acht parallelen Prozessen: MCTS etwa 2,2 und Policy etwa
10,8 Tabellenrunden/s pro aufsummierter gemessener Prozesszeit. Diese Werte
enthalten das Spiel der Gegner und konkurrierende CPU-Last. Sie sind keine
isolierten Inferenzbenchmarks.

Der Vergleich verwendet den Code aus `0b9ba47`. Die späteren Fixes in
`eb9c6c3` betreffen ISS-Verträge und menschliche Gewinnforderungen; beides
kommt in diesen automatischen Spielen nicht vor. Alle acht Teilberichte
haben denselben Code- und Modellhash. Der globale Dealindex samt
Geberrotation bleibt bei der Aufteilung erhalten; ein Regressionstest
prüft Gleichheit mit einem zusammenhängenden Lauf.

Diese Zahlen gelten für die automatische Farb-/Grand-Heuristik mit
Skataufnahme. Null, Hand und Ouvert sind regeltechnisch getestet, aber
ihre Spielstärke ist durch diesen Trainingskorpus nicht belegt. Historische
Ergebnisse aus `vibe-coding` verwenden andere Daten, Budgets und Regeln
und sind damit kein direkter Vergleich.

## Artefakte

Die kompakte, versionierte [Ergebnisdatei](benchmark-results.json) enthält
Manifeste, Modell-/Datenhashes, Rollen, Scores und gepaarte Kennzahlen.
Die großen Artefakte liegen lokal unter `benchmark_runs/consolidation/`:

- `corpus/`: acht neue NPZ-Shards.
- `policy.pt`: frisch trainiertes, versioniertes Modell.
- `results.json`: alle 18.000 Einzelresultate.
- `collection.json`, `train.log` und drei `*-driver.py`: Laufparameter und
  reproduzierbare Aufteilung der separaten langen Läufe.

Diese erzeugten Artefakte sind von Git ausgeschlossen. Frühere Modelle,
Datensätze und der originale Branch `vibe-coding` bleiben erhalten.
