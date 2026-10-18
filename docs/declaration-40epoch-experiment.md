# Ansagemodell: 1M-Korpus mit 40 Epochen

Der [1M-Lauf mit 20 Epochen](declaration-1m-experiment.md) hatte seine beste
Validierung erst in Epoche 20. Deshalb wurde dasselbe Modell auf demselben
Korpus mit identischem Seed, Optimizer und allen übrigen Einstellungen frisch
für 40 Epochen trainiert. Der alte Checkpoint enthält keinen Optimizer-Zustand;
ein nahtloses Fortsetzen war daher nicht möglich. Die ersten 20 Epochen des
neuen Laufs reproduzierten die bisherigen Trainings- und Validierungswerte.
Der beste Validierungscheckpoint des längeren Laufs stammt aus Epoche 38.

Der neue Checkpoint liegt unter `benchmark_runs/declaration-run-4/declaration.pt`
und hat SHA-256
`45afae968cf1709e1cc68729117e28296b19a35126e1a2ea5403644f4b9b5b94`.
Auf demselben zurückgehaltenen Testsplit verbesserte sich die
Wahrscheinlichkeitsvorhersage:

| Lauf | Test-Log-Loss | Test-Brier-Score |
| --- | ---: | ---: |
| 20 Epochen | 0,35543 | 0,11374 |
| 40 Epochen (bester Checkpoint: Epoche 38) | **0,35013** | **0,11212** |

Für den gepaarten Spielvergleich wurden 2.000 neue Deals mit Seed 45 auf allen
drei Kandidatensitzen gegen Random- und ISS-Policy-Gegner ausgespielt. Beide
Varianten verwendeten dieselbe ISS-Kartenpolicy und konservatives Reizen. Von
12.000 ausgeteilten Kandidatenrunden waren bei beiden 576 eingepasst und 3.808
Soloauftritte; es gab keine Überreizungen.

| Lauf | Gewonnene Solos | Offizielle Punkte je ausgeteilter Runde |
| --- | ---: | ---: |
| 20 Epochen | **3.278** | **7,117** |
| 40 Epochen | 3.272 | 7,002 |

Der längere Lauf gewann sechs Solospiele weniger und erzielte 0,114 Punkte je
Runde weniger. Im gepaarten Deal-Bootstrap liegt das 95-%-Intervall für den
Unterschied der Parteisiege auf gespielten Runden bei ungefähr −0,26 bis
+0,16 Prozentpunkten; für die offiziellen Punkte je ausgeteilter Runde bei
−0,307 bis +0,070. Auf diesen Deals gibt es also keinen belegten
Spielstärkezuwachs durch die zusätzlichen Epochen, obwohl die
Wahrscheinlichkeitsmetriken besser wurden. Der 20-Epochen-Checkpoint bleibt
deshalb die bisherige Referenz.

Die feste [Agentenstärkemessung](agent-ratings.json) über 1.000 ausgeteilte
Runden, Seed 42, Kandidat an Sitz 0 und zwei konservative Random-Gegner zeigt
ebenfalls keinen einheitlichen Vorteil. Alle Varianten hatten 330 Soloauftritte
und 54 Pässe.

| Kartenagent | Epochen | Parteisiege / gespielt | Solosiege | Punkte / Runde |
| --- | ---: | ---: | ---: | ---: |
| Random | 20 | 443/946 | 203 | −0,616 |
| Random | 40 | 434/946 | 208 | −0,453 |
| ISS-Policy | 20 | 640/946 | 314 | 11,005 |
| ISS-Policy | 40 | 660/946 | 310 | 10,243 |
| MCTS 600 | 20 | 669/946 | 315 | 9,701 |
| MCTS 600 | 40 | 651/946 | 311 | 9,757 |

Die Unterschiede dieser kleineren Wertung schwanken je nach Kartenagent und
Kennzahl. Der größere gepaarte Vergleich oben ist die aussagekräftigere
Schätzung für die ISS-Kartenpolicy.
