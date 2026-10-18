# Ansagemodell: eine Million ISS-Partien

Der dritte Lauf erweitert den [300k-Lauf](declaration-300k-experiment.md) bei
gleicher Architektur, eingefrorener ISS-Kartenpolicy, Seed 7 und zwei
ausgespielten Drück-/Ansagekandidaten je Partie auf 1.000.000 auditierte
ISS-Partien. Die zeitlichen Session-/Deal-Gruppen bleiben zwischen Training,
Validierung, Kalibrierung und Test getrennt. Es entstanden 1.999.816
Beispiele: 1.598.992 Training, 99.392 Validierung, 100.864 Kalibrierung und
200.568 Test. Der Import nutzte 14 CPU-Prozesse; 20 Trainingsepochen liefen
mit System-Torch und ROCm auf der AMD Radeon RX 5600 XT. Epoche 20 war der
beste Validierungscheckpoint.

Der Checkpoint unter `benchmark_runs/declaration-run-3/declaration.pt` hat
SHA-256 `02d49c064b82b740f37d8f14ddc86a7bb35162bcc12707498db7fc21c43b2f95`.
Auf **demselben 1M-Testsplit** erzielten die beiden Modelle:

| Ansagemodell | Test-Log-Loss | Test-Brier-Score |
| --- | ---: | ---: |
| 300k | 0,36805 | 0,11788 |
| 1M | **0,35543** | **0,11374** |

Für den Spielvergleich wurden 2.000 neue Deals mit dem vor dem Training
festgelegten Seed 44 auf allen drei Kandidatensitzen gegen Random- und
ISS-Policy-Gegner ausgespielt. Die Kartenpolicy und das konservative Reizen
blieben gleich. Alle Ansager hatten 636 Pässe, 3.788 Soloauftritte und keine
Überreizungen in den 12.000 ausgeteilten Kandidatenrunden.

| Ansager | Gewonnene Solos | Solosiege je ausgeteilter Runde | Offizielle Punkte je Runde |
| --- | ---: | ---: | ---: |
| Heuristik | 3.026 | 25,22 % | 5,007 |
| 300k-Modell | 3.195 | 26,63 % | 6,311 |
| 1M-Modell | **3.245** | **27,04 %** | **6,940** |

Das 1M-Modell gewann **50 Solospiele mehr** als 300k und 219 mehr als die
Heuristik. Gegenüber 300k sind das +0,42 Prozentpunkte Solosiege und +0,629
offizielle Punkte je ausgeteilter Runde. Der gepaarte Deal-Bootstrap ergibt
für Parteisiege auf gespielten Runden +0,18 bis +0,69 Prozentpunkte (95 %)
und für offizielle Punkte +0,374 bis +0,897 je ausgeteilter Runde. Getrennt
nach Gegnergruppe war der Zugewinn gegen Random klein und unsicher (+1
Parteispiel von 5.682 gespielten Runden); gegen die ISS-Policy gewann 1M
49 zusätzliche Partien und das Intervall war positiv. Der Gesamteffekt ist
positiv, aber kleiner als beim Schritt von 100k auf 300k.

Das 1M-Modell sagte in diesem Vergleich 446 Nullspiele an und gewann 311;
300k gewann 265 von 388. Handspiele werden weiterhin nicht modelliert.

Die feste [Agentenstärkemessung](agent-ratings.json) mit 1.000 ausgeteilten
Runden, Seed 42, Kandidat an Sitz 0 und zwei konservativen Random-Gegnern
ergab:

| Kartenagent | Ansager | Parteisiege / gespielt | Solosiege | Punkte / Runde |
| --- | --- | ---: | ---: | ---: |
| Random | Heuristik | 440/946 | 215 | −0,965 |
| Random | 300k | 436/946 | 217 | +0,175 |
| Random | 1M | 443/946 | 203 | −0,616 |
| ISS-Policy | Heuristik | 648/946 | 302 | 10,156 |
| ISS-Policy | 300k | 649/946 | 310 | 9,857 |
| ISS-Policy | 1M | 640/946 | 314 | 11,005 |
| MCTS 600 | Heuristik | 651/946 | 313 | 10,381 |
| MCTS 600 | 300k | 641/946 | 317 | 9,755 |
| MCTS 600 | 1M | 669/946 | 315 | 9,701 |

Alle Varianten hatten 330 Soloauftritte und 54 Pässe. Bei der ISS-Policy
steigen mit 1M die Solosiege und Punkte, aber die Parteisiege in diesem kleinen
Panel sinken. Bei MCTS steigen die Parteisiege, bei Random ist das Bild
ebenfalls gemischt. Die größere gepaarte Seed-44-Messung oben belegt einen
Parteisiegvorteil für die ISS-Kartenpolicy; für Random und MCTS sind größere
eigene Vergleiche nötig.
