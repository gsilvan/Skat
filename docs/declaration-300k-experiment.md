# Ansagemodell: 300.000 ISS-Partien

Der zweite Lauf nutzte denselben auditierten ISS-Korpus, dieselbe eingefrorene
Kartenpolicy und dieselben Importparameter wie der [100k-Lauf](declaration-experiment.md):
zwei ausgespielte Drück-/Ansagekandidaten je Partie, Seed 7, vollständige
Session-/Deal-Gruppen in getrennten Partitionen. Es wurden 300.000 Partien
ausgewählt: 239.857 Training, 14.958 Validierung, 15.047 Kalibrierung und
30.138 Test. Daraus entstanden 599.948 Beispiele. Der Import lief mit zwölf
CPU-Prozessen; 20 Trainingsepochen liefen mit System-Torch und ROCm auf der AMD
Radeon RX 5600 XT. Epoche 19 lieferte den besten Validierungscheckpoint.

Der Checkpoint unter `benchmark_runs/declaration-run-2/declaration.pt` hat
SHA-256 `4e86497860f7ce4e17f0b776e4f984be69deca239abd1d6b7e3c5949f1157bdb`.
Seine Temperatur beträgt 1,04128. Auf **demselben 300k-Testsplit** erzielten
die beiden Modelle:

| Ansagemodell | Test-Log-Loss | Test-Brier-Score |
| --- | ---: | ---: |
| 100k | 0,38289 | 0,12232 |
| 300k | 0,36855 | 0,11790 |

Die 100k- und 300k-Trainingsgruppen stammen aus derselben Trainingspartition;
keine der beiden Versionen lernte aus dieser Testpartition.

Im gepaarten Spielvergleich wurden 2.000 neue Deals mit Seed 43 auf allen drei
Kandidatensitzen gegen Random- und ISS-Policy-Gegner ausgespielt. Kartenpolicy
und konservatives Reizen blieben gleich. Die 12.000 ausgeteilten
Kandidatenrunden enthielten bei allen Ansagern 540 Pässe, 3.820 Soloauftritte
und keine Überreizungen.

| Ansager | Gewonnene Solos | Solosiege je ausgeteilter Runde | Offizielle Punkte je Runde |
| --- | ---: | ---: | ---: |
| Heuristik | 3.082 | 25,68 % | 5,513 |
| 100k-Modell | 3.155 | 26,29 % | 5,827 |
| 300k-Modell | **3.271** | **27,26 %** | **6,801** |

Das 300k-Modell gewann 116 Solospiele mehr als das 100k-Modell und 189 mehr als
die Heuristik. Gegenüber 100k entspricht das +0,97 Prozentpunkten Solosiegen
je ausgeteilter Runde und +0,974 offiziellen Punkten je Runde. Getrennt nach
Gegnergruppe stiegen die Parteisiege bei gespielten Runden um 0,61
Prozentpunkte gegen Random (Deal-Bootstrap-Intervall 95 %: +0,26 bis +0,96)
und 1,41 Punkte gegen die ISS-Policy (+0,94 bis +1,90). Auch die gepaarten
Punkteintervalle waren in beiden Gegnergruppen positiv.

Das Modell wählte auf diesen Deals 296 Nullspiele und gewann 200 davon. Die
Heuristik wählte keine Nullspiele; das 100k-Modell gewann 43 von 72. Diese
Aufschlüsselung beschreibt den Vergleich, ersetzt aber keine eigene Prüfung
der Null-Entscheidungsregel. Handspiele werden weiterhin nicht modelliert.

Die feste [Agentenstärkemessung](agent-ratings.json) mit 1.000
ausgeteilten Runden, Seed 42, Kandidat an Sitz 0 und zwei konservativen
Random-Gegnern ergab:

| Kartenagent | Heuristik: Solosiege; Punkte/Runde | 100k: Solosiege; Punkte/Runde | 300k: Solosiege; Punkte/Runde |
| --- | ---: | ---: | ---: |
| Random | 215; −0,965 | 211; −0,870 | 217; +0,175 |
| ISS-Policy | 302; 10,156 | 307; 9,552 | 310; 9,857 |
| MCTS 600 | 313; 10,381 | 308; 9,245 | 317; 9,755 |

Alle neun Varianten hatten 330 Soloauftritte und 54 Pässe. Auf diesem kleinen
Panel verbesserte 300k die Solosiege bei allen drei Kartenagenten gegenüber
100k. Die offiziellen Punkte und Parteisiege sind nicht durchgängig besser
als mit der Heuristik. Der größere gepaarte Vergleich oben belegt den Vorteil
für die ISS-Policy auf den neuen Seed-43-Deals; für Random und MCTS braucht es
größere, eigene Vergleiche.
