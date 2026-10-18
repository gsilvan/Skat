# Ansagemodell: ISS 100k, zwei ausgespielte Kandidaten je Partie

Der vorhandene, vollständig auditierte ISS-Korpus lieferte 100.000 aufgenommene
Partien. Für jede Partie wurden die bisherige Drück-/Ansageheuristik und eine
weitere zufällig gezogene, reizwertdeckende Kombination mit der eingefrorenen
ISS-Kartenpolicy ausgespielt. Einige Partien hatten nur einen Kandidaten. Das
ergab 199.980 Beispiele: 159.670 Training, 9.848 Validierung, 10.042
Kalibrierung und 20.420 Test. Die zeitlichen Session-/Deal-Gruppen des Audits
blieben getrennt. Die Policy hatte SHA-256
`ddfd1e531d173b7253c24232b11f9de5ac54e6f1308cbc387a32eec11fe422ff`.

System-Torch 2.14.1 mit ROCm 7.2 trainierte 20 Epochen auf der AMD Radeon RX
5600 XT. Epoche 19 war der beste Validierungscheckpoint. Die Temperatur beträgt
1,04117; auf dem getrennten Testkorpus: Gewinnanteil 47,90 %, Log-Loss 0,37950,
Brier-Score 0,12105. Der Checkpoint hat SHA-256
`306a42c8e6b1de74110575bb062313f84a1cd92af50df1ee82a1af1ddb34d680`.

Im anschließenden gepaarten Vergleich spielte dieselbe ISS-Kartenpolicy mit
konservativem Reizen entweder den heuristischen oder den neuronalen Ansager.
Auf 2.000 neuen Deals, jeweils allen drei Kandidatensitzen und zwei festen
Gegnergruppen, gab es 12.000 ausgeteilte Kandidatenrunden. Beide Varianten
hatten 504 Pässe, 3.832 Soloauftritte und keine Überreizungen.

| Größe | Heuristik | Transformer | Differenz |
| --- | ---: | ---: | ---: |
| Gewonnene Solos | 3.083 | 3.152 | +69 |
| Solosiege je ausgeteilter Kandidatenrunde | 25,69 % | 26,27 % | +0,575 Prozentpunkte |
| Offizielle Punkte je ausgeteilter Kandidatenrunde | 5,421 | 5,655 | +0,234 |

Ein Bootstrap über vollständige Deals ergibt für den Solosieg-Unterschied
ungefähr +0,22 bis +0,94 Prozentpunkte (95 %). Für die offiziellen Punkte ist
das entsprechende Intervall ungefähr −0,205 bis +0,673 Punkte pro Runde. Die
Siegverbesserung ist das priorisierte Ergebnis; die Punkteverbesserung ist in
diesem Vergleich unsicher.

Die neuronale Strategie sagte 80-mal Null an und gewann 35 dieser Spiele. Die
Auswahl aus 396 Möglichkeiten kann einzelne Wahrscheinlichkeiten überoptimistisch
behandeln, auch wenn der allgemeine Test-Brier-Score gut ist. Diese Null-Zahlen
sind deskriptiv; Änderungen an der Auswahl sollten auf neuen Deals geprüft
werden. Handspiele wurden in diesem Versuch nicht gelernt.

Die feste Agentenstärkemessung aus 1.000 ausgeteilten Runden (Seed 42,
Kandidat an Sitz 0, zwei konservative Random-Gegner) zeigt ein gemischtes Bild:

| Kartenagent | Heuristik: Solosiege; Punkte/Runde | Modell: Solosiege; Punkte/Runde |
| --- | ---: | ---: |
| Random | 215; −0,965 | 211; −0,870 |
| ISS-Policy | 302; 10,156 | 307; 9,552 |
| MCTS 600 | 313; 10,381 | 308; 9,245 |

In allen sechs Läufen gab es jeweils 330 Soloauftritte und 54 Pässe. Die
1.000-Runden-Messung ist für kleine Unterschiede volatil; der größere gepaarte
Vergleich oben ist die aussagekräftigere Schätzung für die ISS-Policy. Für Random
und MCTS ist ein eigener größerer gepaarter Vergleich nötig, bevor man einen
Gewinnvorteil des Modells für diese Kartenstrategien behauptet.
