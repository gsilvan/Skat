# Handspiel oder Skataufnahme

Die Entscheidung fällt vor der Einsicht in den Skat. Der bisherige
Ansage-Transformer benötigt zwölf bekannte Karten und einen Drückvorschlag;
seine Gewinnwahrscheinlichkeit kann daher ein Handspiel nicht unmittelbar
bewerten. Ein eigener kleiner Transformer erhält ausschließlich die zehn
ursprünglichen Karten, Sitzposition und den gewonnenen Reizwert. Er schätzt
zwei Gewinnwahrscheinlichkeiten: Skat aufnehmen und Hand spielen.

Die Entscheidung ist unabhängig von Kartenstrategie, Reizstrategie und
Ansager konfigurierbar. Standard bleibt `always` (Skat aufnehmen). Mit
`neural` wählt der Agent ein Handspiel nur bei höherer geschätzter
Gewinnchance und wenn der sichere Mindestspielwert des Handspiels den
Reizwert deckt. `--pickup-margin` verlangt auf Wunsch einen größeren Vorsprung.
Die bisherige Heuristik legt nach einer Hand-Entscheidung Farbspiel oder Grand
fest; Null Hand wird in dieser ersten Version nicht vorgeschlagen.

Der erste Lauf mit 20.000 ISS-Austeilungen erreichte auf den 2.065 getrennten
Test-Austeilungen 1.464 Siege nach Aufnahme und 982 als Handspiel. Die
unabhängig kalibrierten Vorhersagen wählten bei Schwelle null kein Handspiel
und erreichten ebenfalls 1.464 Siege. Die Brier-Werte betragen 0,180 für
Aufnahme und 0,201 für Hand. Das ist kein Nachweis einer Verbesserung;
`always` bleibt die Voreinstellung. Der Modellcheckpoint und sein
Trainingsbericht liegen unter `benchmark_runs/pickup-run-1/`.

Da der Validierungsverlust nach 20 Epochen weiter fiel, wurde derselbe
Korpus mit 60 Epochen erneut trainiert. Der beste Checkpoint (Epoche 58)
verbesserte die Test-Brier-Werte auf 0,175 für Aufnahme und 0,183 für Hand.
Er wählte 69 Handspiele, gewann dabei fünf zuvor verlorene Spiele, verlor
aber acht zuvor gewonnene: 1.461 statt 1.464 Siege. Eine anhand der
Kalibrierungsmenge vorsichtiger gewählte Schwelle brachte auf der Testmenge
ebenfalls keinen Gewinn. Dieser Checkpoint liegt unter
`benchmark_runs/pickup-run-2/pickup.pt`.
Sein SHA-256-Hash ist
`85660fcdecb72fb85a3ceb24af779b175719759f5e6003785cd36ff6d2566831`.

## Spielvergleich

Im festen README-Protokoll (1.000 ausgeteilte Runden, Seed 42, Kandidat an
Sitz 0) wählte der Klassifikator acht Handspiele mit heuristischem Ansager
und acht mit dem 1M-Ansager. Die Solosiege betrugen 306 statt 302 mit
Heuristik und 311 statt 314 mit 1M-Ansager. Die Gesamt-Parteisiege waren
655 statt 648 beziehungsweise 655 statt 640. Diese kleinen Unterschiede
sind nicht direkt kausal interpretierbar: Sobald sich ein Spiel ändert,
verbrauchen die Random-Gegner unterschiedlich viele Zufallszahlen, was ihre
Spielzüge in späteren Runden verschiebt.

Ein eigenständiger gepaarter Vergleich setzt deshalb die Zufallsströme der
Gegner **für jede Austeilung** neu. Auf 2.000 neuen Deals mit Seed 43 und
rotierendem Geber wählte das Modell je 22 Handspiele. Mit heuristischem
Ansager blieben die Parteisiege bei 1.320 und die Solosiege bei 627; mit
1M-Ansager blieben es 1.339 und 646. Auf sämtlichen 4.000 gepaarten
Entscheidungen stimmte der Sieg-/Niederlage-Ausgang der Varianten überein.
Die offiziellen Kandidatenpunkte stiegen durch den höheren Hand-Spielwert
um 215 beziehungsweise 196 Punkte insgesamt, also um rund 0,11 bzw.
0,10 Punkte pro ausgeteilter Runde. Da die Siegquote hier Vorrang hat,
bleibt `always` die empfohlene Einstellung.

Die vollständigen Runden stehen in
`benchmark_runs/pickup-paired-2000-seed43.json`. Der reproduzierbare Lauf:

```bash
.venv/bin/python -m scripts.compare_pickup \
  --policy benchmark_runs/iss-run-1/policy.pt \
  --pickup-model benchmark_runs/pickup-run-2/pickup.pt \
  --declaration-model benchmark_runs/declaration-run-3/declaration.pt \
  --out benchmark_runs/pickup-paired-2000-seed43.json \
  --rounds 2000 --seed 43
```

Wenn derselbe Nicht-Hand-Kontrakt den Reizwert deckt, hat die Aufnahme für
die reine Siegchance einen strukturellen Vorteil: Man kann die zwei
aufgenommenen Karten wieder drücken und dasselbe Blatt spielen, besitzt aber
zusätzlich Information und andere Drückmöglichkeiten. Hand kann bei höherem
Reizwert durch seinen zusätzlichen Spielwert oder für offizielle Punkte
interessant sein. Beides ist im aktuellen konservativen Reizer selten;
deshalb ist die Entscheidung nicht automatisch ein Spielstärkegewinn.

## Trainingsziel

Die ISS-Dateien enthalten sowohl Handspiele als auch Spiele mit
Skataufnahme. Eine Klassifikation der historischen Wahl wäre verzerrt: Es
fehlt das Ergebnis der Alternative und starke Hände werden häufiger als Hand
gespielt. Stattdessen werden beide Möglichkeiten auf derselben geprüften
ISS-Austeilung mit derselben eingefrorenen Kartenpolicy ausgespielt. Bei
Skataufnahme entscheidet standardmäßig der heuristische Ansager über Drücken
und Spiel; optional kann der bestehende Ansage-Transformer dafür geladen
werden. Die Hand-Seite verwendet die bestehende Hand-Heuristik. Der
ursprüngliche ISS-Ausgang ist kein Trainingslabel. Die zeitlichen
Session-/Deal-Gruppen des Audits bleiben getrennt, alle ISS-Siege und
-Niederlagen kommen infrage. Der Import setzt den historischen Reizwert auch
in den Gegenproben, damit Überreizen als Niederlage zählt.

```bash
.venv/bin/python -m skat.agents.neural_pickup import \
  --audit benchmark_runs/iss-run-1/audit \
  --archive /home/silvan/Downloads/iss-games-2007-05-2026.bz2 \
  --policy benchmark_runs/iss-run-1/policy.pt \
  --out benchmark_runs/pickup-run-1/corpus \
  --max-games 20000 --workers 12 --seed 7

.venv/bin/python -m skat.agents.neural_pickup train \
  --data benchmark_runs/pickup-run-1/corpus \
  --out benchmark_runs/pickup-run-2/pickup.pt --device cuda --seed 7 --epochs 60
```

Training, Test und Kalibrierung haben getrennte Deal-Gruppen. Beide
Wahrscheinlichkeiten werden getrennt kalibriert. Der Testbericht nennt neben
Log-Loss und Brier-Wert die Siege der Modellentscheidung, der ständigen
Aufnahme und eines nur als Obergrenze dienenden Orakels, das beide
Gegenprobenausgänge bereits kennt. Eine bessere Prognose allein belegt noch
keinen Spielstärkezuwachs; dafür dient der feste 1.000-Runden-Vergleich.

```bash
.venv/bin/python -m skat play --agents policy random random \
  --policy benchmark_runs/iss-run-1/policy.pt \
  --pickup0 neural --pickup-model benchmark_runs/pickup-run-2/pickup.pt
```

Der Checkpoint gehört weder zur Kartenpolicy noch zum Drückmodell. Seine
Schema- und Zielversion wird beim Laden geprüft. Da die Gegenproben von der
eingefrorenen Kartenpolicy und dem verwendeten Ansager abhängen, muss die
Entscheidung bei anderen Agenten eigenständig gemessen werden.
