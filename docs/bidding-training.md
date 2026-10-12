# ISS-Handklassifikator für unabhängiges Reizen

Der Kartenagent und seine Reizstrategie sind getrennt. `conservative` entspricht
dem bisherigen `playable()`-Filter mit sicherer Obergrenze. `neural` ersetzt nur
den Filter: Die geschätzte Gewinnwahrscheinlichkeit muss die gewählte Schwelle
erreichen. Auch angenommene Hände behalten ihre sichere Reizgrenze und die
bestehende dazu passende Drück-/Ansageheuristik. Das funktioniert mit Random,
MCTS und beliebigen aktuellen Kartenpolicy-Checkpoints.

Der Klassifikator bekommt die ursprünglichen zehn Karten und die Position
Vorhand/Mittelhand/Hinterhand: 35 Merkmale, keine Gegnerkarten, kein Skat und
keine späteren Ereignisse. Ein kleiner eigener Karten-Transformer liefert einen
Gewinnlogit. Ein Reizcheckpoint ist kein Kartenpolicy-Checkpoint und wird nicht
an einen bestimmten Kartenagenten gebunden.

## Vollständiger Versuch

Der bereits abgeschlossene Archivaudit wird wiederverwendet. Folgender Ablauf
importiert höchstens 200.000 geeignete Spiele, trainiert 20 GPU-Epochen und prüft
das Modell getrennt mit der ISS-Policy, großen MCTS-Policy und MCTS. Danach
folgen 1.000 Partien mit gelerntem Reizen an Sitz 0:

```bash
python scripts/run_bidding_experiment.py \
  --sgf-python .venv/bin/python --device cuda \
  --out benchmark_runs/bidding-run-1
```

Defaults: Audit `benchmark_runs/iss-run-1/audit`, ISS-Kartenpolicy
`benchmark_runs/iss-run-1/policy.pt`, große Kartenpolicy
`trained_models/transformer-teacher-2000-10k.pt`, MCTS mit 2.000 Simulationen und
20 Determinisierungen. Die ISS-Policy dient auf den Referenzsitzen als feste
kompetente Referenz. Fehlende Eingaben werden vor dem Lauf gemeldet.

Pro Stufe gibt es ein Log und `status.json`; bei Fehlern stoppt der Ablauf.
Der Korpus liegt unter `corpus`, das Modell unter `bidding.pt`, der Verlauf unter
`bidding.training.json`. Die drei Benchmarkverzeichnisse enthalten separate
Validierungs-/Testläufe und `report.json`. Bestehende Verzeichnisse bleiben
unberührt. Reale Trainings und diese umfangreichen Benchmarks gehören nicht in
die Testsuite.

## Import und Training einzeln

```bash
.venv/bin/python -m skat.agents.neural_bidding import \
  --audit benchmark_runs/iss-run-1/audit \
  --out data/iss-bidding --max-games 200000 --seed 7

python -m skat.agents.neural_bidding train \
  --config configs/bidding.toml --data data/iss-bidding \
  --out trained_models/iss-bidding.pt --device cuda --seed 7
```

Wir verwenden validierte gewöhnliche aufgenommene Farbspiele und Grand,
Gewinner und Verlierer, ohne Klassenhäufigkeiten künstlich auszugleichen.
Null, Hand und angesagte Sonderziele sind nicht die Spielpläne unserer jetzigen
Automatik und werden deshalb ausgeschlossen. Überreizungsverluste werden über
die rekonstruierte offizielle Wertung erkannt und separat gezählt. Nach deren
Ausschluss können etwas weniger als die angeforderte Zahl von Spielen bleiben.
Nur die tatsächliche Solo-Hand erhält ein Label. Andere Sitze erhalten kein
angenommenes Soloergebnis.

Zeitgrenzen und transitive Session-/Verteilungsgruppen stammen unverändert aus
dem vollständigen Audit. Die historischen Validierungsgruppen werden mit einem
stabilen Hash zusätzlich in Modellvalidierung und Kalibrierung geteilt.
Training, Validierung, Kalibrierung und Test haben ungefähr 80/5/5/10 Prozent.
Der beste Validierungs-Loss wählt den Checkpoint. Erst danach wird auf den
separaten Kalibrierungsgruppen eine positive Temperatur geschätzt und der
historische Test einmal ausgewertet. Log-Loss, Brier-Score und zehn
Wahrscheinlichkeitsbereiche stehen im Trainingsbericht, ebenso ein konstanter
Trainings-Sieganteil als Vergleich. Shards werden speicherabbildbar geladen.

Diese Wahrscheinlichkeiten beschreiben historische ISS-Solos. Spieler haben ihre
Hände bereits ausgewählt und entscheiden anders über Skat und Karten. Das
Modell erhält dadurch sinnvolle Signale, aber keine unverzerrten Gewinnchancen
aller möglichen Hände für jeden beliebigen Agenten. Ob der Filter hilft, zeigt
deshalb der Spielstärkevergleich jeder konkreten Kombination.

## Reizstrategie pro Sitz wählen

```bash
python -m skat play --agents policy policy mcts \
  --policy benchmark_runs/iss-run-1/policy.pt \
  --bidding0 neural --bidding1 conservative --bidding2 neural \
  --bidding-model trained_models/iss-bidding.pt \
  --bidding-threshold 0.70 --rounds 1000 --seed 43
```

Ein unveränderlicher Reizer kann allen neuralen Sitzen dienen. Jeder Sitz speichert
seine einmalige Vorhersage und Reizgrenze für die Runde. Ohne Optionen bleibt
alles konservativ. Menschen reizen weiterhin über stdin.

Python-Aufrufer verwenden zum Beispiel:

```python
bidder = NeuralBidder("trained_models/iss-bidding.pt", threshold=0.70)
agent = MCTSAgent().configure_bidding("neural", bidder)
```

## Erfolg und Wahrscheinlichkeitsschwelle

### Sweep auf einem festen Tisch

Der folgende eigenständige Lauf entspricht `random / random / random`, zunächst
mit konservativen Reizern an allen Sitzen. Danach bekommt **nur Sitz 1** den
Neural-Reizer mit Schwellen 0.00, 0.05, …, 1.00. Jeder der 22 Durchläufe spielt
300 Runden mit Seed 42 und rotierendem Geber; dieselben Deals werden wiederverwendet.
Eingepasste Blätter zählen zu den 300 Runden. An jedem Durchlauf nehmen frische
Agenten mit denselben Ausgangsseeds teil.

```bash
# System-Python mit Torch und Matplotlib; jeweils ein neues Verzeichnis verwenden.
python -m scripts.sweep_bidding_threshold \
  --bidding-model benchmark_runs/bidding-run-1/bidding.pt \
  --out benchmark_runs/threshold-sweep-random-seed42

# Gleiche Random-Gegner, aber eine ISS-Kartenpolicy am untersuchten Sitz 1:
python -m scripts.sweep_bidding_threshold \
  --agents random policy random \
  --policy1 benchmark_runs/iss-run-1/policy.pt \
  --bidding-model benchmark_runs/bidding-run-1/bidding.pt \
  --out benchmark_runs/threshold-sweep-iss-seed42

# Ein Tisch mit drei Policies; --policy0/1/2 erlauben verschiedene Modelle.
python -m scripts.sweep_bidding_threshold \
  --agents policy policy policy --policy benchmark_runs/iss-run-1/policy.pt \
  --bidding-model benchmark_runs/bidding-run-1/bidding.pt \
  --out benchmark_runs/threshold-sweep-iss-table-seed42
```

`--agents` nimmt drei automatische Agenten (`random`, `mcts`, `policy`).
`--policy` ist der gemeinsame Kartencheckpoint; `--policy0`, `--policy1` und
`--policy2` überschreiben ihn pro Sitz. Der jeweils konfigurierte Tisch bleibt
auch im konservativen Baseline-Lauf derselbe. Weitere Optionen sind `--rounds`
(Standard 300), `--seed` (42), `--bidding-seat` (1), `--device` (`cpu`),
`--workers` und `--config configs/teacher-2000.toml` für das MCTS-Suchbudget.

Nach der Baseline und jeder Schwelle wird eine JSON-Datei mit allen
Rundenergebnissen und Kennzahlen der drei Sitze gespeichert. `summary.json`
enthält Fortschritt, Einstellungen, Modellhashes und den besten gemessenen
Threshold nach **offiziellen Gesamtpunkten nach allen Runden am untersuchten
Sitz** sowie die Schwelle mit den meisten Parteisiegen. Bei gleichen Werten
gewinnt die niedrigere Schwelle. Abbrüche erhalten die bereits abgeschlossenen Runden.

`threshold-sweep.csv` enthält Gesamtpunkte, Siege, Niederlagen und eingepasste
Runden. Die Matplotlib-Plots `threshold-sweep.png` und `threshold-sweep.pdf`
zeigen links die Gesamtpunkte und rechts die absoluten Anzahlen der Siege,
Niederlagen und eingepassten Runden. Siege zählen Parteisiege als Alleinspieler
oder Gegenspieler; eingepasste Runden sind keine Niederlagen. Für jeden Sitz
gilt: Siege + Niederlagen + eingepasste Runden = ausgeteilte Runden.
Die konservative Baseline ist jeweils mit eingezeichnet. Eine senkrechte Linie
markiert links die höchsten Gesamtpunkte und rechts die meisten Siege.
Diese beiden Schwellen können verschieden sein.

Für den Einfluss der eigenen Spielstärke bleiben Sitz 0 und 2 unverändert,
während nur Agent/Kartenmodell an Sitz 1 gewechselt werden. Eine Änderung
aller drei Agenten untersucht zusätzlich den Einfluss der Tischbesetzung.
300 Runden dienen der Suche; der gemessene beste Threshold sollte auf mehr
Runden und einem neuen Seed bestätigt werden. Das Intervall ist für jeden
Vergleich einzeln berechnet und berücksichtigt die Auswahl aus 21 Schwellen
nicht. Dieser Sweep ersetzt nicht die README-Ratings mit 1.000 Runden.

Die [Auswertung der Sweeps an Sitz 1](bidding-threshold-sweep/analysis.md)
enthält den Gesamtvergleich und Einzelplots für Random, vier Transformer-
Checkpoints sowie MCTS mit 600 und 2.000 Simulationen. Die zugehörigen
Kennzahlen, Modellhashes und PDF-Plots sind dort ebenfalls gespeichert.

### Gemeinsamer Sweep aller drei Random-Sitze

```bash
python -m scripts.sweep_bidding_table \
  --bidding-model benchmark_runs/bidding-run-1/bidding.pt \
  --rounds 300 --seed 42 --workers 16 \
  --out benchmark_runs/threshold-table-random-seed42
```

Dieser Versuch verwendet ausschließlich Random / Random / Random. Die Baseline
reizt überall konservativ; danach erhält jeder Sitz seinen eigenen Neural-
Threshold. Auf denselben Deals werden zunächst alle 125 Kombinationen von
0.00, 0.25, 0.50, 0.75 und 1.00 gespielt. Anschließend werden einseitige beste
Antworten in 0.05-Schritten untersucht, während die beiden Gegner festbleiben.
Bis zu acht Änderungen suchen eine stabile Kombination; mögliche Zyklen und
verbleibende Verbesserungen werden ausgewiesen. `--step 0.05` untersucht
stattdessen alle 9.261 gemeinsamen Kombinationen und braucht deutlich länger.

Die statischen Matplotlib-Dateien `surface-seat2-*.png` und `.pdf` zeigen je
eine Ansicht: x = Threshold Sitz 0, y = Threshold Sitz 1, z = Gesamtpunkte nach
300 Runden. Jede der drei Farben ist ein Sitz. Der Threshold von Sitz 2 steht
im Titel. Die Flächen verbinden gemessene Rasterwerte, ohne zusätzliche
Zwischenwerte zu simulieren. `selected-surface.png` zeigt den Schnitt durch die
ausgewählte Kombination. `outcomes.png` stellt deren Gesamtpunkte und absolute
Siege/Niederlagen der konservativen Baseline gegenüber.

Zielgröße sind eigene Gesamtpunkte (`--objective wins` wählt stattdessen
Parteisiege). Gleich hohe Punkte am Schnittpunkt zweier Flächen beweisen kein
Gleichgewicht: Entscheidend ist, ob ein Sitz allein durch einen anderen
Threshold besser abschneidet. Ein empirisch stabiles Rasterprofil ist kein
Beweis für ein Nash-Gleichgewicht im vollständigen Spiel. Die ausgewählte
Kombination wird unverändert auf Seed 43 geprüft, einschließlich aller feinen
einseitigen Alternativen. `analysis.md` enthält die Plots und Auswertung,
`summary.json` den Fortschritt und `search.csv` / `validation.csv` die Rohwerte.

Die [Random-Auswertung mit statischen 3D-Plots](bidding-threshold-table/analysis.md)
enthält die gemessenen Gesamtpunkte, Siege/Niederlagen und besten Antworten
auf Suchdeals und neuen Deals.

### Auswahl und unabhängiger Test gegen Referenzgruppen

```bash
python -m skat.agents.neural_bidding benchmark \
  --agent policy --policy benchmark_runs/iss-run-1/policy.pt \
  --reference-policy benchmark_runs/iss-run-1/policy.pt \
  --bidding-model trained_models/iss-bidding.pt \
  --out benchmark_runs/bidding-iss-policy --validation-deals 500 --deals 2000
```

Für die große Kartenpolicy tauscht man `--policy`; für MCTS verwendet man
`--agent mcts --config configs/teacher-2000.toml`. Das Referenzmodell bleibt fest.

Die Ausgangsschwelle ist 2/3: Bei konstantem Spielwert V gleicht
`p*V - 2*(1-p)*V` dort Gewinn und Verlust aus. Die wirklichen Spielwerte variieren,
daher erfolgt die Auswahl empirisch. Jeder Benchmark vergleicht zuerst den
konservativen Reizer und Schwellen 0.55, 0.60, 0.65, 0.70, 0.75 und 0.80 auf
500 gemeinsamen Validierungsdeals (Seed 42). Primär zählen eigene offizielle
Spielpunkte pro ausgeteilter Runde, beide Referenzgruppen gleich gewichtet.
Gleichstand bevorzugt die niedrigere Schwelle. Die Schwelle wird anschließend
fixiert und beide Reizer werden auf 2.000 neuen Deals (Seed 43) getestet.

Alle Sitze werden durchrotiert. Die Referenzsitze bleiben konservativ; nur der
Kandidat erhält den gelernten Filter. `evaluate --bidding neural
--bidding-model PATH --bidding-threshold P` erlaubt auch einen einzelnen Lauf.
Die Manifeste dokumentieren Reizmodell-Hash und Einstellungen; Ergebnisse
enthalten Wahrscheinlichkeit, Reizgrenze, Überreizungen und Soloanteil.

Die Punkteauswertung enthält alle ausgeteilten Runden, auch wenn einer oder
beide Reizer passen. Unsicherheitsintervalle ziehen ganze Deals mit allen Sitz-
und Panelvarianten. Höhere Solo-Siegquoten durch häufigeres Passen allein gelten
nicht als Erfolg. Parteisiege sind zusätzliche beschreibende Kennzahlen. Unsere
aktuelle Wertung gibt Verteidigern keine eigenen Spielpunkte; eine erweiterte
Turnierwertung wäre ein anderes Optimierungsziel.
