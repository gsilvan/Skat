# Direkter Vergleich der Policy-Konfigurationen

Die feste README-Tabelle beantwortet eine bestimmte Frage: Wie viele offizielle
Punkte erzielt ein Kandidat an Sitz 0 gegen zwei konservative Random-Spieler?
Sie ist kein allgemeines Ranking gegen starke Gegner. Der direkte Vergleich
hier hat deshalb einen eigenen Namen, eigene Ergebnisse und andere Referenzen.
Das bisherige GUI-Preset „Beste gemessen“ heißt deshalb jetzt präziser
„Beste Policy gegen Random“; seine Einstellungen bleiben dieselben.

## Prüfung des bisherigen Benchmarkings

- Die Random-Tabelle hat reproduzierbare Austeilungen und berücksichtigt
  eingepasste Runden. Schwache Gegner können aber andere Fehler belohnen oder
  bestrafen als eine Policy. Ihre Rangfolge muss sich nicht übertragen.
- Die vollständige Kombination aus neuronalem Reizen, neuronalem Ansagen/Drücken
  und neuronaler Skataufnahme fehlte bisher in der Tabelle. Insbesondere war die
  neuronale Ansage dort überwiegend mit konservativem Reizen kombiniert.
- Ein `play`-Turnier rotiert bereits den Geber. Ein fester Kandidat bleibt also
  nicht ständig Vorhand oder Hinterhand. Trotzdem bekommen unterschiedliche
  Agenten im einzelnen Turnier unterschiedliche Hände. Dass dieselbe Strategie
  an zwei Sitzen unterschiedliche Punktzahlen erreicht, beweist keine Verzerrung.
  Die CLI-Zusammenfassung nennt jetzt zusätzlich Ansage- und Aufnahmestrategie,
  damit drei `PolicyAgent`-Zeilen im kopierten Ergebnis unterscheidbar bleiben.
- Die Gegner im gemischten Tisch sind unterschiedlich: „Alles neuronal“ tritt
  gegen zwei heuristische Ansager an, jeder heuristische Ansager unter anderem
  gegen den neuronalen Ansager. Ihre Gegenspielquoten sind deshalb keine
  isolierte Messung der Kartenstrategie. Alle drei nutzen dasselbe Kartenmodell.
- `evaluate` spielt dieselben Austeilungen an allen Kandidatensitzen und setzt
  Zufallsströme je Austeilung zurück. Seine Policy-Referenzen verwenden aber
  konservatives Reizen, heuristische Ansage und ständige Skataufnahme. Das ist
  nicht der vom Nutzer gespielte Vergleich mit neuronalem Reizen an allen Sitzen.
- Die festen README-Läufe übernehmen die RNG-Ströme der Random-Gegner von Runde
  zu Runde. Andere Spielverläufe können deren Zufallsverbrauch in späteren Runden
  verschieben. Das ist weiterhin ein gültiger Turnierlauf, aber keine saubere
  gepaarte Gegenprobe. Der neue Vergleich setzt die Ströme je Austeilung zurück.
- Eine Wilson-Spanne über sämtliche Sitzwiederholungen behandelt korrelierte
  Beobachtungen als unabhängig. Beim neuen gepaarten Vergleich werden deshalb
  immer komplette Austeilungen einschließlich aller Sitzwiederholungen gezogen.
- Offizielle Punkte pro ausgeteilter Runde zählen Gegenspiel als null. Zusätzlich
  werden daher Gegenspielergebnisse und der Abstand zu den durchschnittlichen
  Gegnerpunkten am selben Tisch berichtet. Eine Sortierung allein nach der
  bedingten Solosiegquote würde außerdem selektiveres Reizen belohnen.

## Festgelegtes Protokoll

Alle vier Kandidaten nutzen dieselbe ISS-Kartenpolicy und denselben neuronalen
Reizer mit Schwelle 0,55. Ansage und Skataufnahme variieren unabhängig:

| Profil | Ansage/Drücken | Skataufnahme |
| --- | --- | --- |
| `best` (bisher „Beste gemessen“) | Heuristik | Immer aufnehmen |
| `declaration` | Neuronal, 1M/20 Epochen | Immer aufnehmen |
| `pickup` | Heuristik | Neuronal, Mindestvorsprung 0 |
| `all-neural` | Neuronal, 1M/20 Epochen | Neuronal, Mindestvorsprung 0 |

Jeder Kandidat spielt jede Austeilung an allen drei Sitzen gegen ein festes
Referenzpaar. Die drei getrennt ausgewerteten Referenzpaare sind konservative
Random-Spieler, zwei `best`-Spieler und zwei `all-neural`-Spieler. Der Geber
rotiert mit dem Austeilungsindex. Jede Austeilung ist innerhalb eines Seeds über
alle Profile, Sitze und Panels identisch; der Austeilungscode wird gespeichert.

Zufallsströme hängen ausschließlich von Seed, Austeilungsindex und Sitz ab,
nicht von Profilreihenfolge, Workerzahl oder zuvor gespielten Runden. Worker
besitzen eigene Agenten und verwenden jeweils einen Torch-Thread. MCTS läuft
innerhalb eines solchen Spielworkers seriell ohne zusätzlichen Suchpool.

Die vorab festgelegte Hauptgegenprobe ist `all-neural` minus `best`, getrennt
für jedes Referenzpaar. Die anderen Varianten helfen, den Effekt von Ansage und
Skataufnahme auseinanderzuhalten; `all-neural` minus `declaration` isoliert die
Handentscheidung nach neuronalem Ansagen. Schwellen werden auf diesen Deals
nicht nachträglich optimiert.

Die offizielle Punktzahl wird durch **alle ausgeteilten Runden einschließlich
der Pässe** geteilt. Die 95-%-Intervalle stammen aus 2.000 gepaarten Bootstrap-
Wiederholungen ganzer Austeilungen. Drei Wiederholungen derselben Austeilung an
drei Sitzen zählen statistisch als eine unabhängige Austeilung. Die separat
ausgewiesenen gemischten Tische vergleichen Kandidatenpunkte mit den
durchschnittlichen Punkten der beiden Gegner am selben Tisch.
Die Intervalle gelten jeweils für den angegebenen Vergleich; sie sind keine
simultanen 95-%-Grenzen für eine nachträglich ausgewählte Gesamtrangfolge.

## Ergebnisse der Policy-Gegenprobe vom 20.10.2026

Die vollständigen 72.000 Runden auf 2.000 unabhängigen Austeilungen bestätigen
den vom Nutzer beobachteten Vorteil der neuronalen Vorbereitung gegen Policies.
Alle Werte sind offizielle Punkte pro ausgeteilter Runde, einschließlich Pässe.

| Feste Referenzspieler | `best` | `declaration` | `pickup` | `all-neural` | Δ `all-neural` minus `best`, 95-%-Intervall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zwei konservative Random | 12,357 | 12,245 | 12,401 | 12,317 | −0,039 [−0,608; +0,497] |
| Zwei `best`-Policies | 4,414 | 7,188 | 4,453 | 7,285 | +2,871 [+2,099; +3,678] |
| Zwei `all-neural`-Policies | 4,414 | 7,188 | 4,453 | 7,285 | +2,871 [+2,099; +3,678] |

Gegen Policy-Referenzen gewinnt `best` 1.521 von 1.971 Alleinspielen (77,2 %),
`all-neural` 1.680 von denselben 1.971 Alleinspielen (85,2 %). Die Zahl der
gewonnenen Alleinspiele steigt also um 159, bei identischem Reizen und denselben
Händen. Der Punktevorsprung ist auf beiden Seeds vorhanden: +2,890 und +2,851
Punkte pro Runde.

Die neuronale Ansage einschließlich Drücken erklärt +2,773 Punkte pro Runde
[+1,999; +3,589]. Die zusätzliche neuronale Handentscheidung nach neuronalem
Ansagen bringt weitere +0,098 Punkte [+0,033; +0,173], ohne die Gesamtzahl der
Solosiege zu ändern. Hier ist sie ein kleiner Punktgewinn, kein nachgewiesener
Gewinn an Siegquote. Die reine Hand-Variante mit heuristischem Ansager bringt
dagegen +0,038 Punkte [−0,035; +0,096], ohne klare Trennung von null.

Die neuronale Variante wählt andere Spiele und andere gedrückte Karten. Unter
den 1.971 Alleinspielen gegen `best` wählt die Heuristik 1.593 Farbspiele und
378 Grands; `all-neural` wählt 1.751 Farbspiele, 155 Grands und 65 Nullspiele.
Diese nachträgliche Aufteilung beschreibt die Auswahl, sie vergleicht keine
identischen Hände innerhalb einer Spielart.

Unter festen Referenzen sind die Gegenspielergebnisse aller vier Varianten
identisch: Ihre Kartenauswahl im Gegenspiel ist dieselbe Policy. Gegen `best`
sind es jeweils 900/3.942 Gegenspielsiege, gegen `all-neural` jeweils 582/3.942.
Dass die beiden Policy-Panels identische Kandidaten-Solopunkte haben, ist deshalb
erwartbar. Sie sind keine zwei unabhängigen Bestätigungen desselben Effekts.

Gegen Random steigen die Solosiege von 1.807 auf 1.856 bei jeweils 1.918
Alleinspielen, die offiziellen Punkte trennen die Varianten in dieser größeren
gepaarten Gegenprobe aber nicht klar. Ein Siegquotengewinn ist also nicht
automatisch ein Gewinn an Spielpunkten. Die historische README-Messung bleibt
separat: Dort stehen bei 1.000 Runden an Sitz 0 mit Seed 42 weiterhin +13,424
Punkte für `best`, +12,389 für neuronale Ansage und +12,489 für `all-neural`.

Die Zusammenfassungen, Konfigurationen, Intervalle und Prüfsummen der vollständigen
Rundendateien stehen in [preset-benchmark-results.json](preset-benchmark-results.json).
Die Originalrunden bleiben lokal unter `benchmark_runs/presets-paired-20261020/`
erhalten. Die Prüfung bestätigt alle 2.000 Austeilungen als verschieden und
identisch über alle Profile, Referenzpaare und Sitzwiederholungen.

## Kontrollierter Kartenvergleich mit MCTS

Weitere 9.000 Runden auf 1.000 neuen unabhängigen Austeilungen vergleichen die
Kartenstrategien gegen dasselbe Paar vollständig neuronaler Policies. Reizer,
Ansage/Drücken und Handentscheidung sind für Policy und MCTS identisch. In allen
3.000 gepaarten Spielen stimmen Alleinspieler und Spielansage tatsächlich überein.
MCTS verwendet 600 Simulationen, 20 Welten und keinen Belief-Sampler.

| Kandidat gegen zwei `all-neural`-Policies | Punkte / Runde | Solosiege | Gegenspielsiege |
| --- | ---: | ---: | ---: |
| Policy `best` | 4,587 | 761/983 | 260/1.966 |
| Policy `all-neural` | 7,834 | 853/983 | 260/1.966 |
| MCTS mit derselben neuronalen Vorbereitung | 5,937 | 816/983 | 243/1.966 |

Die Policy erzielt gegenüber MCTS **+1,897 Punkte pro Runde** mit einem
95-%-Intervall von **[+1,203; +2,590]**. Der Abstand ist auf beiden Seeds positiv:
+1,937 und +1,857. Auch der relative Punktabstand zu den beiden Gegnern fällt
zugunsten der Policy aus: +2,485 [+1,782; +3,196]. Die Parteisiege steigen um
1,8 Prozentpunkte pro ausgeteilter Runde [+0,73; +2,87 Prozentpunkte].

Damit ist die reine Kartenpolicy in dieser festen Gegnergruppe und bei diesem
Suchbudget besser als MCTS mit denselben vorbereitenden Entscheidungen. Das
Ergebnis behauptet keine Überlegenheit gegenüber beliebigen Suchbudgets,
Belief-Modellen oder anderen MCTS-Konfigurationen.

## MCTS als festes Gegnerpaar

Die weiteren Gegenproben umfassen jeweils 6.000 Runden auf denselben 1.000
MCTS-Austeilungen. Beide Kandidaten verwenden in jeder Gegenprobe denselben
neuronalen Reizer mit Schwelle 0,55 und dieselbe Kartenpolicy.

| Zwei feste MCTS-Referenzen | Policy `best` | Policy `all-neural` | Gepaarter Unterschied, 95-%-Intervall |
| --- | ---: | ---: | ---: |
| MCTS mit neuronaler Vorbereitung, CPU | 6,018 | 8,449 | +2,431 [+1,255; +3,656] |
| Normales MCTS, ROCm-Inferenz der Policies | 5,088 | 7,576 | +2,487 [+1,227; +3,751] |

Auch am gemischten Tisch liegt die vollständig neuronale Policy vorne: Gegen
zwei neuronal vorbereitete MCTS-Spieler erzielt sie 8,449 Punkte pro Runde,
jeder Referenzspieler durchschnittlich 6,383. Ihr relativer Vorsprung beträgt
+2,067 Punkte [+1,432; +2,691]. Gegen normales MCTS sind es 7,576 gegenüber
1,016 Punkten je Referenzspieler, also +6,560 [+5,518; +7,552]. Diese relativen
Werte berücksichtigen auch, welche offiziellen Punkte die Gegenspieler erzielen.
Sie ersetzen nicht die offizielle Spielwertung.

Normales MCTS verwendet konservatives Reizen, heuristische Ansage und ständige
Skataufnahme. Der ROCm-Lauf wurde mit System-Python, Torch 2.14.1 / HIP 7.2.53211
und einer AMD Radeon RX 5600 XT durchgeführt. Er verwendet dieselben
Checkpoint-Hashes, ist aber ein eigener Geräte- und Gegnervergleich. Die
CPU- und GPU-Messungen werden nicht zu einer gemeinsamen Gegenprobe gepoolt.

Insgesamt wurden damit **93.000 Runden auf 3.000 verschiedenen unabhängigen
Austeilungen** ausgewertet. Die drei MCTS-Studien teilen sich 1.000 dieser
Austeilungen. Der ROCm-Funktionstest und die feste README-Messung zählen nicht
zu diesen 93.000 Runden. Sämtliche Gegenproben wurden vollständig bis zur
vorab festgelegten Anzahl ausgewertet; Schwellen und Modelle wurden nicht auf
diesen Ergebnissen nachoptimiert.

Zusätzlich enthält die feste README-Tabelle nun alle drei fehlenden
Policy-Kombinationen und vollständig neuronal vorbereitetes MCTS. Letzteres
erreicht im unveränderten 1.000-Runden-Random-Protokoll +11,895 Punkte pro
Austeilung, die vollständig neuronale Policy +12,489. Die historischen
Varianten, ihre Ergebnisse und ihre Checkpoint-Hashes bleiben erhalten; neu
gemessene Zeilen haben eigene Herkunftsangaben. Ein ergänzender Ratinglauf
schreibt die Austeilungscodes in profilspezifische Logdateien.

## Reproduktion

```bash
uv run --extra cpu python -m scripts.benchmark_presets \
  --out benchmark_runs/presets-paired-NEU \
  --deals 1000 --seeds 20261020 20261021 --workers 16
```

Das sind 2.000 unabhängige Austeilungen, 6.000 Runden pro Kandidat und
Referenzpaar, insgesamt 72.000 ausgespielte Runden. `--deals` gilt **pro Seed**.
`--include-mcts` ergänzt MCTS mit 600 Simulationen und 20 Welten, unter denselben
neuronalen Reiz-, Ansage- und Aufnahmeentscheidungen, als Kandidat und Referenz.
Damit ist auch ein Vergleich der Kartenstrategie möglich, während die vier
Policy-Profile ausschließlich die Entscheidungen außerhalb des Kartenspiels
vergleichen.

Eine gezielte Gegenprobe nur der Kartenstrategie gegen zwei starke, vollständig
neuronale Policy-Referenzen verwendet neue Seeds und hält die anderen
Entscheidungen für MCTS identisch:

```bash
uv run --extra cpu python -m scripts.benchmark_presets \
  --out benchmark_runs/presets-mcts-NEU \
  --deals 500 --seeds 20261022 20261023 --workers 4 \
  --include-mcts --candidates best all-neural --panels all-neural
```

Dies sind 1.000 unabhängige Austeilungen und 9.000 ausgespielte Runden. Die
entscheidende Karten-Gegenprobe ist `all-neural-minus-mcts`; die Variante `best`
bleibt zusätzlich als Bezug zur ursprünglichen Konfiguration enthalten. MCTS
nutzt in dieser Gegenprobe keinen Belief-Checkpoint. Ein Ergebnis daraus darf
nicht ungeprüft auf andere Suchbudgets oder Belief-Sampler übertragen werden.

Die umgekehrte Gegnergruppe, also zwei MCTS-Spieler als festes Referenzpaar,
wird separat mit denselben Austeilungen und denselben neuronalen Strategien
für die Entscheidungen außerhalb des Kartenspiels geprüft:

```bash
uv run --extra cpu python -m scripts.benchmark_presets \
  --out benchmark_runs/presets-vs-mcts-NEU \
  --deals 500 --seeds 20261022 20261023 --workers 8 \
  --candidates best all-neural --panels mcts
```

Das ergänzt 6.000 Runden. Die beiden MCTS-Gegenproben teilen dieselben 1.000
unabhängigen Austeilungen; zusammen sind es daher nicht 2.000 unabhängige Deals.

Als zusätzliche vollständige Agentenkonfiguration wird normales MCTS mit
konservativem Reizen, heuristischer Ansage und ständiger Skataufnahme als
Referenzpaar gemessen. Die oben beschriebene kontrollierte Karten-Gegenprobe
und dieser Vergleich beantworten unterschiedliche Fragen: Die neuronalen
Vorbereitungsmodelle wurden mit einer Policy trainiert; ihre Kalibrierung muss
sich nicht unverändert auf MCTS übertragen.

```bash
taskset -c 0-13 python -m scripts.benchmark_presets \
  --out benchmark_runs/presets-vs-default-mcts-rocm-NEU \
  --device cuda --workers 6 --deals 500 --seeds 20261022 20261023 \
  --candidates best all-neural --panels mcts-default
```

Dieser ROCm-Lauf ergänzt weitere 6.000 Runden. Er verwendet dieselben 1.000
MCTS-Austeilungen, bildet aber einen eigenen Geräte- und Gegnervergleich.

Das neue Ausgabeverzeichnis enthält die festgelegte Konfiguration,
Checkpoint-Hashes, Quellcode-Hash, Python-/Torch-Versionen, vollständige
Rundenergebnisse in `worker-*.jsonl`, Fortschritt und die Zusammenfassung.
Vorhandene Ergebnisse werden nicht überschrieben. Unvollständige oder doppelte
Austeilungs-/Sitzgruppen dürfen nicht in eine fertige Auswertung eingehen.
Lange Vergleiche sind eigenständige Benchmarks; die Tests verwenden nur kleine
Regelspiele und synthetische Gegenproben.

Für die lokale ROCm-Installation kann anstelle von `uv run` das System-Python
mit `--device cuda` verwendet werden. Der vollständige Ablauf mit allen vier
Modellen wurde auf der RX 5600 XT mit einem kleinen ROCm-Probelauf geprüft:

```bash
taskset -c 0-13 python -m scripts.benchmark_presets \
  --out benchmark_runs/presets-rocm-NEU --device cuda --workers 1 \
  --deals 1000 --seeds 20261020 20261021
```

`taskset` lässt auf diesem Rechner zwei der 16 logischen CPU-Kerne frei.
`--workers 1` startet nur einen GPU-Kontext. ROCm führt die Modellinferenz auf der GPU aus;
die eigentliche MCTS-Suche bleibt CPU-Arbeit. CPU- und GPU-Messungen werden
getrennt geführt, da geringfügige numerische Unterschiede Modellentscheidungen
beeinflussen können. Der ROCm-Probelauf mit zwölf Runden ist kein zusätzlicher
Nachweis der Spielstärke.

Die historische README-Tabelle lässt sich unabhängig um die fehlenden Profile
ergänzen, ohne andere gemessene Checkpoints oder ihre Hashes zu entfernen:

```bash
uv run --extra cpu python -m scripts.rate_agents \
  --out benchmark_runs/readme-presets-NEU --device cpu \
  --profiles policy-iss-neural policy-iss-neural-bidding-declaration-1m \
  policy-iss-neural-bidding-pickup policy-iss-all-neural mcts-600-all-neural
```

Die konservative Random-Baseline wird stets erneut gemessen. Bei Teilupdates
steht die Herkunft jeder alten und neuen Messung direkt am jeweiligen Profil
in `docs/agent-ratings.json`. Das feste Protokoll bleibt unverändert.
