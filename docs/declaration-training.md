# Drücken und Spielansage als austauschbare Strategie

Die Kartenstrategie (Random, Policy oder MCTS), der Reizer und die Ansage sind
unabhängig. Standardmäßig nimmt der automatische Agent den Skat auf und nutzt
die bisherige Heuristik. `neural` benötigt einen eigenen Ansagecheckpoint; der
ISS-1M-Checkpoint liegt unter `benchmark_runs/declaration-run-3/declaration.pt`.
Der menschliche CLI-Agent trifft die Entscheidung weiterhin selbst.

Nach der Aufnahme bewertet der neuronale Ansager 66 Drückpaare mit vier
Farbspielen, Grand und Null. Die Eingabe trennt zehn behaltene Karten, zwei
gedrückte Karten und die zwei ursprünglich aufgenommenen Skatkarten. Die
Sitzposition und das vorgeschlagene Spiel sind weitere Merkmale. Kandidaten,
deren sicherer Spielwert den gewonnenen Reizwert nicht deckt, werden vor der
Modellbewertung ausgeschlossen. Bei gleicher Wahrscheinlichkeit entscheidet die
stabile Karten-/Spielreihenfolge. Wenn überhaupt kein Kandidat deckt, fällt der
Agent auf die bisherige Heuristik zurück.

## Daten und Training

Historische ISS-Ergebnisse beschreiben nur das tatsächlich gespielte Paar aus
Drückkarten und Ansage. Für Alternativen nutzt der Import daher vollständige,
regelgeprüfte ISS-Austeilungen und spielt andere Entscheidungen mit einem
festgehaltenen Kartenpolicy-Checkpoint aus. Die Policy sieht dabei nur ihre
reguläre `PlayerView`; Gegnerkarten gelangen nicht in die Merkmale. Alle
historischen Siege und Niederlagen bleiben im Korpus. Die zeitlichen und
Session-/Deal-Gruppen des Audits bleiben getrennt. Standardmäßig werden pro
Spielart vier Drückpaare sowie die bisherige Heuristik ausgewertet. Für große
Korpora begrenzt `--candidate-budget 2` die Simulationen auf die Heuristik und
eine zufällig gezogene legale Alternative pro Austeilung. Der Import
ist ein eigenständiger, potenziell langer Benchmark und gehört nicht zur
Testsuite.

```bash
python -m skat.agents.neural_declaring import \
  --audit benchmark_runs/iss-run-1/audit \
  --archive /home/silvan/Downloads/iss-games-2007-05-2026.bz2 \
  --policy benchmark_runs/iss-run-1/policy.pt \
  --out benchmark_runs/declaration-run-3/corpus \
  --max-games 1000000 --candidate-budget 2 --workers 14 --seed 7

/usr/bin/python -m skat.agents.neural_declaring train \
  --data benchmark_runs/declaration-run-3/corpus \
  --out benchmark_runs/declaration-run-3/declaration.pt --device cuda --seed 7
```

Der Import benötigt die optionale SGF-Abhängigkeit (`uv sync --extra sgf`),
Training und Inferenz benötigen Torch (`uv sync --extra ai`). Der Checkpoint
speichert Modell-, Regel-, Merkmals- und Zielversion sowie den Hash der Policy,
mit der die Gegenproben gespielt wurden. Kalibrierung und Test nutzen getrennte
Audit-Gruppen. Da die Labels mit einer festgelegten Kartenpolicy entstehen,
muss die Spielstärke mit anderen Kartenagenten gesondert geprüft werden. Der
[100k-Lauf](declaration-experiment.md) entstand aus 199.980 Beispielen. Der
[300k-Lauf](declaration-300k-experiment.md) entstand aus 599.948 Beispielen
und gewann auf neuen, gepaarten Deals 3.271 statt 3.155 Solospiele mit dem
100k-Modell und 3.082 mit der Heuristik. Beide Läufe trainierten 20 Epochen
auf System-Torch mit ROCm. Der [1M-Lauf](declaration-1m-experiment.md) gewann
auf einem weiteren neuen Vergleich 3.245 statt 3.195 Solospiele mit 300k.
Ein [40-Epochen-Versuch](declaration-40epoch-experiment.md) auf demselben
1M-Korpus verbesserte die Wahrscheinlichkeitsmetriken, aber nicht die
Solosiege auf neuen Deals. Der 20-Epochen-Checkpoint bleibt die Referenz.

## Agenten verwenden und prüfen

```bash
python -m skat play --agents random policy mcts \
  --policy benchmark_runs/iss-run-1/policy.pt \
  --declaration0 neural --declaration1 heuristic --declaration2 neural \
  --declaration-model benchmark_runs/declaration-run-3/declaration.pt

python -m skat evaluate --agents random policy mcts \
  --policy benchmark_runs/iss-run-1/policy.pt \
  --declaration neural \
  --declaration-model benchmark_runs/declaration-run-3/declaration.pt \
  --out benchmark_runs/declaration-pilot.json
```

Python-Aufrufer können einen `DeclarationScorer` einmal laden und denselben
Scorer mehreren Agenten zuweisen:

```python
from skat.agents.neural_declaring.inference import DeclarationScorer
from skat.agents.random import RandomAgent

scorer = DeclarationScorer("benchmark_runs/declaration-run-3/declaration.pt")
agent = RandomAgent().configure_declaration("neural", scorer)
```

Die Entscheidung für ein Handspiel erfolgt vor Sicht auf den Skat. Beide
automatischen Ansagestrategien nehmen standardmäßig den Skat auf. Der
unabhängige [Hand-/Aufnahme-Klassifikator](pickup-training.md) kann diese
frühere Entscheidung für beide Strategien treffen.
