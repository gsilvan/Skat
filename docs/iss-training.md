# Transformer aus ISS-Spielzügen

Der Import lernt Kartenspiel aus tatsächlich aufgezeichneten Entscheidungen.
Bot- und Menschenspiele bleiben zusammen; Spielernamen sind keine verlässliche
Bot-Klassifikation und alle Akteure werden zunächst als `unknown` geführt.
Gewonnene und verlorene Spiele sowie alle drei Sitze liefern Beispiele.
Reizung, Drücken und Ansagen lernen wir hier nicht.

## Vollständiger automatischer Lauf

Die vorhandenen Vergleichsmodelle müssen existieren. System-Python benutzt den
GPU-Torch-Build; die Projektumgebung liefert den optionalen SGF-Parser:

```bash
uv sync --extra sgf
python scripts/run_iss_experiment.py \
  --input /tmp/iss-games-2007-05-2026.bz2 \
  --sgf-python .venv/bin/python \
  --out benchmark_runs/iss-run-1 --device cuda
```

Der Ablauf prüft das gesamte Archiv, wählt höchstens 100.000 validierte Spiele,
trainiert 20 Epochen mit `configs/iss.toml` und bewertet den besten
Validierungs-Checkpoint. Anschließend folgen zwei Evaluationen über je 1.000
Deals, der gepaarte Vergleich und 1.000 Partien ISS / bisherige Policy / MCTS.
Alle Ergebnisse einschließlich Korpus und Modell liegen unter dem angegebenen
Run-Verzeichnis. Pro Stufe gibt es ein Log; `status.json` dokumentiert Fortschritt
und Fehler. Bei Fehlern startet keine nachfolgende Stufe. Bestehende Lauf- und
Korpusverzeichnisse werden nicht überschrieben.

Die Referenz für die beiden übrigen Policy-Sitze ist immer
`trained_models/transformer-run-2.pt`. Die bisherige Kandidaten-Policy ist
`trained_models/transformer-run-2-100epochs.pt`, MCTS verwendet 600 Simulationen
und 20 Determinisierungen. Die CPU-Evaluation hält das bekannte Vergleichssetup
bei; das Training verwendet die GPU. Der vollständige Archivaudit kann deutlich
länger dauern als der Pilotimport und läuft speicherschonend sequenziell.

Logs verfolgen, wenn ein Lauf im Hintergrund gestartet wurde:

```bash
cat benchmark_runs/iss-run-1/status.json
tail -f benchmark_runs/iss-run-1/train.log
```

## Einzelne Schritte

```bash
.venv/bin/python -m iss audit \
  --input /tmp/iss-games-2007-05-2026.bz2 --out benchmark_runs/iss-audit-full

.venv/bin/python -m iss import \
  --audit benchmark_runs/iss-audit-full \
  --out data/iss-pilot --max-games 100000 --seed 7

python -m skat train --config configs/iss.toml \
  --data data/iss-pilot --out trained_models/iss-pilot.pt \
  --device cuda --seed 7

python -m skat evaluate --agents random mcts policy \
  --policy trained_models/transformer-run-2-100epochs.pt \
  --reference-policy trained_models/transformer-run-2.pt \
  --deals 1000 --seed 42 --device cpu \
  --out benchmark_runs/iss-baseline-eval.json

python -m skat evaluate --agents policy \
  --policy trained_models/iss-pilot.pt \
  --reference-policy trained_models/transformer-run-2.pt \
  --deals 1000 --seed 42 --device cpu \
  --out benchmark_runs/iss-pilot-eval.json

python -m skat.compare \
  benchmark_runs/iss-baseline-eval.json benchmark_runs/iss-pilot-eval.json \
  --out benchmark_runs/iss-comparison.json

python -m skat play --agents policy policy mcts \
  --policy0 trained_models/iss-pilot.pt \
  --policy1 trained_models/transformer-run-2-100epochs.pt \
  --rounds 1000 --seed 43 --device cpu
```

Für Vergleiche muss der Engine-Code während beider Evaluationen identisch sein.
Das Vergleichswerkzeug prüft Code-, Referenz- und Schema-Hashes, Dealparameter,
Suchkonfiguration sowie vollständige Ergebnisse für beide Panels und alle Sitze.
Alte Berichte vor dieser Codeänderung werden daher neu erzeugt.

## Datenprüfung und Split

Der [offizielle ISS-Client](https://skatgame.net/iss/download.html) dokumentiert
Sitz 0 als Vorhand, Reizung und Ansageflags. Unser eigener Parser rekonstruiert
diese Zustandsfolge einschließlich ursprünglichem Skat und gedrückten Karten;
die alten `ISSGames`- und Gewinner-Extraktionsfunktionen werden nicht verwendet.

Jedes Spiel wird vollständig mit aufgezeichneten Karten nachgespielt. Besitz,
Bedienpflicht, Zugreihenfolge, Ende, Augen, Stiche, Spitzen und Spielwert müssen
passen. Sonderereignisse wie Show Cards, Aufgabe, Timeout oder unbekannte
Aktionen werden in dieser Version mit dem ganzen Spiel ausgeschlossen. Fehler
sind im SQLite-Index und Auditbericht gezählt; es entstehen keine Beispiele aus
unvollständigen oder widersprüchlichen Spielen.

Vor einer nicht erzwungenen Entscheidung speichern wir die 376er Observation,
die Legalitätsmaske und den Kartenindex. Gegnerkarten und zukünftige Ereignisse
bleiben unsichtbar. Ouvert und bekannte Skatkarten folgen der aktuellen Engine.
Das Format `iss-action-v1` verwendet speicherabbildbare NumPy-Shards und
Spielmetadaten in `games.jsonl`. Training liest höchstens einen Shard plus Batch,
nicht den gesamten Korpus. ISS nutzt maskierte Cross-Entropy auf der realen
Aktion; bestehende MCTS-Korpora behalten ihr bisheriges Besuchszahlen-Ziel.

Die Pilotwahl erfolgt deterministisch anhand von Seed und Spiel-ID über das
gesamte validierte Archiv. Exakte ID-Duplikate werden entfernt; widersprüchliche
IDs werden ausgeschlossen. Wiederholte Ausgangsverteilungen werden auch bei
rotierten Sitzen oder anderer Kartenreihenfolge erkannt. Sessions und gleiche
Verteilungen bilden transitive Gruppen. Zeitgrenzen liegen bei ungefähr 80/90
Prozent des vollständigen validierten Archivs. Gruppen über einer Grenze werden
vollständig ausgeschlossen. Die tatsächlichen Größen und Ausschlüsse stehen
im Manifest; ein leerer Split führt zu einem Fehler statt einem zufälligen Ersatz.

Validierungs-Loss wählt den Checkpoint. Erst anschließend wird der eingefrorene
zeitliche Testsplit einmal ausgewertet. `.training.json` enthält Verlauf,
Datenherkunft, Hashes und Loss/Übereinstimmung nach Partei und Spielart.
Einzelne Expertenzüge können mehrere gleichwertige Alternativen haben; erst
Gewinnraten und gepaarte Intervalle gegen feste Referenzen belegen Spielstärke.

### Austeilungscodes für ISS

Der strikte Parser in `iss.protocol` bewahrt die ursprünglichen 32 Karten aus
`MV[w ...]` in `Record.original` auf. `Record.deal_id` und `Record.deal_code`
verwenden diese Karten; `metadata` enthält zusätzlich `deal_code`.
ISS-Plätze 0/1/2 entsprechen Vorhand/Mittelhand/Hinterhand, daher ist der Geber
im Austeilungscode Platz 2. Eine absolute Tischposition lässt sich daraus nicht
ableiten. Der bestehende rotationsunabhängige `deal_hash` bleibt für die
Datengruppierung unverändert.

Ein solcher Code lässt sich mit `python -m skat play --deal-code CODE` ab dem
Reizen erneut spielen. Er reproduziert keine historischen Entscheidungen.
Der ältere ISS-Import und das reine Karten-Replay verwenden dagegen bereits
gedrückte Hände. Aus diesen allein wird kein ursprünglicher Austeilungscode
erzeugt; dafür werden die Rohdaten beziehungsweise `Record.original` benötigt.

Die optionale SGF-Abhängigkeit bleibt auf `pysgf>=0.9.0,<1` begrenzt: Version
1.0 entfernt die vom vorhandenen Import verwendete `SGF`-API.
