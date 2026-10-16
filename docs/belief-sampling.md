# ISS-Handvorhersage für MCTS

MCTS verwendet ohne Zusatzoption weiterhin gleichverteilte vollständige Welten.
`--belief-model` aktiviert einen eigenen Transformer, der vor jedem Kartenentscheid
die noch unbekannten Karten auf die beiden anderen Hände und den Skat gewichtet.
Die Bedienpflicht, bekannte Karten, offene Blätter und exakte Handgrößen bleiben
harte Regeln des Samplers. Eine Netzvorhersage kann sie nicht außer Kraft setzen.

## Daten und Training

Der vorhandene ISS-Aktionskorpus speichert keine versteckten Hände. Der
Handkorpus liest deshalb das ursprüngliche SGF-Archiv erneut und nutzt die
bereits geprüften 100.000 Spiel-IDs samt Train/Val/Test-Zuordnung. Der angegebene
Archivpfad muss den SHA-256-Hash der geprüften Auswahl treffen. Pro Partie wird
aus jedem Drittel höchstens ein nicht erzwungener Kartenzug gewählt. Die Eingabe
enthält nur die damalige Spielersicht und zusätzlich die öffentliche Reihenfolge
der ausgespielten Karten; die tatsächlichen Resthände dienen allein als Ziel.

```bash
.venv/bin/python -m iss belief-import \
  --selection benchmark_runs/iss-run-1/corpus \
  --archive /home/silvan/Downloads/iss-games-2007-05-2026.bz2 \
  --out data/iss-belief-100k --seed 7

python -m skat.agents.belief.train \
  --data data/iss-belief-100k \
  --out trained_models/iss-belief-100k.pt \
  --epochs 12 --d-model 64 --num-layers 2 \
  --device cpu --seed 7
```

Das Modell minimiert maskierte Cross-Entropy für den tatsächlichen Ort jeder
unbekannten Karte. Der Checkpoint der Epoche mit dem kleinsten
Validierungsverlust wird gewählt. Danach wird die Stärke der Gewichtung auf
einer festen Stichprobe der Validierung anhand der Wahrscheinlichkeit der ganzen
realen Verteilung gewählt; Stärke null ist exakt der gleichverteilte Sampler.
Der zeitliche Testsplit wird erst anschließend ausgewertet. Die Berichte stehen
neben dem Checkpoint unter `.training.json`.

Der frühere Belief-Versuch auf `vibe-coding` wurde nicht übernommen. Seine
begrenzte Spielstärke hat keine nachgewiesene Ursache und sagt nichts über das
Ergebnis dieses ISS-Trainings aus.

Für größere Läufe kann `belief-import` direkt aus dem vollständigen Audit eine
neue Auswahl treffen. Der Audit-Hash des Archivs wird auch geprüft, wenn das
Archiv inzwischen an einem anderen Pfad liegt. Die Auswahl samt Gruppen und
Split bleibt unter `selection/` im neuen Korpus erhalten:

```bash
.venv/bin/python -m iss belief-import \
  --audit benchmark_runs/iss-run-1/audit \
  --archive /home/silvan/Downloads/iss-games-2007-05-2026.bz2 \
  --out data/iss-belief-300k --max-games 300000 --seed 7

python -m skat.agents.belief.train \
  --data data/iss-belief-300k \
  --out trained_models/iss-belief-300k.pt \
  --epochs 12 --d-model 64 --num-layers 2 \
  --device cpu --seed 7
```

Die Auswahl verwendet denselben Seed und dieselben zeitlichen Grenzen wie der
100k-Lauf. Dessen ausgewählte Spiele behalten ihren ursprünglichen Split.

### Siegerfilter: abgeschlossener 300k-Vergleich

Das reguläre Handmodell wurde auf gewonnenen **und** verlorenen, vollständig
geprüften ISS-Partien trainiert. Das Audit enthält 6.615.551 solche Spiele:
4.896.916 Siege und 1.718.635 Niederlagen des Alleinspielers.

Für einen einmaligen Kontrollversuch wurden aus derselben 300k-Auswahl zwei
weitere Modelle mit je 177.585 Trainingsspielen und identischer Validierung
und identischem Test trainiert: eines nur mit Siegen, eines mit zufällig
gewählten ganzen Partien beider Ausgänge. Die Trainingsdaten enthalten
532.755 beziehungsweise 532.253 Positionen (0,09 % Unterschied). Die
Zufallsauswahl enthält 131.497 Siege und 46.088 Niederlagen. Der
Siegerfilter ist **keine** Option des regulären Imports oder Trainings.

Nach zwölf Epochen ergaben die unveränderten 90.046 Testpositionen folgende
Vorhersagewerte; kleiner ist besser. Die Weltmetrik nutzt dieselben 1.000
Testpositionen und für alle Modelle Gewichtungsstärke 1,0:

| Training | Trainingsspiele | Kartenverlust | Ganze Verteilung |
| --- | ---: | ---: | ---: |
| Alle Spiele | 239.902 | 0,6928 | 6,508 |
| Zufällige ganze Spiele | 177.585 | 0,6979 | 6,535 |
| Nur Siege | 177.585 | 0,7002 | 6,571 |

Im gepaarten Spieltest wurden 1.000 identische Deals, alle drei
Kandidatensitze, konservative Gebote und zwei getrennte Gegnerfelder genutzt.
Je Gegnerfeld gab es 2.877 gespielte Kandidatenrunden und 123 Pässe. Die
**Parteisiegquoten** waren:

| Gegner | Alle 300k | Zufallsauswahl | Nur Siege | Nur Siege minus Zufallsauswahl (95-%-Intervall) |
| --- | ---: | ---: | ---: | ---: |
| Random | 69,17 % | 68,89 % | 68,16 % | −0,73 (−2,12 bis +0,70) Prozentpunkte |
| Feste Policy | 48,49 % | 48,91 % | 49,43 % | +0,52 (−0,66 bis +1,63) Prozentpunkte |

Die Intervalle bootstrappen vollständige Deals. Der kleine beobachtete
Siegquotenvorteil gegen die feste Policy ist nicht abgesichert; gegen Random
zeigt die Richtung nach unten. Bei gleicher Spielzahl sagt das Sieger-Modell
die Hände etwas schlechter voraus, besonders bei Nullspielen (Kartenverlust
0,7616 statt 0,7471). Aus diesem Versuch folgt kein Grund, verlorene
Partien aus dem regulären Training zu entfernen.

Die vollständigen Resultate liegen unter
`benchmark_runs/iss-belief-300k-{wins-eval,game-matched-eval,filter-vs-matched,wins-vs-all,matched-vs-all}.json`.
Die experimentellen Checkpoints bleiben unter
`trained_models/iss-belief-300k-{wins,game-matched}.pt` erhalten; ihre
SHA-256-Hashes sind
`847fdb6d55b4d7bc0f59dd5d798ea16284058e3a49519127edecfa3c482e2700`
und `e66c780960b5ea49494ab7a182e7dd74ac87794b0562e1644e7b813dc025060d`.

## Spielen und vergleichen

```bash
python -m skat play --agents mcts random random \
  --belief-model trained_models/iss-belief-100k.pt --workers 8

python -m skat evaluate --agents mcts \
  --policy trained_models/transformer-run-2.pt \
  --out benchmark_runs/iss-belief-uniform.json --deals 1000 --seed 42

python -m skat evaluate --agents mcts \
  --policy trained_models/transformer-run-2.pt \
  --belief-model trained_models/iss-belief-100k.pt \
  --out benchmark_runs/iss-belief-weighted.json --deals 1000 --seed 42

python -m scripts.compare_belief \
  benchmark_runs/iss-belief-uniform.json \
  benchmark_runs/iss-belief-weighted.json \
  --out benchmark_runs/iss-belief-comparison.json
```

`compare_belief` prüft identische Deals und vergleicht Siege sowie offizielle
Punkte pro ausgeteilter Runde mit einem Bootstrap über vollständige Deals.
Auch das feste README-Rating wird mit
`python -m scripts.rate_agents --out NEW_DIRECTORY --device cpu` neu gemessen.
Die gewählte Suche bleibt bei 600 Simulationen und 20 Welten, sodass nur der
Sampler wechselt. Die Vorhersage wird einmal pro Kartenentscheid berechnet;
SearchPool-Worker erhalten nur vollständige gezogene Welten.

## Gemessener erster Pilot

Der Lauf mit 100.000 ISS-Spielen erzeugte 239.610 Trainings-, 29.684
Validierungs- und 30.443 Testpositionen. Der beste Checkpoint nach zwölf
Epochen hat SHA-256
`aae1047030e4bbc0b397b8a5349fcb794ea32482f948251a6c33dbe8c28e931d`.
Die Validierung wählte Gewichtungsstärke 1,0. Auf 1.000 vorher nicht benutzten
Testpositionen sank die mittlere negative Log-Wahrscheinlichkeit der **ganzen
wirklichen Verteilung** von 7,863 bei Gleichverteilung auf 6,679. Der
Kartenverlust über den vollständigen zeitlichen Testsplit beträgt 0,713.

Die Spielstärke wurde auf denselben 1.000 Deals, allen drei Kandidatensitzen,
600 Simulationen und 20 Welten verglichen. Die Gegner waren entweder zwei
Random-Agenten oder zweimal `trained_models/transformer-run-2.pt`. Die Zahl in
Klammern ist das gepaarte 95-%-Bootstrapintervall über vollständige Deals:

| Gegner | Gleichverteilung: Punkte / Deal | Handmodell: Punkte / Deal | Differenz (95 % Intervall) |
| --- | ---: | ---: | ---: |
| Random | +9,49 | +9,21 | −0,27 (−0,86 bis +0,27) |
| Feste Policy | +3,61 | +3,45 | −0,16 (−0,90 bis +0,56) |

Die Handvorhersage ist in ISS-Positionen deutlich besser, aber diese
Spielvergleiche belegen keinen Gewinn durch den neuen Sampler. Gegen die feste
Policy stieg die Parteisiegquote von 47,24 % auf 48,07 %; auch deren gepaartes
Intervall schließt null ein. Der Sampler bleibt deshalb optional. Vollständige
Rundenergebnisse und Intervalle stehen in
`benchmark_runs/iss-belief-{uniform,weighted,comparison}.json`.

## Größerer ISS-Korpus

Der zweite Lauf wählte mit demselben Seed 300.000 geprüfte Spiele. Daraus
entstanden 719.008 Trainings-, 90.103 Validierungs- und 90.046 Testpositionen.
Die Architektur und die zwölf Epochen blieben gleich. Der beste Checkpoint nach
Epoche 12 hat SHA-256
`446e5e98be9afbd4fd487c48794b129de799a80c66d12732dfb5199eded8fd57`.
Seine Validierung wählte ebenfalls Gewichtungsstärke 1,0. Auf seinem zeitlichen
Testsplit beträgt der Kartenverlust 0,6928 und die negative Log-Wahrscheinlichkeit
der vollständigen Verteilung 6,508 statt 7,826 bei Gleichverteilung.

Für einen direkten Vergleich wurden **beide eingefrorenen Checkpoints** noch
einmal auf den alten, gemeinsamen Testpositionen ausgewertet. Die 1.000 Welten
für die Ganzverteilungsmetrik sind mit demselben Seed gezogen:

| Modell | Kartenverlust, kompletter alter Testsplit | Ganze Verteilung, 1.000 alte Testpositionen |
| --- | ---: | ---: |
| 100k | 0,7131 | 6,679 |
| 300k | 0,6920 | 6,501 |

Die vollständigen Werte und Checkpoint-Hashes stehen in
`benchmark_runs/iss-belief-300k-heldout.json`. Eine bessere Vorhersage belegt
für sich genommen noch keinen Gewinn an Spielstärke.

Die anschließende gepaarte Bewertung verwendete denselben Quellcode, dieselben
1.000 Deals, alle drei Kandidatensitze und dieselbe feste Referenz-Policy wie der
100k-Pilot. Bei 600 Simulationen und 20 Welten stieg die **Parteisiegquote**
gegen die feste Policy von 47,24 % auf 48,49 %. Die gepaarte Differenz beträgt
+1,25 Prozentpunkte; ihr 95-%-Bootstrapintervall liegt bei +0,03 bis +2,43
Prozentpunkten. Gegen Random stieg die Siegquote von 68,34 % auf 69,17 %;
das Intervall für die Differenz von +0,83 Prozentpunkten reicht von −0,80 bis
+2,47 Prozentpunkten.

Die zusätzlich gemessenen offiziellen Punkte pro ausgeteilter Kandidatenrunde
zeigen ein gemischtes Bild:

| Gegner | Gleichverteilung | 100k-Modell | 300k-Modell | 300k minus Gleichverteilung (95-%-Intervall) |
| --- | ---: | ---: | ---: | ---: |
| Random | +9,49 | +9,21 | +9,73 | +0,25 (−0,32 bis +0,80) |
| Feste Policy | +3,61 | +3,45 | +2,97 | −0,64 (−1,41 bis +0,08) |

Gegen Random erzielte das 300k-Modell mehr Punkte als das 100k-Modell
(gepaarte Differenz +0,52; 95-%-Intervall +0,13 bis +0,95). Gegen die feste
Policy war die Differenz −0,48 (−1,15 bis +0,19). Der Vorteil bei der
Parteisiegquote gegen die feste Policy ist damit das positive Spielergebnis;
für die Punktwertung und gegen Random ist ein Vorteil nicht nachgewiesen.
Gleichverteilung bleibt voreingestellt, das Handmodell ist per
`--belief-model` wählbar.
Im festen README-Rating (1.000 Deals, Kandidat auf Sitz 0 gegen zwei konservative
Random-Agenten) erreichte 300k +10,47, 100k +10,48 und Gleichverteilung +10,38
offizielle Punkte pro ausgeteilter Runde.
Vollständige Ergebnisse und Intervalle liegen in
`benchmark_runs/iss-belief-300k-{weighted,comparison}.json` und
`benchmark_runs/iss-belief-100k-vs-300k.json`.
