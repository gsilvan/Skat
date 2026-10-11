# Regelabgleich für die konsolidierte Engine

Maßgeblich ist die vom Skatgericht veröffentlichte
[Internationale Skatordnung, November 2022](https://dskv.de/app/uploads/sites/43/2022/11/ISkO-2022.pdf).
Diese Datei beschreibt die Implementierung in eigenen Worten; sie reproduziert
nicht den Regeltext. Die frühere, inzwischen veraltete Bestandsaufnahme bleibt
im Branch `vibe-coding` erhalten.

| Bereich | Umsetzung | Prüfungen |
|---|---|---|
| Karten, Rang, Augen (§1.2, §2.2) | 32 unveränderliche Karten, 120 Augen; Buben als Trumpf außer bei Null | Karten-, Hand- und Stichtests |
| Spielklassen und Ansagen (§2.1, §2.6, §3.5) | `GameDeclaration`, Hand/Skataufnahme, Ouvert und zulässige Ansagen | `tests/test_rules.py` |
| Spitzen (§2.3) | Höchste ununterbrochene Trumpffolge mit/ohne, einschließlich Skat | `tests/games/test_game_value.py` |
| Reizen (§3.3) | Sagen/Hören, zulässige Sprünge, Nullwerte, Vorhand-Übernahme, Eingepasst | `tests/test_bidding.py` |
| Spielwert (§2.4–§2.5) | Grundwert × Spitzen/Gewinnstufen, alle vier Nullwerte, doppelte Verlustwertung | `tests/test_rules.py` |
| Überreizen (§3.6) | Endgültiger Vergleich nach erreichten Stufen; aufgerundete Verlustwertung; Punktspielersatz bei nicht durchführbarem Null | `tests/test_rules.py`, `tests/test_tournament.py` |
| Durchführung (§4.1–§4.3) | Vorhand beginnt, Stichgewinner führt weiter, Bedienpflicht, legale Aktionen | Stichtests, `tests/test_rules.py` |
| Nullgewinn (§2.5.9) | Kein eigener Stich; Verlust auch durch einen Stich ohne Augen | `tests/test_rules.py` |
| Schneider/Schwarz (§2.5.5–§2.5.8) | Schneider über Augen, Schwarz über Stiche; verfehlte Ansagen verlieren | `tests/test_rules.py` |
| Aufgabe (§4.4.1–§4.4.3) | Soloist ab neun Handkarten ohne Zustimmung; danach Zustimmung eines Gegenspielers; gemeinsame Haftung der Gegner | `tests/test_rules.py` |
| Soloist deckt auf (§4.4.4) | Alle Gegner stimmen dem geforderten Gewinn zu oder Spiel geht mit offenen Karten weiter | `tests/test_rules.py` |
| Gegenspieler deckt auf (§4.4.5) | Exakte Prüfung aller legalen Restspiele; falsche Abkürzung gibt die Reststiche dem Soloisten | `tests/test_rules.py` |
| Bereits entschiedener Gewinn (§4.1.3) | Aufgabe hebt einen gesicherten Ausgang nicht auf | `tests/test_rules.py` |
| Wissensgrenzen (§2.6) | Abgetrennte Spieleransichten; nur Soloist mit Aufnahme kennt Skat; Ouvertkarten öffentlich | Regel- und Suchtests |

## Architektur

`skat/rules.py` enthält Deklarationen und die gemeinsame reine Wertung.
`RoundResult` trennt Augen, Stichzahlen, Spielwert, signed Soloist-Score und
Verlustgrund. Bei überreiztem Null ist zusätzlich das zur Abrechnung verwendete
Punktspiel hinterlegt. MCTS ruft denselben Evaluator auf wie die Engine.

Vorzeitige Ergebnisse bewerten nur vereinbarte oder bereits gesicherte Stufen;
unverteilte Reststiche werden nicht automatisch als Schwarz ausgelegt. Nullspiele
enden nach dem ersten Soloistenstich. Ein solches Ergebnis enthält tatsächlich
bis dahin gewonnene Augen/Stiche; deshalb summieren frühe Ergebnisse nicht
zwangsläufig auf 120 Augen beziehungsweise zehn Stiche.

## Grenzen des digitalen Modells

Die Engine verhindert illegale Eingaben durch atomare Validierung; sie simuliert
keine absichtlichen Regelverstöße mit anschließender Schiedsrichterentscheidung.
Geben erzeugt eine gleichverteilte 10/10/10/2-Aufteilung, ohne den physischen
3–Skat–4–3-Vorgang und Abheben zu animieren. Vierer-/Mehrpersonentischverwaltung,
physische Kartenfehler, verbale Einflussnahme, offenes Hinwerfen und Einsprüche
sind nicht als separate Eingaben modelliert. Informelle Varianten sind ausgeschlossen.

Die Restspielprüfung bei Gegenspieler-Abkürzung ist exakt und kann für große
Resthände teuer sein; sie läuft nur bei dieser ausdrücklichen Aktion, nie im
normalen Kartenspiel oder in automatischen MCTS-Entscheidungen.
