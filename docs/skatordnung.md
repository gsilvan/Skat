# Skatordnung — rules reference & compliance check

This document is a **paraphrased summary**, in our own words, of the parts of the
*Internationale Skatordnung* (ISkO, Stand November 2018 — the official rulebook
jointly maintained by the Deutscher Skatverband e.V. and ISPA-World) that are
relevant to this engine. It is not a reprint of the original text — the ISkO itself
is marked "Nachdruck, auch auszugsweise, nur mit Genehmigung beider Verbände
gestattet" (reproduction, even excerpts, requires permission from both
associations). Section numbers (`§x.y.z`) are cited throughout so each claim can be
checked against the original; they were verified against a clean, line-by-line
transcription of the official PDF (see `internationale-skatordnung.md` at the repo
root — gitignored, kept local-only for the same copyright reason).

The second half of this document (["Abgleich mit der Implementierung"](#abgleich-mit-der-implementierung))
cross-references each rule area against the current code in this repository.

## 1. Karte & Werte (§1.2)

32 Karten in 4 Farben zu je 8 Karten. Farbrang: **Kreuz > Pik > Herz > Karo** (§1.2.1).
Kartenwerte (§1.2.2): Ass 11, Zehn 10, König 4, Dame 3, Bube 2, Neun/Acht/Sieben 0.
Die Skatkarte zählt insgesamt 120 Augen.

## 2. Spielmöglichkeiten & Spielklassen (§2.1, §2.6)

Zwei Spielklassen: Spiele **mit Skataufnahme** (Alleinspieler nimmt den Skat auf, drückt
zwei Karten) und **Handspiele** (Skat bleibt unbesehen liegen, nur die 10 Handkarten
zählen für die Ansage) — §2.6.1–§2.6.4. Innerhalb jeder Klasse gibt es drei
Spielgattungen: Farbspiele (Karo/Herz/Pik/Kreuz), Grandspiele, Nullspiele (§2.1.1–§2.1.2).
Farb- und Grandspiele haben zusätzlich eine offene ("ouvert") Variante; Nullspiele
existieren in vier Ausprägungen: Null, Null Hand, Null ouvert, Null Hand ouvert.

## 3. Kartenrang je Spielgattung (§2.2)

- **Farbspiel/Grand** (§2.2.1–§2.2.3): Die vier Buben sind die höchsten Trümpfe, in
  Farbreihenfolge Kreuz > Pik > Herz > Karo; danach (bei Farbspielen) die restlichen
  sieben Karten der Trumpffarbe nach Augenwert.
- **Nullspiel** (§2.2.4): Buben zählen als normale Farbkarte, keine Trümpfe.
  Rangfolge je Farbe: Ass, König, Dame, Bube, Zehn, Neun, Acht, Sieben.

## 4. Spitzen (§2.3)

Ununterbrochene Folge von Trümpfen ab Kreuz-Bube (§2.3.1). Hat der Alleinspieler
(inkl. der beiden Skatkarten) den Kreuz-Buben, zählen die vorhandenen Spitzen ("mit
X"), andernfalls die fehlenden ("ohne X") — §2.3.2. Maximal 11 Spitzen bei
Farbspielen (4 Buben + 7 Trumpfkarten), maximal 4 bei Grandspielen (nur die Buben) —
§2.3.3.

## 5. Reizen (§3.3)

Mindestreizgebot: **18**, Höchstreizgebot 264 (§3.3.1). Der Ablauf folgt dem
"Sagen/Hören"-Prinzip, nicht einer symmetrischen Gebotsrunde:

1. Vorhand fordert Mittelhand zum Reizen auf. **Mittelhand sagt** (nennt Zahlen,
   aufsteigend); **Vorhand hört** — sie hält (akzeptiert exakt diesen Wert) oder
   passt (§3.3.2).
2. Wer aus dieser ersten Runde als "nicht gepasst" hervorgeht (das kann Mittelhand
   *oder* Vorhand sein, je nachdem wer zuerst nicht mehr mithalten will/kann) reizt in
   der zweiten Runde gegen Hinterhand weiter, wieder nach Sagen/Hören (§3.3.3–§3.3.5).
3. Passen und Halten sind unwiderruflich (§3.3.8).
4. Bietet niemand etwas, darf Vorhand den Skat ohne Erklärung aufnehmen oder ein
   Handspiel ansagen — sie ist dann zur Spieldurchführung verpflichtet (§3.3.6).
5. Zeigt niemand Interesse, wird "eingepasst" — das ist ein **gültiges**
   Rundenergebnis (§3.2.19, §3.3.7), keine sofortige Neuverteilung. Der nächste
   Kartengeber gibt die Karten für das *nächste* Spiel.

## 6. Spielansage (§3.5)

Die Ansage muss das vollständige Spiel benennen (Farbe/Grand/Null plus Hand/ouvert,
falls beansprucht) und ist unwiderruflich (§3.5.1). Kann der Alleinspieler sein
ursprünglich beabsichtigtes Spiel nicht mehr zum gereizten Wert durchführen, darf er
stattdessen ein günstigeres, aber gültiges Spiel ansagen und dieses ggf. sofort
aufgeben (§3.5.6).

## 7. Spielbewertung (§2.4, §2.5, §3.6)

**Grundwerte** (§2.4.1): Karo 9, Herz 10, Pik 11, Kreuz 12, Grand 24.
**Nullspiel-Festwerte** (§2.4.2, unabhängig von Spitzen/Gewinnstufe): Null 23, Null
Hand 35, Null ouvert 46, Null ouvert Hand 59.

**Gewinnstufen** (§2.5.1) — für Spiele mit Skataufnahme nur 3 Stufen (einfach,
Schneider, Schwarz); für Handspiele 7 Stufen, wobei "einfach gewonnen" bereits als
Fall "2" zählt (weil "Hand" selbst ein Fall ist), bis hinauf zu Schneider angesagt,
Schwarz angesagt und Offen.

**Spielwert-Formel** (§2.5.2–§2.5.3, nur für Farb-/Grandspiele — Nullspiele haben
Festwerte): `Spielwert = (Spitzen + Gewinnstufen) × Grundwert`. Verlorene Spiele
werden mit dem doppelten Wert verbucht (§2.5.11).

**Überreizt** (§3.6.1, §3.6.3): Erreicht der tatsächliche Spielwert nicht den
gereizten Wert, wird so lange das Vielfache des Grundwerts genommen, bis der
Reizwert erreicht/überschritten ist; das Spiel gilt als verloren und wird als
"überreizt" eingetragen.

## 8. Spieldurchführung (§4)

Vorhand spielt zum ersten Stich aus, danach jeweils der Stichgewinner (§4.1.1).
Farbzwang (§4.2.1–§4.2.2): wer bedienen kann, muss die angespielte Farbe zugeben
(oder Trumpf, falls Trumpf gefordert und keine passende Farbe vorhanden). Ein Stich
gehört, wer die höchste Karte der Anspielfarbe bzw. den höchsten Trumpf im Stich
gespielt hat (§4.3.2).

**"Das 61. Auge gewinnt"** (Wissenswertes, "Das 61. Auge gewinnt"; siehe auch
§2.5.4): Sobald der Alleinspieler 61 Augen erreicht hat, ist das Spiel bei
regelgerechter Durchführung endgültig gewonnen — ein späterer Regelverstoß beim
Versuch, Schneider/Schwarz zu erreichen, kann diesen Grundgewinn nicht mehr
rückgängig machen, sondern höchstens die höhere Gewinnstufe verhindern.

## 9. Abgekürzte Spiele / Aufgabe (§4.4)

Der Alleinspieler darf sein Spiel eigenmächtig aufgeben, solange er noch mindestens
9 Handkarten hat (§4.4.1); bei weniger Handkarten braucht die Aufgabe die Zustimmung
mindestens eines Gegenspielers (§4.4.2).

---

## Abgleich mit der Implementierung

Legende: ✅ konform · ⚠️ teilweise/lückenhaft · ❌ nicht implementiert · 🐛 fehlerhaft

*(Re-verifiziert in dieser Session gegen den aktuellen Code-Stand; einige
Zeilennummern und Detaileinschätzungen wurden gegenüber der vorherigen Fassung
korrigiert.)*

### ❌ Keine offizielle Spielwert-Berechnung vorhanden (§2.4–§2.5)

`skat/games/suit.py:49-52`, `skat/games/grand.py:35-37`, `skat/games/null.py:31-33` —
die `value`-Property jeder Game-Klasse liefert nur den nackten Grundwert (9/10/11/12/24
bzw. pauschal 23 für Null). Nirgendwo in `skat/game.py` wird `Fälle = Spitzen +
Gewinnstufen` gebildet oder mit dem Grundwert multipliziert; es gibt keine
Spitzen-Zählung (§2.3) und keine Fälle/Gewinnstufen-Verwaltung. Die Engine kann Sieg/
Niederlage und Schneider/Schwarz feststellen (`soloist_won`, `soloist_won_schneider`,
`soloist_won_schwarz` etc., `skat/game.py:236-264`), aber nie einen tatsächlichen
ISkO-Spielwert für die Spielliste ausgeben. Das ist die größte Lücke gegenüber der
Skatordnung.

### ✅ Nullspiel-Varianten (behoben)

`skat/games/null.py`: `Null` nimmt jetzt `hand`/`ouvert`-Flags entgegen; `value`
schlägt in `Null.VALUES` nach und liefert 23/35/46/59 korrekt je Variante (§2.4.2).
(War zuvor 🐛 — immer `23`, kein `hand`/`ouvert`-Zustand.)

### ✅ Reizverfahren implementiert jetzt Sagen/Hören (behoben)

`skat/game.py:_duel()`/`bidding()`: das Reizen läuft jetzt als echtes
Sagen/Hören-Duell (§3.3.2–§3.3.6) — Mittelhand sagt aufsteigende Werte gegen
Vorhand, die je exakt hält oder passt; wer aus Stufe 1 hervorgeht, trifft in
Stufe 2 auf Hinterhand (die immer sagt). Bietet niemand etwas, darf Vorhand den
Skat/ein Handspiel unaufgefordert claimen (§3.3.6); lehnt sie ab, bleibt
`solo_player_id` auf dem Sentinel `-42` — kein `IndexError` mehr. Der `offer`-
Parameter von `Agent.bid` (`agents/__init__.py`, `random.py`, `command_line.py`)
ist jetzt tatsächlich verdrahtet: `offer=None` → Rolle "Sagen", `offer=<Wert>` →
Rolle "Hören". `Tournament.start()` nutzt außerhalb von ISS-Replays jetzt auch
wirklich `bidding()`, statt den Alleinspieler fest auf den Kartengeber zu setzen.
Regressionstests: `tests/test_bidding.py`.

### ✅ "Eingepasst" ist jetzt ein gültiges Rundenergebnis (behoben)

`skat/game.py:Round.start()`: Bleibt `solo_player_id` nach `bidding()` auf dem
Sentinel `-42` (niemand hat geboten, §3.2.19/§3.3.7), setzt `start()` die Phase auf
das neue `GamePhase.EINGEPASST` und gibt `None` zurück, statt in
`self.player[self.solo_player_id]` mit `IndexError` abzustürzen. `Tournament.start()`
behandelt `None` explizit (kein Punktestand, Kartengeber rotiert trotzdem weiter).

### ❌ Überreizt-Regel nicht implementiert (§3.6)

`self.highest_bid` (`skat/game.py:113`, gesetzt in `bidding()` Zeilen 415-437) wird
danach nirgendwo mehr gelesen — kein Vergleich mit dem tatsächlichen Spielwert, kein
"überreizt"-Verlustpfad.

### ❌ Aufgabe / Spielabkürzung nicht implementiert (§4.4)

Repo-weite Suche bestätigt: es gibt keine `concede`/`aufgabe`/`give_up`-Methode oder
Ähnliches irgendwo im Code. Weder die eigenmächtige Aufgabe ab ≥9 Handkarten
(§4.4.1) noch die zustimmungspflichtige Aufgabe darunter (§4.4.2) existieren in
irgendeiner Form.

### ✅ Korrekt umgesetzte Bereiche

- Kartenwerte (§1.2.2) — exakte Übereinstimmung (`skat/card.py:16`).
- Farb-Grundwerte `MULTIPLIERS = (9, 10, 11, 12)` für ♦♥♠♣ (`skat/games/suit.py:10`) —
  exakt §2.4.1; Grand = 24 (`skat/games/grand.py:37`) — exakt.
- Nullspiel-Kartenrang 7-8-9-10-B-D-K-A (`skat/games/null.py:12-14`, über
  `HandOrder(ranks="789XJQKA")`, sowie unabhängig nochmal in
  `NullTrick.better_than`, `null.py:66-72`) — exakt §2.2.4.
- Buben-Trumpfrang Kreuz > Pik > Herz > Karo und Jack-vor-Nicht-Jack-Stich-Logik —
  exakt §2.2.1–§2.2.2/§2.3.1 (`skat/games/suit.py:109-145`,
  `skat/games/grand.py:64-83`), inklusive bestehender Regressionstests
  (`test_bug_8/9/10`, `test_not_a_bug_11` in `tests/games/test_suit_game.py`).
- Mindestreizgebot 18 als unterste Stufe von `VALID_BIDS`
  (`skat/agents/__init__.py:13-15`) — exakt §3.3.1.
- Skat-Punkte werden auch im Handspiel dem Alleinspieler gutgeschrieben
  (`skat/game.py:480-482`, unbedingt, unabhängig von `hand_game`) — das ist
  **korrekt** und kein Bug (Korrektur gegenüber der vorherigen Fassung dieses
  Dokuments): §2.6.1 stellt klar, dass der Skat "in allen Fällen" dem
  Alleinspieler zusteht, also auch im Handspiel, wo er ungesehen bleibt.

### Außerhalb des aktuellen Funktionsumfangs

Offene Spiele (Grand ouvert, offene Farbhandspiele) existieren im Code überhaupt
nicht — `CommandLineAgent` bietet ausschließlich das einfache Null an.
Neuverteilungs-Regeln bei Fehlern beim Geben (§3.2.8–§3.2.13) sind nicht relevant,
da die Engine stets einen frisch gemischten, korrekten 32-Karten-Stapel
programmatisch erzeugt. Beides wird hier als bekannt fehlender Funktionsumfang
vermerkt, nicht als Bug.
