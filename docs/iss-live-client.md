# ISS-Client in Python

Der Live-Client spricht das veröffentlichte, zeilenorientierte ISS-Protokoll
direkt über TCP. Er benötigt weder Java noch den offiziellen Client zur
Laufzeit. Das Protokoll und die Zugsyntax stammen aus dem offiziellen
[`clients.tgz`](https://skatgame.net/iss/clients.tgz), insbesondere
`src/client/ServerClient.java`, `src/client/ServiceClient.java`,
`src/common/Table.java`, `src/common/SimpleState.java` und
`doc/skat-game-format`.

Die eigene Hand und der öffentliche Spielverlauf werden in eine `PlayerView`
und exakt die 376 Features unserer Kartenpolicy übersetzt. Der Client verwaltet
Reizen, Skataufnahme, Drücken, Ansage und Kartenspiel; er nutzt dafür die
bereits modularen Agentenstrategien. Gegenspielerhände und unbekannter Skat
werden nie aus dem Serverprotokoll übernommen. Die Engine und der ISS-Client
brauchen nur Python und die Abhängigkeiten des gewählten Agenten.

Ein ISS-Konto und ein Tisch sind nötig. Das Passwort kommt aus der Umgebung,
nicht aus einem Kommandozeilenargument:

```bash
read -rsp 'ISS-Passwort: ' ISS_PASSWORD; echo
export ISS_PASSWORD

.venv/bin/python -m skat.integrations.iss_client \
  --id MEIN_ISS_NAME --table TISCH_ID \
  --agent policy --policy benchmark_runs/iss-run-1/policy.pt \
  --bidding neural --bidding-model benchmark_runs/bidding-run-1/bidding.pt \
  --bidding-threshold 0.55 \
  --declaration neural \
  --declaration-model benchmark_runs/declaration-run-3/declaration.pt
```

Der Hand-Klassifikator ist mit `--pickup neural --pickup-model
benchmark_runs/pickup-run-2/pickup.pt` zuschaltbar. Ohne diese Optionen nimmt
der Agent den Skat auf. `--agent mcts --simulations 600
--determinizations 20` (optional mit `--belief-model PFAD`) und
`--agent random` verwenden dieselben Reiz-, Ansage- und Pickup-Optionen.
Mit `--accept-invites` kann der Client statt
`--table` auf eine Einladung warten; mit beiden Optionen nimmt er nur
Einladungen für den angegebenen Tisch an. `--host` und `--port` überschreiben
die Standardadresse aus dem offiziellen Beispielclient.

Der Client setzt sich nach einem Spiel automatisch wieder bereit. Wenn er
während einer schon laufenden Partie an den Tisch kommt, wartet er auf das
nächste Spiel; ein Wiedereinstieg in eine begonnene Partie wird derzeit nicht
rekonstruiert. ISS-Sonderaktionen wie Aufgabe und „Show Cards“ werden als
Serverereignisse verarbeitet, der Bot sendet selbst reguläre Karten und keine
vorzeitigen Gewinnansprüche. Bei unbekannten oder widersprüchlichen
Spielereignissen bricht er ab, statt einen möglicherweise unerlaubten Zug zu
senden.

Die Zustandsübersetzung und die Nachrichtenhülle sind mit lokalen Tests
geprüft; ein echter ISS-Login benötigt eigene Zugangsdaten und wurde bei der
Implementierung nicht durchgeführt.
