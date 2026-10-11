# ISS demonstrations as a transformer training corpus

The initial implementation is now documented in [iss-training.md](iss-training.md).
This document retains the design rationale and later experiment ideas.

This is a proposed second data source, alongside the existing MCTS self-play
pipeline. It does not change the collection currently running. The downloaded
archive is `/tmp/iss-games-2007-05-2026.bz2` (about 580 MiB on disk).

## What the archive supplies

A bounded inspection of the first 2,000 records found SGF properties for game
ID (`ID`), session (`SE`), timestamp (`DT`), players (`P0/P1/P2`), the original
32-card deal, bids, declarations, discards and card actions (`MV`), and result
(`R`). This sample is from the beginning of the historical archive, not a
representative estimate of its full bot/human mix or modern playing strength.

The sample contains 1,430 logged soloist wins, 461 losses, 30 penalties and 79
passed-out games. The current parser accepts 1,837 records and rejects 163;
its acceptance is insufficient for replay validation. It interprets a sampled
`GHS` declaration as plain Grand Hand and loses the announcement. The parser
also currently throws away some leave/timeout records, and `iss.extractor`
explicitly keeps only soloist wins. Neither path is an appropriate policy
training importer as it stands.

A restricted replay prototype successfully reconstructed 1,650 of those
records and 48,288 card actions, checking turn order, legal cards, terminal
party outcome and soloist card points. Of the parser-accepted records, 186
contained special/non-card actions and one lost its announcement; these were
excluded from the prototype. All tested first plays were seat 0. This supports
the feasibility of action extraction, but is not validation of the full
historical protocol or full-archive quality. The prototype's bidding extraction
is deliberately restricted and must be replaced with the state machine.
Local audit results are in `benchmark_runs/iss-audit/results.json`.

The [ISS home page](https://skatgame.net/iss/) describes the public game archive
and names kermit, zoot and theCount among its strong programs. The
[client source distribution](https://skatgame.net/iss/software.html) is the
reference for the move protocol. Player identifiers in our sample provide no
verified general human/bot flag; `R0/R1/R2` must not be guessed to mean one.

## First policy: supervised action imitation

Train the existing card transformer on the card actually played by an expert.
For every non-forced card decision:

1. Replay the recorded game through our rule engine up to that action.
2. Build the actor's current egocentric observation with `Round.get_state(seat)`.
3. Obtain the legal-card mask from the game's follow-suit implementation.
4. Use the recorded legal card as the target action.

Use masked categorical cross-entropy, with the actual action as a one-hot
target. This is a separate demonstration source, not fabricated MCTS visit
counts. Shards must explicitly identify `iss-action-v1`, recording action,
mask, observation, game/session IDs, date and source metadata. Actor identity,
quality estimates and outcome stay in metadata, not network input. The served
network can retain the same observation/output layout and checkpoint family;
training metadata identifies ISS versus MCTS and any fine-tuning stages.

All three seats contribute examples. Include both winning and losing games:
a loss does not make every card bad, and win-only selection overrepresents
favourable hands and distorts defence. Final outcome is useful for audits and
role-specific evaluation; it does not establish the counterfactual value of
each card and should not be treated as its optimal action-value label.

The full deal in the SGF is used internally to reconstruct legality. Inputs
must reveal only the actor's hand, public history, known skat, bid, contract
and genuinely exposed cards at that instant. Never feed opponent hands,
future tricks or the final result into an observation. A replay importer must
preserve original pickup/discards, rather than invoking a heuristic agent to
choose them again. It must reconstruct seat order and bidding exactly; the
existing tournament ISS path samples deals/contracts but does not replay moves
or restore the original bidding history.

## Protocol and validation

Use a protocol state machine, with `pysgf` handling property escaping and SGF
syntax. Parse initial dealing, bids, pickup, discards, full declarations,
card actions, reveal/concession events and terminal server events separately.
Handle both historical dot and pipe separators explicitly.

There is a material ISS rule difference: the
[official GUI guide](https://skatgame.net/iss/skatgui-help.html) says that Show
Cards reveals the soloist's hand and continues play; it is not an all-tricks
claim. Give Up can award remaining tricks rather than necessarily producing
a loss for a soloist who already secured enough points. Do not map these
events blindly to our official-ISkO claim/concession actions. Represent ISS
reveal evidence in a replay adapter and validate terminal server semantics
separately, without changing the live engine's rules.

For the first importer, accept unambiguous, completed, card-only records whose
contracts can be fully reconstructed. Check card-pack uniqueness, ownership,
actor turn, legal play, trick winner, terminal result and compatible scoring.
Log rejections by precise reason. Add SC/RE and other supported events with
fixtures from real records after their protocol is confirmed. Valid prefixes
before an unsupported event can later supply action-imitation data, labelled
as partial and without inventing a final game outcome. Invalid actions or
unsupported public information must never be silently dropped while keeping
later observations.

Duplicate records, IDs and repeated deals must be identified before splitting.
Namespace IDs by source/server. Group repeated initial deals and complete
games/sessions together so near-identical positions cannot appear in both
training and validation. Use a frozen, later-date test set, keeping groups
that cross a time boundary out of conflicting partitions. A second test with
held-out actors can measure transfer beyond memorized player styles.

## Humans, bots and data volume

Build an explicit verified actor registry, retaining `unknown` where identity
is uncertain. Do not assume a bot is stronger than every human, that a name
implies a bot, or that modern strength applies to all its historical versions.
Bot versions/populations change over this archive's timespan.

Start with a pilot of approximately 100,000 validated games. Compare all valid
actions against a curated set of known strong actors; in mixed games, actor
selection can retain a bot's actions while using human actions as observed
history. Cap dominance by a single actor and check coverage by role, suit/
Grand/Null, Hand/ouvert, time period and actor type. Keep strong human examples
when evidence supports them. Sampling quality and diversity matter more than
simply counting downloaded games.

Read bzip2 and SGF incrementally, write bounded versioned shards and maintain
an on-disk grouping/index database. The current `ISSGames.read_sgf` stores the
whole archive in a Python list; `load_shards` concatenates the whole corpus.
Both need streaming/shard-aware alternatives before training on the full
archive. Avoid random per-row decompression of compressed NPZ files; use a
bounded shard cache or a memory-mapped training representation. Build splits
at game/session/deal boundaries before shuffling training batches.

## Experiments, in order

1. Run a full archive audit: dates, actors, contracts, clean replay counts,
   special actions, duplicates and skip reasons. Keep the raw archive intact.
2. Build a bounded ISS demonstration corpus and train a standalone transformer
   policy. Check held-out action loss/agreement, then live playing strength.
3. Evaluate the ISS policy, existing 100-epoch policy and MCTS on common deals,
   all seats and the same fixed reference checkpoints. Report each role and
   game type separately. This distinguishes population imitation from actual
   engine playing strength.
4. Test ISS pretraining followed by MCTS fine-tuning as a separate experiment.
   The current trainer always initializes a new network: this requires explicit
   initialization from a compatible policy checkpoint, fresh optimizer state,
   and provenance for both corpora. It must not be described as supported
   resume functionality today. A mixed demonstration/MCTS training experiment
   can subsequently check whether it preserves ISS-specific skills.
5. If demonstrations outperform our teacher in some situations, do not assume
   fine-tuning on that teacher improves them. A later alternative is to use
   the learned ISS policy to guide MCTS, then distil the improved search.

This provides a teacher from real strong play without running MCTS for every
training example. ISS action labels are sharper than shallow visit targets,
but single recorded actions can be noisy and other legal moves may be equally
good. Winning-games filtering, hidden-information leakage or approximate replay
can create misleading apparent improvements; rule-checked replay and fixed
live evaluations are therefore part of the method, not optional cleanup.

The initial goal remains card play. Learning bidding, discard choice and
declaration would require separate observations/actions and experiments, and
should not be bundled into the first ISS policy implementation.
