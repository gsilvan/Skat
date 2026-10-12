"""Original soloist hands from validated, ordinary ISS point games."""

from collections import Counter
from pathlib import Path

from iss import require_sgf_support
from iss.pipeline import connect, records, select_games
from iss.protocol import parse_record
from skat.agents.neural_bidding.features import bidding_state
from skat.agents.neural_bidding.samples import BiddingWriter
from skat.rules import evaluate_game


def import_bidding(audit_directory, output, *, max_games=200000, seed=7, progress=True):
    require_sgf_support()
    predicate = """ AND json_extract(metadata, '$.declaration.kind') IN ('suit', 'grand')
        AND json_extract(metadata, '$.declaration.hand')=0
        AND json_extract(metadata, '$.declaration.ouvert')=0
        AND json_extract(metadata, '$.declaration.schneider_announced')=0
        AND json_extract(metadata, '$.declaration.schwarz_announced')=0 """
    archive, provenance = select_games(
        audit_directory,
        output,
        max_games=max_games,
        seed=seed,
        predicate=predicate,
        validation_halves=True,
    )
    output = Path(output)
    writer = BiddingWriter(output)
    skipped = Counter()
    with connect(output / "index.sqlite") as db:
        targets = iter(
            db.execute("SELECT line, split, group_id FROM selection ORDER BY line")
        )
        target = next(targets, None)
        with records(archive, progress=progress) as lines:
            for index, line in lines:
                if target is None:
                    break
                if index != target[0]:
                    continue
                game = parse_record(line)
                # The prior full replay validated these recorded points/tricks.
                result = evaluate_game(
                    game.contract,
                    abs(game.matadors),
                    (game.points, 120 - game.points),
                    (game.tricks, 10 - game.tricks),
                    game.bid,
                    soloist=game.soloist,
                )
                if result.overbid:
                    skipped["overbid"] += 1
                    db.execute("DELETE FROM selection WHERE line=?", (index,))
                else:
                    state = bidding_state(game.original[game.soloist], game.soloist, 2)
                    writer.add(
                        state,
                        game.won,
                        target[1],
                        dict(group=target[2], line=index, **game.metadata),
                    )
                target = next(targets, None)
        if target is not None:
            raise ValueError("archive ended before selected record")
    provenance.update(
        skipped=dict(skipped),
        label="actual soloist win; other seats have no labels",
        eligibility="ordinary pickup suit/grand; no overbids",
        calibration="disjoint validation groups, sha256(seed:calibration:group) parity",
    )
    return writer.finish(provenance)
