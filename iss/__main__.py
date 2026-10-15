"""Build rule-checked ISS demonstration corpora without loading the archive."""

import argparse
import json

from iss.pipeline import audit, import_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("audit", help="Validate a .sgf or .bz2 archive")
    check.add_argument("--input", required=True)
    check.add_argument("--out", required=True)
    convert = commands.add_parser(
        "import", help="Import a pilot from an audited archive"
    )
    convert.add_argument("--audit", required=True)
    convert.add_argument("--out", required=True)
    convert.add_argument("--max-games", type=int, default=100000)
    convert.add_argument("--seed", type=int, default=7)
    belief = commands.add_parser("belief-import", help="Import hidden-hand labels")
    source = belief.add_mutually_exclusive_group(required=True)
    source.add_argument("--selection", help="Existing audited ISS action corpus")
    source.add_argument("--audit", help="Audit used to select new games")
    belief.add_argument(
        "--archive", help="SGF archive matching the audit or selection hash"
    )
    belief.add_argument("--out", required=True)
    belief.add_argument("--max-games", type=int)
    belief.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            report = audit(args.input, args.out)
            print(
                json.dumps({k: v for k, v in report.items() if k != "actors"}, indent=2)
            )
        elif args.command == "belief-import":
            from iss.belief import import_belief, import_belief_from_audit

            if args.audit:
                report = import_belief_from_audit(
                    args.audit,
                    args.out,
                    archive=args.archive,
                    max_games=300000 if args.max_games is None else args.max_games,
                    seed=args.seed,
                )
            else:
                if args.archive is None or args.max_games is not None:
                    parser.error(
                        "--selection requires --archive and excludes --max-games"
                    )
                report = import_belief(
                    args.selection, args.archive, args.out, seed=args.seed
                )
            print(json.dumps({key: report[key] for key in ("rows", "games")}, indent=2))
        else:
            report = import_corpus(
                args.audit, args.out, max_games=args.max_games, seed=args.seed
            )
            print(json.dumps(report["provenance"], indent=2))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
