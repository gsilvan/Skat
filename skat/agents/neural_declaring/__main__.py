"""Opt-in ISS counterfactual import and declaration-model training."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_ = commands.add_parser("import")
    import_.add_argument("--audit", required=True)
    import_.add_argument(
        "--archive", help="Moved ISS archive; SHA-256 must match audit"
    )
    import_.add_argument("--policy", required=True)
    import_.add_argument("--out", required=True)
    import_.add_argument("--max-games", type=int, default=2000)
    import_.add_argument("--pairs-per-game", type=int, default=4)
    import_.add_argument(
        "--candidate-budget", type=int, help="Maximum simulated choices per deal"
    )
    import_.add_argument("--seed", type=int, default=7)
    import_.add_argument("--device", default="cpu")
    import_.add_argument("--workers", type=int, default=1)
    train_ = commands.add_parser("train")
    train_.add_argument("--data", required=True)
    train_.add_argument("--out", required=True)
    train_.add_argument("--config")
    train_.add_argument("--seed", type=int, default=7)
    train_.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            from iss.declaration import import_declarations

            result = import_declarations(
                args.audit,
                args.out,
                policy=args.policy,
                max_games=args.max_games,
                pairs_per_game=args.pairs_per_game,
                seed=args.seed,
                device=args.device,
                archive_override=args.archive,
                candidate_budget=args.candidate_budget,
                workers=args.workers,
            )
        else:
            from skat.agents.neural_declaring.train import train
            from skat.config import load_config

            result = train(
                args.data,
                args.out,
                config=load_config(args.config)[1] if args.config else None,
                seed=args.seed,
                device=args.device,
            )
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
