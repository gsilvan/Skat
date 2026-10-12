"""Import ISS original hands, train/calibrate a bidder, benchmark combinations."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_ = commands.add_parser("import")
    import_.add_argument("--audit", required=True)
    import_.add_argument("--out", required=True)
    import_.add_argument("--max-games", type=int, default=200000)
    import_.add_argument("--seed", type=int, default=7)
    train_ = commands.add_parser("train")
    train_.add_argument("--data", required=True)
    train_.add_argument("--out", required=True)
    train_.add_argument("--config")
    train_.add_argument("--seed", type=int, default=7)
    train_.add_argument("--device", default="cpu")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--bidding-model", required=True)
    benchmark.add_argument("--policy", required=True, help="Candidate card policy")
    benchmark.add_argument("--reference-policy", required=True)
    benchmark.add_argument(
        "--agent", choices=("policy", "mcts", "random"), default="policy"
    )
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--config")
    benchmark.add_argument("--workers", type=int)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--validation-deals", type=int, default=500)
    benchmark.add_argument("--deals", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            from iss.bidding import import_bidding

            report = import_bidding(
                args.audit, args.out, max_games=args.max_games, seed=args.seed
            )
            print(
                json.dumps(
                    {k: report[k] for k in ("rows", "wins", "provenance")}, indent=2
                )
            )
        elif args.command == "train":
            from skat.agents.neural_bidding.train import train
            from skat.config import load_config

            config = load_config(args.config)[1] if args.config else None
            report = train(
                args.data, args.out, config=config, seed=args.seed, device=args.device
            )
            print(
                json.dumps(
                    {k: v for k, v in report.items() if k != "history"}, indent=2
                )
            )
        else:
            from skat.agents.neural_bidding.benchmark import benchmark_bidding
            from skat.config import load_config

            report = benchmark_bidding(
                args.out,
                agent=args.agent,
                policy=args.policy,
                reference_policy=args.reference_policy,
                bidding_model=args.bidding_model,
                mcts=load_config(args.config)[0],
                workers=args.workers,
                device=args.device,
                validation_deals=args.validation_deals,
                deals=args.deals,
            )
            print(json.dumps(report, indent=2))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
