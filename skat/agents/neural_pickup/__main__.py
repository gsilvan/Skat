"""Import paired ISS rollouts and train the pickup decision transformer."""

import argparse
import json

from skat.config import TrainConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser("import")
    importer.add_argument("--audit", required=True)
    importer.add_argument("--archive")
    importer.add_argument("--policy", required=True)
    importer.add_argument("--declaration-model")
    importer.add_argument("--out", required=True)
    importer.add_argument("--max-games", type=int, default=2000)
    importer.add_argument("--seed", type=int, default=7)
    importer.add_argument("--workers", type=int, default=1)
    importer.add_argument("--device", default="cpu")
    trainer = sub.add_parser("train")
    trainer.add_argument("--data", required=True)
    trainer.add_argument("--out", required=True)
    trainer.add_argument("--seed", type=int, default=7)
    trainer.add_argument("--device", default="cpu")
    trainer.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "import":
        from iss.pickup import import_pickup

        result = import_pickup(
            args.audit,
            args.out,
            policy=args.policy,
            declaration_model=args.declaration_model,
            max_games=args.max_games,
            seed=args.seed,
            device=args.device,
            workers=args.workers,
            archive_override=args.archive,
        )
    else:
        from skat.agents.neural_pickup.train import train

        result = train(
            args.data,
            args.out,
            seed=args.seed,
            device=args.device,
            config=TrainConfig(
                epochs=args.epochs, batch_size=512, d_model=64, num_layers=2
            ),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
