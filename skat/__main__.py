import argparse

from skat.game import Tournament


def cli():
    parser = argparse.ArgumentParser(description="The Skat program.")
    parser.add_argument("--rounds", type=int, default=33)
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    rounds = args.rounds
    t = Tournament(rounds)
    t.start()
    print(f"{t.scores}")
