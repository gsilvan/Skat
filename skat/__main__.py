import argparse

from skat.game import Tournament


def get_args():
    parser = argparse.ArgumentParser(description="The Skat program.")
    parser.add_argument(
        "--rounds", type=int, default=33, help="Play the game for n-rounds."
    )
    parser.add_argument("-p0", type=str, default="random")
    parser.add_argument("-p1", type=str, default="random")
    parser.add_argument("-p2", type=str, default="random")
    parser.add_argument("-v", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def get_player(arg: str):
    from skat.agents.command_line import CommandLineAgent
    from skat.agents.random import RandomAgent

    match arg:
        case "random":
            return RandomAgent()
        case "cli":
            return CommandLineAgent()
        case _:
            raise Exception("Agent not found.")


if __name__ == "__main__":
    args = get_args()
    rounds = args.rounds
    agent_args = [args.p0, args.p1, args.p2]
    agents = []
    for i, a in enumerate(agent_args):
        agents.append(get_player(a))
    t = Tournament(rounds=rounds, agents=agents, verbose=args.v)
    t.start()
    print(f"{t.scores}")
