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
    parser.add_argument("--seed", default=None)
    parser.add_argument("--suit-game-only", action=argparse.BooleanOptionalAction)
    parser.add_argument("--hold-position", action=argparse.BooleanOptionalAction)
    parser.add_argument("--declare-game", type=int)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("-t0", action=argparse.BooleanOptionalAction)
    parser.add_argument("-t1", action=argparse.BooleanOptionalAction)
    parser.add_argument("-t2", action=argparse.BooleanOptionalAction)
    parser.add_argument("-m0", type=str, default="./trained_models/checkpoint_p0.pt")
    parser.add_argument("-m1", type=str, default="./trained_models/checkpoint_p1.pt")
    parser.add_argument("-m2", type=str, default="./trained_models/checkpoint_p2.pt")
    parser.add_argument("-v", action=argparse.BooleanOptionalAction)
    parser.add_argument("--iss-file", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.98)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    return parser.parse_args()


def get_player(
    arg: str,
    training: bool = False,
    path: str = None,
    epsilon: float = None,
    epsilon_decay: float = None,
):
    from skat.agents.command_line import CommandLineAgent
    from skat.agents.dqn import DQNAgent
    from skat.agents.random import RandomAgent

    match arg:
        case "random":
            return RandomAgent()
        case "cli":
            return CommandLineAgent()
        case "dqn":
            return DQNAgent(
                train=training, path=path, epsilon=epsilon, epsilon_decay=epsilon_decay
            )
        case _:
            raise Exception("Agent not found.")


def get_game(arg: int):
    from skat.games.suit import SuitGame

    match arg:
        case 0 | 1 | 2 | 3:
            return SuitGame(arg)
        case _:
            print("did not get this game declaration use 0-3")
            return None


if __name__ == "__main__":
    args = get_args()
    rounds = args.rounds
    agent_args = [args.p0, args.p1, args.p2]
    train_args = [args.t0, args.t1, args.t2]
    path_args = [args.m0, args.m1, args.m2]
    agents = []
    for i, a in enumerate(agent_args):
        agents.append(
            get_player(
                a,
                training=train_args[i],
                path=path_args[i],
                epsilon=args.epsilon,
                epsilon_decay=args.epsilon_decay,
            )
        )

    t = Tournament(
        rounds=rounds,
        agents=agents,
        verbose=args.v,
        seed=args.seed,
        hold_position=args.hold_position,
        declare_game=get_game(args.declare_game),
        iss_file=args.iss_file,
    )
    t.start()
    print(f"{t.scores}")
