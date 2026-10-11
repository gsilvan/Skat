"""Four independent workflows: play, collect, train, evaluate."""

import argparse
from dataclasses import asdict

from skat.config import load_config

AGENTS = ("random", "cli", "mcts", "policy")


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Official Skat engine and MCTS policy training"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("play", "Play a tournament"),
        ("collect", "Record MCTS self-play"),
        ("train", "Train a transformer policy"),
        ("evaluate", "Compare on common deals and seats"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", help="TOML file with [mcts] and [train] settings")
        p.add_argument("--seed", type=int, default=1)
        if name in ("play", "evaluate"):
            p.add_argument(
                "--agents",
                nargs="+" if name == "evaluate" else 3,
                choices=AGENTS if name == "play" else ("random", "mcts", "policy"),
                default=["random", "random", "random"]
                if name == "play"
                else ["random", "mcts", "policy"],
            )
            p.add_argument(
                "--policy",
                help="Transformer checkpoint (also used for the policy opponent panel)",
            )
            p.add_argument("--device", default="cpu")
        if name in ("play", "collect"):
            p.add_argument("--rounds", type=int, default=33 if name == "play" else 1000)
        if name == "play":
            p.add_argument("--verbose", action="store_true")
            p.add_argument(
                "--iss-file", help="Replay a pre-parsed .pkl or optional .sgf corpus"
            )
        if name == "collect":
            p.add_argument("--out", required=True, help="New corpus directory")
            p.add_argument(
                "--workers", type=int, help="CPU workers (default: all available CPUs)"
            )
        if name == "train":
            p.add_argument("--data", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--device", default="cpu")
        if name == "evaluate":
            p.add_argument("--deals", type=int, default=1000)
            p.add_argument(
                "--out", required=True, help="JSON results including paired outcomes"
            )
    return parser.parse_args(argv)


def get_player(name, *, seed=None, policy=None, device="cpu", mcts=None, recorder=None):
    if name == "random":
        from skat.agents.random import RandomAgent

        return RandomAgent(seed=seed)
    if name == "cli":
        from skat.agents.command_line import CommandLineAgent

        return CommandLineAgent(seed=seed)
    if name == "mcts":
        from skat.agents.mcts import MCTSAgent, RecordingMCTSAgent

        kwargs = asdict(mcts) if mcts is not None else {}
        return (
            RecordingMCTSAgent(recorder, seed=seed, **kwargs)
            if recorder is not None
            else MCTSAgent(seed=seed, **kwargs)
        )
    if name == "policy":
        from skat.agents.policy import PolicyAgent

        if not policy:
            raise ValueError("policy agent requires --policy PATH")
        return PolicyAgent(policy, device=device, seed=seed)
    raise ValueError(f"unknown agent {name}")


def main(argv=None):
    args = get_args(argv)
    try:
        mcts, training = load_config(args.config)
        if args.command == "play":
            from skat.game import Tournament

            agents = [
                get_player(
                    name,
                    seed=args.seed + i,
                    policy=args.policy,
                    device=args.device,
                    mcts=mcts,
                )
                for i, name in enumerate(args.agents)
            ]
            tournament = Tournament(
                args.rounds,
                agents,
                seed=args.seed,
                verbose=args.verbose,
                iss_file=args.iss_file,
                progress=True,
            )
            tournament.start()
            print(tournament.summary())
        elif args.command == "collect":
            from skat.collection import collect

            result = collect(
                args.out,
                rounds=args.rounds,
                seed=args.seed,
                mcts=mcts,
                workers=args.workers,
            )
            print(f"Recorded {result['decisions']} decisions in {args.out}")
        elif args.command == "train":
            from skat.agents.distill.train import train

            print(
                train(
                    args.data,
                    args.out,
                    config=training,
                    seed=args.seed,
                    device=args.device,
                )
            )
        else:
            from skat.evaluation import evaluate

            evaluate(
                args.agents,
                args.policy,
                args.out,
                deals=args.deals,
                seed=args.seed,
                device=args.device,
                mcts=mcts,
            )
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
