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
                help="Shared checkpoint for policy seats; evaluation candidate checkpoint",
            )
            p.add_argument("--device", default="cpu")
            p.add_argument("--bidding-model", help="Independent ISS hand classifier")
            p.add_argument("--bidding-threshold", type=float, default=2 / 3)
            p.add_argument(
                "--declaration-model", help="Pickup discard/declaration checkpoint"
            )
            p.add_argument("--pickup-model", help="Pre-skat Hand/pickup checkpoint")
            p.add_argument("--pickup-margin", type=float, default=0.0)
        if name in ("play", "collect"):
            p.add_argument("--rounds", type=int, default=33 if name == "play" else 1000)
        if name == "play":
            for seat in range(3):
                p.add_argument(
                    f"--policy{seat}",
                    help=f"Checkpoint for policy agent at seat {seat} (overrides --policy)",
                )
            for seat in range(3):
                p.add_argument(
                    f"--bidding{seat}",
                    choices=("conservative", "neural"),
                    default="conservative",
                )
                p.add_argument(
                    f"--declaration{seat}",
                    choices=("heuristic", "neural"),
                    default="heuristic",
                )
                p.add_argument(
                    f"--pickup{seat}", choices=("always", "neural"), default="always"
                )
            p.add_argument("--verbose", action="store_true")
            p.add_argument(
                "--iss-file", help="Replay a pre-parsed .pkl or optional .sgf corpus"
            )
        if name in ("play", "collect", "evaluate"):
            p.add_argument(
                "--workers", type=int, help="CPU workers (default: all available CPUs)"
            )
            p.add_argument(
                "--belief-model",
                help="Optional ISS hand-prediction checkpoint for MCTS",
            )
        if name == "collect":
            p.add_argument("--out", required=True, help="New corpus directory")
            p.add_argument("--device", default="cpu")
        if name == "train":
            p.add_argument("--data", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--device", default="cpu")
        if name == "evaluate":
            p.add_argument(
                "--bidding", choices=("conservative", "neural"), default="conservative"
            )
            p.add_argument(
                "--declaration", choices=("heuristic", "neural"), default="heuristic"
            )
            p.add_argument("--pickup", choices=("always", "neural"), default="always")
            p.add_argument(
                "--reference-policy",
                "--opponent-policy",
                dest="opponent_policy",
                help="Fixed reference model for the other two policy seats (default: --policy)",
            )
            p.add_argument("--deals", type=int, default=1000)
            p.add_argument(
                "--out", required=True, help="JSON results including paired outcomes"
            )
    return parser.parse_args(argv)


def get_player(
    name,
    *,
    seed=None,
    policy=None,
    device="cpu",
    mcts=None,
    recorder=None,
    search_pool=None,
    bidding="conservative",
    bidder=None,
    belief_model=None,
    declaration="heuristic",
    declaration_model=None,
    pickup="always",
    pickup_model=None,
    pickup_margin=0.0,
):
    if bidding == "neural" and bidder is None:
        raise ValueError("neural bidding requires --bidding-model")
    if name == "cli" and bidding != "conservative":
        raise ValueError("human bidding is controlled by stdin")
    if name == "cli" and declaration != "heuristic":
        raise ValueError("human declaration is controlled by stdin")
    if name == "cli" and pickup != "always":
        raise ValueError("human pickup is controlled by stdin")
    if declaration == "neural" and declaration_model is None:
        raise ValueError("neural declaration requires --declaration-model")
    if pickup == "neural" and pickup_model is None:
        raise ValueError("neural pickup requires --pickup-model")

    def configured(agent):
        agent.configure_bidding(bidding, bidder)
        agent.configure_declaration(declaration, declaration_model)
        return agent.configure_pickup(pickup, pickup_model, pickup_margin)

    if name == "random":
        from skat.agents.random import RandomAgent

        return configured(RandomAgent(seed=seed))
    if name == "cli":
        from skat.agents.command_line import CommandLineAgent

        return configured(CommandLineAgent(seed=seed))
    if name == "mcts":
        from skat.agents.mcts import MCTSAgent, RecordingMCTSAgent

        kwargs = asdict(mcts) if mcts is not None else {}
        if belief_model:
            from skat.agents.belief.inference import BeliefSampler

            kwargs["sampler"] = BeliefSampler(belief_model, device=device)
        return configured(
            RecordingMCTSAgent(recorder, seed=seed, search_pool=search_pool, **kwargs)
            if recorder is not None
            else MCTSAgent(seed=seed, search_pool=search_pool, **kwargs)
        )
    if name == "policy":
        from skat.agents.policy import PolicyAgent

        if not policy:
            raise ValueError("policy agent requires a checkpoint")
        return configured(PolicyAgent(policy, device=device, seed=seed))
    raise ValueError(f"unknown agent {name}")


def main(argv=None):
    args = get_args(argv)
    try:
        mcts, training = load_config(args.config)
        if args.command == "play":
            from skat.agents.search.parallel import SearchPool
            from skat.game import Tournament

            bidder = None
            if any(getattr(args, f"bidding{i}") == "neural" for i in range(3)):
                if not args.bidding_model:
                    raise ValueError("neural bidding requires --bidding-model")
                from skat.agents.neural_bidding.inference import NeuralBidder

                bidder = NeuralBidder(
                    args.bidding_model,
                    device=args.device,
                    threshold=args.bidding_threshold,
                )
            declarer = None
            if any(getattr(args, f"declaration{i}") == "neural" for i in range(3)):
                if not args.declaration_model:
                    raise ValueError("neural declaration requires --declaration-model")
                from skat.agents.neural_declaring.inference import DeclarationScorer

                declarer = DeclarationScorer(args.declaration_model, device=args.device)
            pickup_scorer = None
            if any(getattr(args, f"pickup{i}") == "neural" for i in range(3)):
                if not args.pickup_model:
                    raise ValueError("neural pickup requires --pickup-model")
                from skat.agents.neural_pickup.inference import PickupScorer

                pickup_scorer = PickupScorer(args.pickup_model, device=args.device)
            with SearchPool(
                args.workers, capacity=mcts.determinizations
            ) as search_pool:
                agents = [
                    get_player(
                        name,
                        seed=args.seed + i,
                        policy=getattr(args, f"policy{i}") or args.policy,
                        device=args.device,
                        mcts=mcts,
                        search_pool=search_pool,
                        bidding=getattr(args, f"bidding{i}"),
                        bidder=bidder,
                        belief_model=args.belief_model,
                        declaration=getattr(args, f"declaration{i}"),
                        declaration_model=declarer,
                        pickup=getattr(args, f"pickup{i}"),
                        pickup_model=pickup_scorer,
                        pickup_margin=args.pickup_margin,
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
                belief_model=args.belief_model,
                device=args.device,
            )
            print(f"Recorded {result['decisions']} decisions in {args.out}")
        elif args.command == "train":
            from skat.agents.distill.train import train

            report = train(
                args.data,
                args.out,
                config=training,
                seed=args.seed,
                device=args.device,
            )
            print({key: value for key, value in report.items() if key != "history"})
        else:
            from skat.agents.search.parallel import SearchPool
            from skat.evaluation import evaluate

            with SearchPool(
                args.workers, capacity=mcts.determinizations
            ) as search_pool:
                evaluate(
                    args.agents,
                    args.policy,
                    args.out,
                    deals=args.deals,
                    seed=args.seed,
                    device=args.device,
                    mcts=mcts,
                    search_pool=search_pool,
                    opponent_policy=args.opponent_policy,
                    bidding=args.bidding,
                    bidding_model=args.bidding_model,
                    bidding_threshold=args.bidding_threshold,
                    belief_model=args.belief_model,
                    declaration=args.declaration,
                    declaration_model=args.declaration_model,
                    pickup=args.pickup,
                    pickup_model=args.pickup_model,
                    pickup_margin=args.pickup_margin,
                )
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
