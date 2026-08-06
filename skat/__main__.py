import argparse
from typing import Optional

from skat.agents.bidding import BiddingStyle
from skat.game import Tournament


def get_args():
    parser = argparse.ArgumentParser(description="The Skat program.")
    parser.add_argument(
        "--rounds", type=int, default=33, help="Play the game for n-rounds."
    )
    parser.add_argument("-p0", type=str, default="random")
    parser.add_argument("-p1", type=str, default="random")
    parser.add_argument("-p2", type=str, default="random")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducible deals. Each round still gets its own hand.",
    )
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        metavar="{auto,cpu,cuda,...}",
        help="Torch device the DQN agents train on. "
        "auto picks cuda when it is usable, else cpu. Accepts anything "
        "torch understands, e.g. cuda:1.",
    )
    # These all default to None so that `DQN.__init__` stays the one place the
    # defaults live. --epsilon-decay used to default to 0.995 here while DQN
    # said 0.99995, and the CLI silently won by a factor of 100 - at 0.995 the
    # decay is per *action*, so exploration collapsed to its floor roughly 46
    # rounds after learning began.
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Transitions per gradient step (default: 256).",
    )
    parser.add_argument(
        "--learn-start",
        type=int,
        default=None,
        help="Transitions to collect before the first gradient step "
        "(default: 10000). Play is fully random until then, so the seed data "
        "covers a wide spread of positions. A DQN agent contributes 10 "
        "transitions per round.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Replay buffer capacity, FIFO (default: 100000).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default: 1.0). A round is a fixed ten tricks "
        "and the shaped rewards telescope to the final outcome, so "
        "undiscounted returns are both bounded and the quantity of interest.",
    )
    styles = "{" + ",".join(style.value for style in BiddingStyle) + "}"
    for seat in range(3):
        parser.add_argument(
            f"-b{seat}",
            type=BiddingStyle,
            choices=list(BiddingStyle),
            metavar=styles,
            default=BiddingStyle.CONSERVATIVE,
            help=f"Bidding style for player {seat} (default: conservative).",
        )
    parser.add_argument(
        "--simulations",
        type=int,
        default=600,
        help="MCTS playouts per card, split over the sampled determinizations.",
    )
    parser.add_argument(
        "--determinizations",
        type=int,
        default=None,
        help="Worlds an MCTS agent samples per card (default: 20). "
        "--simulations is split across them, so 600/20 is only 30 playouts "
        "per world - worth raising both when collecting training data.",
    )
    parser.add_argument(
        "--record-mcts",
        type=str,
        default=None,
        metavar="DIR",
        help="Write every searched MCTS decision to .npz shards in DIR, as "
        "training data for `python -m skat.agents.distill.train`. Forced moves "
        "are not recorded - there is nothing to learn from them.",
    )
    parser.add_argument(
        "--record-shard-size",
        type=int,
        default=None,
        help="Decisions per shard file (default: 50000).",
    )
    parser.add_argument(
        "--dqn-arch",
        type=str,
        choices=("mlp", "transformer"),
        default=None,
        help="Q-network for `-p{i} dqn` (default: mlp, the dense QNet). "
        "`transformer` uses CardTransformer, which as a policy beat the dense "
        "net on the same corpus with a fifth of the parameters. Checkpoints "
        "are not interchangeable between the two.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="./trained_models/policy.pt",
        help="Checkpoint for `-p{i} policy`. One file serves all three seats: "
        "the observation is egocentric, so the same net fits every chair.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=0.0,
        help="0 (default) plays the argmax and is deterministic; >0 samples.",
    )
    parser.add_argument(
        "--record-belief",
        type=str,
        default=None,
        metavar="DIR",
        help="Write every decision to DIR as hand-prediction training data: "
        "the observation, the void masks, and where the hidden cards really "
        "were. Works behind any agent - collect with `policy` seats, which are "
        "fast and play realistically.",
    )
    parser.add_argument(
        "--belief",
        type=str,
        default="./trained_models/belief.pt",
        help="Hand-prediction checkpoint for `-p{i} belief-mcts`, which is "
        "`mcts` with its determinizations drawn from that belief rather than "
        "uniformly.",
    )
    return parser.parse_args()


def get_player(
    arg: str,
    training: bool = False,
    path: str = None,
    epsilon: float = None,
    epsilon_decay: float = None,
    simulations: int = 600,
    seed=None,
    device: str = "auto",
    batch_size: Optional[int] = None,
    learn_start: Optional[int] = None,
    buffer_size: Optional[int] = None,
    gamma: Optional[float] = None,
    dqn_arch: Optional[str] = None,
    bidding: BiddingStyle = BiddingStyle.CONSERVATIVE,
    determinizations: Optional[int] = None,
    recorder=None,
    belief_recorder=None,
    policy_path: str = "./trained_models/policy.pt",
    policy_temperature: float = 0.0,
    belief_path: str = "./trained_models/belief.pt",
):
    from skat.agents.command_line import CommandLineAgent
    from skat.agents.dqn import DQNAgent
    from skat.agents.mcts import MCTSAgent, RecordingMCTSAgent
    from skat.agents.random import RandomAgent

    def build(agent_class, **kwargs):
        """Construct `agent_class`, recording beliefs if asked to.

        The mixin is applied to the class rather than there being a recording
        subclass per agent, because the labels do not depend on who is playing
        - the engine knows the hands whoever sits there. Which agent plays only
        decides which positions the corpus covers.
        """
        if belief_recorder is None:
            return agent_class(**kwargs)
        from skat.agents.belief_recorder import BeliefRecorder

        recording = type(
            f"BeliefRecording{agent_class.__name__}",
            (BeliefRecorder, agent_class),
            {},
        )
        return recording(recorder=belief_recorder, **kwargs)

    match arg:
        case "random":
            return build(RandomAgent, bidding=bidding)
        case "cli":
            return CommandLineAgent(bidding=bidding)
        case "mcts":
            options = {"simulations": simulations, "seed": seed, "bidding": bidding}
            if determinizations is not None:
                options["determinizations"] = determinizations
            if recorder is None:
                return build(MCTSAgent, **options)
            return RecordingMCTSAgent(recorder=recorder, **options)
        case "belief-mcts":
            from skat.agents.belief_mcts import BeliefMCTSAgent

            return build(
                BeliefMCTSAgent,
                path=belief_path,
                device=device,
                simulations=simulations,
                seed=seed,
                bidding=bidding,
                **(
                    {"determinizations": determinizations}
                    if determinizations is not None
                    else {}
                ),
            )
        case "policy":
            # Local like the rest, so a table of RandomAgents never imports torch.
            from skat.agents.policy import PolicyAgent

            return build(
                PolicyAgent,
                path=policy_path,
                device=device,
                temperature=policy_temperature,
                seed=seed,
                bidding=bidding,
            )
        case "dqn":
            return DQNAgent(
                train=training,
                path=path,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                device=device,
                batch_size=batch_size,
                learn_start=learn_start,
                buffer_size=buffer_size,
                gamma=gamma,
                arch=dqn_arch,
                bidding=bidding,
            )
        case _:
            raise Exception("Agent not found.")


def get_game(arg: Optional[int]):
    from skat.games.suit import SuitGame

    if arg is None:
        return None  # no forced game type - every soloist declares their own

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
    bidding_args = [args.b0, args.b1, args.b2]

    recorder = None
    if args.record_mcts:
        from skat.agents.distill import DEFAULT_SHARD_SIZE, ShardWriter
        from skat.models import PolicyNet

        recorder = ShardWriter(
            args.record_mcts,
            shard_size=args.record_shard_size or DEFAULT_SHARD_SIZE,
            input_size=PolicyNet.INPUT_SIZE,
            tag=str(args.seed),
            provenance={
                "simulations": args.simulations,
                "determinizations": args.determinizations or 20,
            },
        )

    belief_recorder = None
    if args.record_belief:
        from skat.agents.belief import BeliefShardWriter
        from skat.models.belief import BeliefNet

        belief_recorder = BeliefShardWriter(
            args.record_belief,
            input_size=BeliefNet.INPUT_SIZE,
            tag=str(args.seed),
        )

    agents = []
    for i, a in enumerate(agent_args):
        agents.append(
            get_player(
                a,
                training=train_args[i],
                path=path_args[i],
                epsilon=args.epsilon,
                epsilon_decay=args.epsilon_decay,
                simulations=args.simulations,
                # Three MCTS agents at one table must not share an RNG: with
                # one seed between them their determinizations march in step.
                seed=args.seed if args.seed is None else args.seed + i,
                device=args.device,
                batch_size=args.batch_size,
                learn_start=args.learn_start,
                buffer_size=args.buffer_size,
                gamma=args.gamma,
                dqn_arch=args.dqn_arch,
                bidding=bidding_args[i],
                determinizations=args.determinizations,
                recorder=recorder,
                belief_recorder=belief_recorder,
                policy_path=args.policy,
                policy_temperature=args.policy_temperature,
                belief_path=args.belief,
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
    try:
        t.start()
    finally:
        # An hour into a collection run, a Ctrl-C must not cost the tail.
        if recorder is not None:
            recorder.close()
        if belief_recorder is not None:
            belief_recorder.close()
    print(t.summary())
