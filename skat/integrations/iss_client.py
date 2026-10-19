"""Pure Python ISS network client using the published line protocol.

Reference: https://skatgame.net/iss/clients.tgz (ServerClient.java,
ServiceClient.java, Table.java, doc/skat-game-format).
"""

import argparse
import logging
import os
import select
import socket

from skat.integrations.iss import choose
from skat.integrations.iss_state import LiveGame

LOG = logging.getLogger(__name__)


class ISSClient:
    def __init__(self, agent, *, table=None, table_password=".", accept_invites=False):
        self.agent = agent
        self.table = table
        self.table_password = table_password
        self.accept_invites = accept_invites
        self.player_name = None
        self.game = None
        self.re_sent = False
        self.send = None

    def handle(self, line):
        """Apply one server line, writing only through the injected send method."""
        parts = line.strip().split(" ")
        if not parts or not parts[0]:
            return
        if parts[0] == "error":
            raise RuntimeError("ISS server error: " + " ".join(parts[1:]))
        if parts[0] == "invite" and len(parts) >= 4 and self.accept_invites:
            if self.table is None or parts[2] == self.table:
                self.table = parts[2]
                self.send(f"join {parts[2]} {parts[3]}")
            return
        if parts[0] == "create" and len(parts) >= 4:
            if parts[1] == self.table and parts[2] != ".":
                self.player_name = parts[2]
                self.send(f"table {self.table} {self.player_name} ready")
            return
        if parts[0] == "destroy" and len(parts) >= 3:
            if parts[1] == self.table and parts[2] == self.player_name:
                self.game = None
                self.player_name = None
            return
        if parts[0] != "table" or len(parts) < 4 or parts[1] != self.table:
            return
        if parts[2] != self.player_name:
            return
        event = parts[3]
        if event == "error":
            raise RuntimeError("ISS table error: " + " ".join(parts[4:]))
        if event == "start":
            if len(parts) < 11:
                raise ValueError("short ISS start message")
            players = (parts[5], parts[7], parts[9])
            if self.player_name not in players:
                self.game = None  # Dealer at a four-player table.
                return
            if len(parts) > 11:
                LOG.warning("joined during a game; waiting for the next deal")
                self.game = None
                return
            self.game = LiveGame(players.index(self.player_name))
            self.agent.new_round_event()
            self.re_sent = False
            return
        if event == "end":
            self.game = None
            self.re_sent = False
            self.send(f"table {self.table} {self.player_name} ready")
            return
        if event != "play" or self.game is None:
            return
        if len(parts) < 9:
            raise ValueError("short ISS play message")
        source = 4 if parts[4] == "w" else int(parts[4])
        action = parts[5]
        self.game.apply(source, action)
        if action == "RE":
            if (
                not self.re_sent
                and self.game.seat != self.game.solo
                and source != self.game.seat
                and source != self.game.solo
            ):
                self.send(f"table {self.table} {self.player_name} play RE")
                self.re_sent = True
            return
        if action.startswith("SC."):
            return
        if self.game.to_move == self.game.seat and self.game.phase not in (0, 7):
            move = choose(self.agent, self.game.snapshot())
            self.send(f"table {self.table} {self.player_name} play {move}")


class ISSWire:
    """UTF-8 line transport without file buffering hiding pending socket data."""

    def __init__(self, sock):
        self.sock = sock
        self.pending = bytearray()

    def send(self, message, *, log=True):
        if "\n" in message or "\r" in message:
            raise ValueError("invalid ISS command")
        if log and not message.startswith("join "):
            LOG.info("send %s", message)
        self.sock.sendall((message + "\n").encode("utf-8"))

    def readline(self, *, timeout=20, keepalive=False):
        while b"\n" not in self.pending:
            ready, _, _ = select.select([self.sock], [], [], timeout)
            if not ready:
                if keepalive:
                    self.send("time", log=False)
                    continue
                raise TimeoutError("ISS server did not answer")
            block = self.sock.recv(65536)
            if not block:
                raise ConnectionError("ISS connection closed")
            self.pending.extend(block)
            if len(self.pending) > 4 * 1024 * 1024:
                raise ValueError("ISS line exceeds 4 MiB")
        raw, _, rest = self.pending.partition(b"\n")
        self.pending = bytearray(rest)
        return raw.rstrip(b"\r").decode("utf-8")


def build_agent(args):
    if args.agent == "policy":
        from skat.agents.policy import PolicyAgent

        if not args.policy:
            raise ValueError("policy agent requires --policy")
        agent = PolicyAgent(args.policy, device=args.device, seed=args.seed)
    elif args.agent == "mcts":
        from skat.agents.mcts import MCTSAgent

        sampler = None
        if args.belief_model:
            from skat.agents.belief.inference import BeliefSampler

            sampler = BeliefSampler(args.belief_model, device=args.device)
        agent = MCTSAgent(
            args.simulations, args.determinizations, seed=args.seed, sampler=sampler
        )
    else:
        from skat.agents.random import RandomAgent

        agent = RandomAgent(seed=args.seed)
    bidder = declarer = pickup = None
    if args.bidding == "neural":
        if not args.bidding_model:
            raise ValueError("neural bidding requires --bidding-model")
        from skat.agents.neural_bidding.inference import NeuralBidder

        bidder = NeuralBidder(
            args.bidding_model, device=args.device, threshold=args.bidding_threshold
        )
    if args.declaration == "neural":
        if not args.declaration_model:
            raise ValueError("neural declaration requires --declaration-model")
        from skat.agents.neural_declaring.inference import DeclarationScorer

        declarer = DeclarationScorer(args.declaration_model, device=args.device)
    if args.pickup == "neural":
        if not args.pickup_model:
            raise ValueError("neural pickup requires --pickup-model")
        from skat.agents.neural_pickup.inference import PickupScorer

        pickup = PickupScorer(args.pickup_model, device=args.device)
    return (
        agent.configure_bidding(args.bidding, bidder)
        .configure_declaration(args.declaration, declarer)
        .configure_pickup(args.pickup, pickup, args.pickup_margin)
    )


def run(args):
    if (
        args.agent == "policy"
        or "neural" in (args.bidding, args.declaration, args.pickup)
        or args.belief_model
    ):
        import torch

        torch.set_num_threads(1)
    agent = build_agent(args)
    client = ISSClient(
        agent,
        table=args.table,
        table_password=args.table_password,
        accept_invites=args.accept_invites,
    )
    password = os.environ.get("ISS_PASSWORD")
    if not password:
        raise ValueError("set ISS_PASSWORD in the environment")
    if not args.table and not args.accept_invites:
        raise ValueError("specify --table or --accept-invites")
    with socket.create_connection((args.host, args.port), timeout=15) as sock:
        sock.settimeout(None)
        wire = ISSWire(sock)
        client.send = wire.send
        wire.send(args.id)
        if wire.readline(timeout=15) != "password:":
            raise RuntimeError("ISS did not request a password")
        wire.send(password, log=False)
        welcome = wire.readline(timeout=15)
        if not welcome.startswith("Welcome "):
            raise RuntimeError(f"ISS login failed: {welcome}")
        actual_id = welcome.split()[1]
        LOG.info("logged in as %s", actual_id)
        if args.table:
            wire.send(f"join {args.table} {args.table_password}")
        while True:
            line = wire.readline(keepalive=True)
            LOG.debug("receive %s", line)
            client.handle(line)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Play on ISS using a Python Skat agent"
    )
    parser.add_argument("--host", default="bodo1.cs.ualberta.ca")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--id", required=True, help="ISS account name")
    parser.add_argument("--table", help="ISS table ID to join")
    parser.add_argument("--table-password", default=".")
    parser.add_argument("--accept-invites", action="store_true")
    parser.add_argument(
        "--agent", choices=("policy", "mcts", "random"), default="policy"
    )
    parser.add_argument("--policy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=600)
    parser.add_argument("--determinizations", type=int, default=20)
    parser.add_argument("--belief-model")
    parser.add_argument(
        "--bidding", choices=("conservative", "neural"), default="conservative"
    )
    parser.add_argument("--bidding-model")
    parser.add_argument("--bidding-threshold", type=float, default=2 / 3)
    parser.add_argument(
        "--declaration", choices=("heuristic", "neural"), default="heuristic"
    )
    parser.add_argument("--declaration-model")
    parser.add_argument("--pickup", choices=("always", "neural"), default="always")
    parser.add_argument("--pickup-model")
    parser.add_argument("--pickup-margin", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args)


if __name__ == "__main__":
    main()
