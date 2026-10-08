import os
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from skat.agents import Agent
from skat.agents.bidding import BiddingStyle
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.rl.dqn import DQN, DQN_ARCHITECTURES
from skat.card import Card
from skat.hand import FULL_HAND, Hand


class DQNAgent(HeuristicDeclarer, Agent):
    """
    Learns card play; everything else is automatic.

    It used to subclass `CommandLineAgent` to borrow prompts for the decisions
    the network does not make, which meant an unattended training run blocked
    on `input()` at the first bid. Bidding now comes from `Agent` (so the
    `BiddingStyle` applies, conservative by default) and the skat/declaration
    phase from `HeuristicDeclarer`.
    """

    def __init__(
        self,
        train=False,
        path=None,
        epsilon=None,
        epsilon_decay=None,
        device=None,
        batch_size=None,
        learn_start=None,
        buffer_size=None,
        gamma=None,
        arch=None,
        bidding=BiddingStyle.CONSERVATIVE,
        seed=None,
        log_dir=None,
    ):
        super().__init__(bidding=bidding, seed=seed)
        # Only forward what was actually given - passing None through would
        # overwrite DQN's own defaults with it and blow up on first use.
        options = {
            "result_priority": True,
            "model_path": path,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "device": device,
            "batch_size": batch_size,
            "learn_start": learn_start,
            "buffer_size": buffer_size,
            "gamma": gamma,
            "seed": seed,
            "log_dir": os.path.join(log_dir, "optimizer") if log_dir else None,
        }
        if arch is not None:
            if arch not in DQN_ARCHITECTURES:
                raise ValueError(
                    f"unknown dqn architecture {arch!r}; known: "
                    f"{', '.join(sorted(DQN_ARCHITECTURES))}"
                )
            model, model_kwargs = DQN_ARCHITECTURES[arch]
            options["model"] = model
            options["model_kwargs"] = model_kwargs
        self.dqn = DQN(**{k: v for k, v in options.items() if v is not None})
        self.train = train
        self.initial_state: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_cumulative_reward: float = 0
        self.last_cumulative_bonus: float = 0
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, "agent") if log_dir else None
        )
        self.epoch = 0

    def _current_reward(self) -> float:
        return self.state.public_state.cumulative_reward(self.state.seat_id)

    def choose_card(self, valid_actions: set[Card]) -> Card:
        self.initial_state = self.state.public_state.get_state_t(
            player_id=self.state.seat_id
        )
        self.last_cumulative_reward = self._current_reward()
        self.last_cumulative_bonus = self.state.public_state.cumulative_bonus(
            self.state.seat_id
        )

        valid_actions = Hand(tuple(valid_actions)).as_tensor_mask

        known_overbid = self.state.public_state.observer_overbid(self.state.seat_id)
        # An uncertain overbid cannot invalidate a defender's 60 public points,
        # but it can still invalidate a soloist's apparent point victory.
        allow_bonus = known_overbid is False or (
            known_overbid is None
            and self.state.seat_id != self.state.public_state.solo_player_id
        )

        # get a prediction
        prediction = self.dqn.select_action(
            self.initial_state,
            valid_actions,
            explore=self.train,
            allow_bonus=allow_bonus,
        )
        self.last_action = prediction
        choice = FULL_HAND[int(prediction.argmax())]
        self.state.hand.remove(choice)
        return choice

    def trick_done_event(self, is_terminal=False):
        if not self.train:
            # do not optimize if it's not in trainig
            return

        reward = torch.empty((1, 2), dtype=torch.float64)
        reward[0][0] = self._current_reward() - self.last_cumulative_reward
        reward[0][1] = (
            self.state.public_state.cumulative_bonus(self.state.seat_id)
            - self.last_cumulative_bonus
        )

        if is_terminal:
            next_state = None
            next_mask = None
            if self.state.seat_id == self.state.public_state.solo_player_id:
                points = self.state.public_state.points_soloist
                self.writer.add_scalar("Points solo", points, self.epoch)
            else:
                points = self.state.public_state.points_defenders
                self.writer.add_scalar("Points defenders", points, self.epoch)
            self.epoch += 1
        else:
            next_state = self.state.public_state.get_state_t(
                player_id=self.state.seat_id
            )
            # The cards still in hand - `choose_card` already removed the one
            # just played. The next trick has not been led yet, so follow-suit
            # cannot narrow this further; the remaining hand is the tightest
            # legal-move mask that exists at this point, and it is what stops
            # the bootstrap from valuing cards the agent cannot play.
            next_mask = self.state.hand.as_tensor_mask

        # store in buffer
        self.dqn.replay_buffer.push(
            state=self.initial_state,
            action=self.last_action,
            next_state=next_state,
            next_mask=next_mask,
            reward=reward,
        )

        # optimize model
        self.dqn.optimize_model()

    def save(self) -> None:
        self.dqn.save()

    def close(self) -> None:
        self.dqn.close()
        self.writer.flush()
        self.writer.close()
