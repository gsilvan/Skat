from typing import Optional

import torch

import skat.agents.command_line
from skat.agents.rl.dqn import DQN
from skat.card import Card
from skat.hand import FULL_HAND, Hand


class DQNAgent(skat.agents.command_line.CommandLineAgent):
    def __init__(self, train=True):
        super().__init__()
        self.dqn = DQN()  # TODO: pass train to dqn
        self.train = train
        self.initial_state: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_cumulative_reward: int = 0
        self.lcr = 0
        self.writer = SummaryWriter()
        self.epoch = 0

    def choose_card(self, valid_actions: set[Card]) -> Card:
        self.initial_state = self.state.public_state.get_state_t(
            player_id=self.state.seat_id
        )
        self.last_cumulative_reward = self.state.trick_stack_value

        valid_actions = Hand(tuple(valid_actions)).as_tensor_mask

        # get a prediction
        prediction = self.dqn.select_action(self.initial_state, valid_actions)
        self.last_action = prediction
        choice = FULL_HAND[int(prediction.argmax())]
        self.state.hand.remove(choice)
        return choice

    def trick_done_event(self, is_terminal=False):
        if not self.train:
            # do not optimize if it's not in trainig
            return

        reward = torch.empty((1, 1), dtype=torch.long)
        reward[0][0] = self.state.trick_stack_value - self.last_cumulative_reward

        if is_terminal:
            next_state = None
            self.writer.add_scalar("Points", self.state.trick_stack_value, self.epoch)
            self.epoch += 1
        else:
            next_state = self.state.public_state.get_state_t(
                player_id=self.state.seat_id
            )

        # store in buffer
        self.dqn.replay_buffer.push(
            state=self.initial_state,
            action=self.last_action,
            next_state=next_state,
            reward=reward,
        )

        self.dqn.episode += 1

        # optimize model
        self.dqn.optimize_model()
