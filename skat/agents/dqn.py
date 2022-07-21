from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

import skat.agents.command_line
from skat.agents.rl.dqn import DQN
from skat.card import Card
from skat.hand import FULL_HAND, Hand


class DQNAgent(skat.agents.command_line.CommandLineAgent):
    def __init__(self, train=False, path=None, epsilon=None, epsilon_decay=None):
        super().__init__()
        self.dqn = DQN(model_path=path, epsilon=epsilon, epsilon_decay=epsilon_decay)
        self.train = train
        self.initial_state: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_cumulative_reward: float = 0
        self.writer = SummaryWriter()
        self.epoch = 0

    def _current_reward(self) -> float:
        return self.state.public_state.cumulative_reward(self.state.seat_id)

    def choose_card(self, valid_actions: set[Card]) -> Card:
        self.initial_state = self.state.public_state.get_state_t(
            player_id=self.state.seat_id
        )
        self.last_cumulative_reward = self._current_reward()

        valid_actions = Hand(tuple(valid_actions)).as_tensor_mask

        # get a prediction
        prediction = self.dqn.select_action(
            self.initial_state, valid_actions, explore=self.train
        )
        self.last_action = prediction
        choice = FULL_HAND[int(prediction.argmax())]
        self.state.hand.remove(choice)
        return choice

    def trick_done_event(self, is_terminal=False):
        if not self.train:
            # do not optimize if it's not in trainig
            return

        reward = torch.empty((1, 1), dtype=torch.float64)
        reward[0][0] = self._current_reward() - self.last_cumulative_reward

        if is_terminal:
            next_state = None
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

        # store in buffer
        self.dqn.replay_buffer.push(
            state=self.initial_state,
            action=self.last_action,
            next_state=next_state,
            reward=reward,
        )

        # optimize model
        self.dqn.optimize_model()
