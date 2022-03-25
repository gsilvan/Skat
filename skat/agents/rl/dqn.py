import random

import torch
import torch.nn as nn
import torch.optim

import skat.models

from .buffer import ReplayBuffer, Transition
from .mask import MaskedCategorical


class DQN:
    def __init__(
        self,
        device: str = "cpu",
        epsilon: float = 0.90,
        epsilon_decay: float = 0.995,
        batch_size: int = 128,
        gamma: float = 0.999,
        target_update: int = 200,
        buffer_size: int = 10000,
        learning_rate: float = 1e-3,
        model=skat.models.SuitSoloNet,
    ) -> None:
        # use either 'cuda' or 'cpu'
        self.device = device

        # Hyperparameter
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.learning_rate = learning_rate

        # Networks
        self.policy_net = model().to(self.device)
        self.target_net = model().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

        # buffer
        self.replay_buffer = ReplayBuffer(buffer_size=10000)

        # step counter
        self.episode = 0

    def select_action(self, state, valid_actions: torch.Tensor) -> int:
        # decay epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)

        # TODO: invalid action masking
        valid_indices = [i for i, t in enumerate(valid_actions) if t]

        if random.random() <= self.epsilon:
            # select with epsilon probability a random action
            return random.choice(valid_indices)
        else:
            # select with (1-epsilon) probability a greedy argmax action
            mc = MaskedCategorical(
                logits=self.policy_net(torch.tensor(state)), mask=valid_actions
            )
            return int(mc.probs.argmax())
        # return action.detach().cpu().numpy()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # get a random sample from buffer (size=batch_size)
        transitions = self.replay_buffer.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # predict Q(s)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # calculate loss
        loss = nn.SmoothL1Loss(state_action_values, expected_state_action_values)

        # optimize model
        self.optimizer.zero_grad()
        loss.backwards()
        for param in self.policy_net.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

        # Update the target net
        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
