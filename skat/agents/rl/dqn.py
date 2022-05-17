import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

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

        # tensorboard
        self.writer = SummaryWriter()

    def select_action(
        self, state: torch.Tensor, valid_actions: torch.Tensor
    ) -> torch.Tensor:
        # decay epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.085)

        valid_indices: list[int] = [i for i, t in enumerate(valid_actions) if t]

        action = torch.zeros((1, 32), dtype=torch.long)
        if random.random() <= self.epsilon:
            # select with epsilon probability a random action
            action[0][random.choice(valid_indices)] = 1
            return action
        else:
            # select with (1-epsilon) probability a greedy argmax action
            prediction = self.policy_net(state)
            mc = MaskedCategorical(
                logits=prediction,
                mask=valid_actions,
            )
            argmax = mc.probs.argmax()
            action[0][argmax] = 1
            return action

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

        # Q(s_t, a)
        # state_action_values
        q = (
            self.policy_net(state_batch)
            .gather(1, action_batch.argmax(dim=1).unsqueeze(1))
            .squeeze(1)
        )

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch.squeeze(1)
        loss = F.mse_loss(q, expected_state_action_values)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target net
        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
