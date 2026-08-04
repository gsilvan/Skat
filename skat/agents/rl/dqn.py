import os
import random
import sys

import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

import skat.models

from .buffer import ReplayBuffer, Transition
from .mask import MaskedCategorical

DEFAULT_MODEL_PATH = "./trained_models/checkpoint.pt"


def gpu_backend() -> str:
    """
    Which GPU stack this torch build was compiled against.

    `hip` is checked first because a ROCm build reports `version.cuda is None`
    while still answering to the device string "cuda" - AMD cards are driven
    through the CUDA API, so "cuda" does not imply NVIDIA.
    """
    if getattr(torch.version, "hip", None):
        return f"ROCm {torch.version.hip}"
    if torch.version.cuda:
        return f"CUDA {torch.version.cuda}"
    return "no GPU support"


def describe_device(device: torch.device) -> str:
    """`device`, named, for the training log."""
    if device.type != "cuda":
        return str(device)
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:  # pragma: no cover - driver-dependent
        name = "unknown gpu"
    return f"{device} ({name}, {gpu_backend()})"


def resolve_device(spec: str = "auto") -> torch.device:
    """
    Turn a `--device` string into a `torch.device`.

    "auto" takes cuda when it is genuinely usable and falls back to cpu, so the
    same command works on a laptop and on a training box. Anything torch
    understands is accepted, including "cuda:1".

    A gpu request that cannot be honoured is refused here rather than left to
    surface as an opaque failure on the first forward pass.
    """
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {spec!r} was requested but torch reports no usable GPU.\n"
            f"  torch:      {torch.__version__} ({gpu_backend()})\n"
            f"  python:     {sys.executable}\n"
            "A GPU needs a torch built for one - CUDA for NVIDIA, ROCm for AMD; "
            'both are addressed as "cuda". Check that the interpreter above is '
            "the one holding that build: this project's own venv pins the "
            "CPU-only wheel index (see [tool.uv.sources] in pyproject.toml), so "
            "a GPU run usually means a different interpreter or a re-pinned "
            "index."
        )
    return device


class DQN:
    def __init__(
        self,
        device: str = "auto",
        epsilon: float = 0.98,
        epsilon_decay: float = 0.99995,
        epsilon_min: float = 0.10,
        batch_size: int = 10000,
        gamma: float = 0.5,
        target_update: int = 4096,
        buffer_size: int = 50000,
        learning_rate: float = 1e-4,
        model=skat.models.SuitSoloNet,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        # "auto", "cpu", "cuda", "cuda:1", ... - see resolve_device
        self.device = resolve_device(device)
        # Two torch installs can resolve differently (venv vs system), so say
        # which one won rather than let a long run sit on the cpu unnoticed.
        print(f"DQN device: {describe_device(self.device)}")

        # Hyperparameter
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.learning_rate = learning_rate

        # Networks
        self.policy_net = model().to(self.device)
        self.target_net = model().to(self.device)

        self.model_path = model_path

        if os.path.exists(self.model_path):
            # the resolved device, not the raw spec: map_location has no idea
            # what "auto" means and refuses to restore the storages
            existing_model = torch.load(self.model_path, map_location=self.device)
            self.policy_net.load_state_dict(existing_model)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

        # buffer
        self.replay_buffer = ReplayBuffer(buffer_size=100000)

        # step counter
        self.episode = 0

        # tensorboard
        self.writer = SummaryWriter()

    def select_action(
        self, state: torch.Tensor, valid_actions: torch.Tensor, explore: bool = True
    ) -> torch.Tensor:
        # decay epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        valid_indices: list[int] = [i for i, t in enumerate(valid_actions) if t]

        # The observation and the legal-move mask are built on the cpu (that is
        # where the replay buffer wants them); move them to wherever the net is.
        state = state.to(self.device)
        valid_actions = valid_actions.to(self.device)

        action = torch.zeros((1, 32), dtype=torch.long)
        if explore and random.random() <= self.epsilon:
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
        self.episode += 1
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
        ).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

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
            next_state_values * self.gamma + reward_batch.squeeze(1)
        )
        loss = F.smooth_l1_loss(expected_state_action_values, q)
        self.writer.add_scalar("Loss/train", loss, self.episode)
        self.writer.add_scalar("epsilon", self.epsilon, self.episode)
        self.writer.add_scalar("buffer size", len(self.replay_buffer), self.episode)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target net
        if self.episode % self.target_update == 0:
            print(
                f"\n********\nepisode: {self.episode}\nepsilon: {self.epsilon}\nloss: {loss}"
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # if not os.path.exists(self.model_path):
            #     os.mkdir(self.model_path)
            torch.save(self.policy_net.state_dict(), self.model_path)

        if self.episode % 2222 == 0:
            pass
