import os
import sys
from contextlib import contextmanager, nullcontext
from random import Random

import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

import skat.models
from skat.models.observation import OBSERVATION_SCHEMA, validate_observation_schema
from skat.rewards import REWARD_OBJECTIVE

from .buffer import ReplayBuffer, Transition
from .mask import MaskedCategorical
from .result_q import ResultQ, resolved_result, secured_victory

DEFAULT_MODEL_PATH = "./trained_models/checkpoint.pt"

#: What `--dqn-arch` takes. Separate from `skat.models.ARCHITECTURES`, which is
#: the *policy* family: "mlp" there means `PolicyNet`, which carries dropout
#: for supervised training, and that is the wrong net to regress Q-values with.
#:
#: The transformer is built with `dropout=0.0` deliberately. It is here to test
#: whether attention over a card set helps a Q-function the way it helped the
#: policy, and leaving dropout on would confound that with a regularisation
#: change - and put noise on the bootstrap target besides.
DQN_ARCHITECTURES: dict[str, tuple] = {
    "mlp": (skat.models.QNet, {}),
    "transformer": (skat.models.CardTransformer, {"dropout": 0.0}),
}


@contextmanager
def evaluating(net: torch.nn.Module):
    """
    Run `net` in eval mode with autograd off, then put it back as it was.

    Both halves matter and for different reasons. Autograd off is the cheap
    win: `select_action` built a graph on every card it chose and threw it
    away.

    Eval mode is the correctness half, and it is only latent today because
    `QNet` has no dropout or batch norm. The moment a model that does is used
    - `CardTransformer` has dropout, and `--dqn-arch transformer` hands it
    straight to `DQN` - a training-mode forward would make action selection
    stochastic on top of the epsilon-greedy schedule, and would turn the
    bootstrap target into a random variable. TD learning chases a moving
    target already; multiplicative noise on top of it is not a help.
    """
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            yield net
    finally:
        net.train(was_training)


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
        batch_size: int = 256,
        learn_start: int = 10000,
        gamma: float = 1.0,
        target_update: int = 4096,
        buffer_size: int = 100000,
        learning_rate: float = 1e-4,
        model=skat.models.QNet,
        model_kwargs: dict | None = None,
        model_path: str = DEFAULT_MODEL_PATH,
        seed: int | None = None,
        log_dir: str | None = None,
        result_priority: bool = False,
    ) -> None:
        # "auto", "cpu", "cuda", "cuda:1", ... - see resolve_device
        self.device = resolve_device(device)
        # Two torch installs can resolve differently (venv vs system), so say
        # which one won rather than let a long run sit on the cpu unnoticed.
        print(f"DQN device: {describe_device(self.device)}")

        if learn_start < batch_size:
            raise ValueError(
                f"learn_start ({learn_start}) is below batch_size ({batch_size}): "
                "the first gradient step would ask the replay buffer for more "
                "transitions than it can hold at that point."
            )
        if result_priority and gamma != 1.0:
            raise ValueError(
                "win-first rewards require gamma=1: a later win must not be discounted"
            )

        # Hyperparameter
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learn_start = learn_start
        # Undiscounted, on purpose. A round is exactly ten tricks and every
        # terminal transition is flagged, so returns are bounded without a
        # discount doing it. The per-trick reward is a difference of
        # `Round.cumulative_reward`, so the rewards telescope: summed
        # undiscounted over a round they come to exactly the final outcome,
        # which is the number the agent should be predicting. gamma was 0.5,
        # which shrank the terminal win/schneider/schwarz bonus to 0.5^9 ~
        # 0.002 by the first trick - it was, in effect, playing for card points
        # and ignoring whether it won.
        self.gamma = gamma
        self.target_update = target_update
        self.learning_rate = learning_rate

        self.seed = seed
        self.result_priority = result_priority
        self.rng = Random(seed)

        # Networks.  fork_rng makes construction reproducible without changing
        # the torch stream owned by another agent in the same tournament.
        model_kwargs = dict(model_kwargs or {})
        init_context = (
            torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
        )
        with init_context:
            if seed is not None:
                torch.manual_seed(seed)
            self.policy_net = (
                ResultQ(model, model_kwargs)
                if result_priority
                else model(**model_kwargs)
            ).to(self.device)
            self.target_net = (
                ResultQ(model, model_kwargs)
                if result_priority
                else model(**model_kwargs)
            ).to(self.device)
        print(
            f"DQN model: {model.__name__} "
            f"({sum(p.numel() for p in self.policy_net.parameters())} parameters)"
        )

        self.model_path = model_path

        if os.path.exists(self.model_path):
            # the resolved device, not the raw spec: map_location has no idea
            # what "auto" means and refuses to restore the storages
            existing_model = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
            if result_priority:
                if existing_model.get("objective") != REWARD_OBJECTIVE:
                    raise RuntimeError(
                        f"{self.model_path} uses the old reward objective. "
                        "Keep it as a baseline and train win-first DQN at a fresh -m path."
                    )
                validate_observation_schema(
                    existing_model.get("observation_schema"), self.model_path
                )
                existing_model = existing_model["state_dict"]
            try:
                self.policy_net.load_state_dict(existing_model)
            except RuntimeError as mismatch:
                raise RuntimeError(
                    f"{self.model_path} does not fit the current network.\n"
                    f"  expected input width: {skat.models.INPUT_SIZE}\n"
                    "A checkpoint is only loadable by the network and the state "
                    "layout it was trained on, so a change to either - to "
                    "`Round.get_state`, or to the module names inside "
                    "`skat.models.QNet` - retires the ones that came before. "
                    "Move the file aside (or point -m at a fresh path) and "
                    "train again.\n"
                    f"  original error: {mismatch}"
                ) from mismatch

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

        # buffer
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, seed=seed)

        # step counter
        self.episode = 0
        self.gradient_steps = 0
        # latched by the first gradient step, so the announcement happens once
        self.learning = False

        # tensorboard
        self.writer = SummaryWriter(log_dir=log_dir)

    @property
    def warming_up(self) -> bool:
        """True until the buffer holds enough transitions to learn from."""
        return len(self.replay_buffer) < self.learn_start

    @property
    def exploration_rate(self) -> float:
        """
        How often to pick a random card.

        Flat 1.0 for the whole warm-up. The seed transitions are the only thing
        the net ever sees before its first gradient step, so they should come
        from as wide a spread of positions as possible - and a decaying epsilon
        would instead spend most of the warm-up acting greedily on a network
        that is still at its random initialisation, i.e. one arbitrary
        deterministic policy played over and over.
        """
        return 1.0 if self.warming_up else self.epsilon

    def select_action(
        self,
        state: torch.Tensor,
        valid_actions: torch.Tensor,
        explore: bool = True,
        allow_bonus: bool = True,
    ) -> torch.Tensor:
        rate = self.exploration_rate
        if explore and not self.warming_up:
            # decay per action, and only while exploring - epsilon is never read
            # on an evaluation pass, so decaying there just burns the schedule
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        valid_indices: list[int] = [i for i, t in enumerate(valid_actions) if t]

        # The observation and the legal-move mask are built on the cpu (that is
        # where the replay buffer wants them); move them to wherever the net is.
        state = state.to(self.device)
        valid_actions = valid_actions.to(self.device)

        action = torch.zeros((1, 32), dtype=torch.long)
        if explore and self.rng.random() <= rate:
            # select with epsilon probability a random action
            action[0][self.rng.choice(valid_indices)] = 1
            return action
        else:
            # select with (1-epsilon) probability a greedy argmax action
            with evaluating(self.policy_net):
                if (
                    getattr(self, "result_priority", False)
                    and allow_bonus
                    and secured_victory(state).item()
                ):
                    prediction = self.policy_net.bonus(state)
                else:
                    prediction = self.policy_net(state)
            mc = MaskedCategorical(
                logits=prediction,
                mask=valid_actions,
            )
            argmax = mc.probs.argmax()
            action[0][argmax] = 1
            return action

    def bootstrap(
        self, next_states: torch.Tensor, next_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        `max_a' Q_target(s', a')`, over the *legal* continuations only.

        Unmasked, this maxed over all 32 cards - and a hand holds at most ten,
        so the target was nearly always the value of a move the agent could not
        make, most often one already lying in a completed trick. Every state
        ended up valued by a fantasy continuation.
        """
        # The target network is never trained, only copied into - but it is
        # constructed in training mode like any other module, so this is what
        # keeps a dropout-bearing architecture from randomising the target.
        with evaluating(self.target_net):
            q = self.target_net(next_states)
        masked = q.masked_fill(~next_masks.to(q.device), torch.finfo(q.dtype).min)
        values = masked.max(1)[0]
        if getattr(self, "result_priority", False):
            values = values.masked_fill(resolved_result(next_states), 0.0)
        return values

    def optimize_model(self):
        self.episode += 1
        if self.warming_up:
            return
        if not self.learning:
            # a long warm-up looks identical to a hung run from the outside
            self.learning = True
            print(
                f"DQN: warm-up done, {len(self.replay_buffer)} transitions "
                f"collected in {self.episode} steps - learning starts "
                f"(batch {self.batch_size})"
            )

        # get a random sample from buffer (size=batch_size)
        transitions = self.replay_buffer.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        live_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = (
            torch.cat(live_states).to(self.device) if live_states else None
        )
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

        # The masks belonging to those next states, in the same order.
        live_masks = [m for m in batch.next_mask if m is not None]
        next_masks = torch.stack(live_masks) if live_masks else None

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if live_states:
            next_state_values[non_final_mask] = self.bootstrap(
                non_final_next_states, next_masks
            )

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma + reward_batch[:, 0]
        )
        loss = F.smooth_l1_loss(expected_state_action_values, q)
        if getattr(self, "result_priority", False):
            next_bonus = torch.zeros(self.batch_size, device=self.device)
            if live_states:
                with evaluating(self.target_net):
                    bonus_q = self.target_net.bonus(non_final_next_states)
                next_bonus[non_final_mask] = (
                    bonus_q.masked_fill(
                        ~next_masks.to(self.device), torch.finfo(bonus_q.dtype).min
                    )
                    .max(1)
                    .values
                )
            bonus = (
                self.policy_net.bonus(state_batch)
                .gather(1, action_batch.argmax(dim=1).unsqueeze(1))
                .squeeze(1)
            )
            # Learn this head only on states where victory was already secure.
            secure = secured_victory(state_batch)
            if secure.any():
                bonus_loss = F.smooth_l1_loss(
                    bonus[secure],
                    (reward_batch[:, 1] + self.gamma * next_bonus)[secure],
                )
                loss = loss + bonus_loss
                self.writer.add_scalar("Loss/bonus", bonus_loss, self.episode)
        self.writer.add_scalar("Loss/train", loss, self.episode)
        self.writer.add_scalar("epsilon", self.epsilon, self.episode)
        self.writer.add_scalar("buffer size", len(self.replay_buffer), self.episode)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.gradient_steps += 1

        # Update the target net
        if self.episode % self.target_update == 0:
            print(
                f"\n********\nepisode: {self.episode}\nepsilon: {self.epsilon}\nloss: {loss}"
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # if not os.path.exists(self.model_path):
            #     os.mkdir(self.model_path)
            self.save()

        if self.episode % 2222 == 0:
            pass

    def save(self) -> None:
        """Persist the current policy, including the tail after the last update."""
        directory = os.path.dirname(self.model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = self.policy_net.state_dict()
        if self.result_priority:
            payload = {
                "objective": REWARD_OBJECTIVE,
                "observation_schema": OBSERVATION_SCHEMA,
                "state_dict": payload,
            }
        torch.save(payload, self.model_path)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
