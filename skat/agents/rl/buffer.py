import random
from collections import deque, namedtuple

# `next_mask` is the legal-move mask that belongs to `next_state`, and it is
# what keeps the bootstrap honest: without it the target maxes over all 32
# cards, including the ones the agent does not hold. It is None exactly when
# `next_state` is (a terminal transition bootstraps from nothing).
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "next_mask", "reward")
)


class ReplayBuffer:
    def __init__(self, buffer_size: int = 1000) -> None:
        """ReplayBuffer stores Transitions in a FIFO buffer."""
        self.buffer: deque[Transition] = deque([], maxlen=buffer_size)

    def push(self, **kwargs):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(**kwargs))

    def sample(self, batch_size: int):
        """Return a random sample from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Buffer size."""
        return len(self.buffer)
