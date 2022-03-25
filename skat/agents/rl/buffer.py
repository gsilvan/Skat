import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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
