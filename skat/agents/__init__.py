from abc import ABC, abstractmethod


class Agent(ABC):
    """Defines a minimal Agent Object"""

    @abstractmethod
    def play(self):
        raise NotImplementedError
