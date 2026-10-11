from skat.agents import Agent
from skat.agents.declaring import HeuristicDeclarer


class RandomAgent(HeuristicDeclarer, Agent):
    """Uniform legal card play; shares the conservative declaration baseline."""

    def choose_card(self, valid_moves):
        return self.rng.choice(sorted(valid_moves, key=lambda card: card.np_index))
