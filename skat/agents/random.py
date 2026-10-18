from skat.agents import Agent


class RandomAgent(Agent):
    """Uniform legal card play; shares the conservative declaration baseline."""

    def choose_card(self, valid_moves):
        return self.rng.choice(sorted(valid_moves, key=lambda card: card.np_index))
