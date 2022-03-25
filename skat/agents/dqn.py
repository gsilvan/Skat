from skat.agents import Agent
from skat.agents.rl.dqn import DQN
from skat.card import Card
from skat.games import Game
from skat.hand import FULL_HAND, Hand


class DQNAgent(Agent):
    def __init__(self, train=True):
        self.dqn = DQN()  # TODO: pass train to dqn
        self.train = train
        self.initial_state = None
        self.last_action = None
        self.last_cumulative_reward = 0

    def choose_card(self, valid_actions: set[Card]) -> Card:
        self.initial_state = self.state.public_state.to_numpy()
        self.last_cumulative_reward = self.state.trick_stack_value

        valid_actions = Hand([valid_actions]).as_tensor_mask

        # get an action
        self.last_action = self.dqn.select_action(
            self.state.public_state.to_numpy(), valid_actions
        )
        return FULL_HAND.index(self.last_action)

    def pickup_skat(self, state) -> bool:
        pass

    def bid(self, current_bid, offer=False) -> int:
        pass

    def declare_game(self, state) -> Game:
        pass

    def press_skat(self) -> list[Card]:
        pass

    def trick_done_event(self, is_terminal=False):
        if not self.train:
            # do not optimize if it's not in trainig
            return

        reward = self.state.trick_stack_value - self.last_cumulative_reward

        if is_terminal:
            next_state = None
        else:
            next_state = self.state.public_state.to_numpy()

        # store in buffer
        self.dqn.replay_buffer.push(
            state=self.initial_state,
            action=self.last_action,
            next_state=next_state,
            reward=reward,
        )

        self.dqn.episode += 1

        # optimize model
        self.dqn.optimize_model()
