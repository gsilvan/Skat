import unittest

from skat.agents.random import RandomAgent
from skat.game import Round


class RoundTest(unittest.TestCase):
    def setUp(self) -> None:
        agents = [RandomAgent(), RandomAgent(), RandomAgent()]
        self.simple_round = Round(
            skip_bidding=True, solo_player_id=0, agents=agents, start=False
        )

    def test_points(self):
        self.assertEqual(0, self.simple_round.points_soloist)
        self.assertEqual(0, self.simple_round.points_defenders)
        # start the round
        self.simple_round.start()
        points_dealt = (
            self.simple_round.points_soloist + self.simple_round.points_defenders
        )
        self.assertEqual(120, points_dealt)
