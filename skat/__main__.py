from .agents.command_line import CommandLineAgent
from .agents.random import RandomAgent
from .game import Round

if __name__ == "__main__":
    a = [CommandLineAgent(), RandomAgent(), RandomAgent()]
    r = Round(skip_bidding=True, solo_player_id=0, agents=a, start=False, verbose=True)
    r.start()
