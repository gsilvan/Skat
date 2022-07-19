#!/usr/bin/env python3
from iss import ISSGames


if __name__ == "__main__":
    iss_games = ISSGames("/home/silvan/Desktop/iss-tail-100000.sgf")
    sample = iss_games.sample(n=10000, seed=None)
    result = [0, 0, 0, 0, 0]
    for game in sample:
        result[0] += 1
        if game.is_valid:
            result[1] += 1
        else:
            result[2] += 1
        if game.is_won:
            result[3] += 1
        else:
            result[4] += 1
    print(result)
