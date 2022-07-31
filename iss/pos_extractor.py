#!/usr/bin/env python3
from iss import ISSGames


if __name__ == "__main__":
    suit_games = []
    grand_games = []
    null_games = []

    positions = [[], [], []]

    iss_games = ISSGames("/home/silvan/iss_games/suit_win_loss.sgf")
    full_sample = iss_games.sample(n=len(iss_games))
    for game in full_sample:
        positions[game.soloist].append(game)

    for i, games in enumerate(positions):
        with open(f"/home/silvan/iss_games/suit_win_los_pos_{i}", "a") as file:
            for game in games:
                file.write(f"{str(game)}\n")

