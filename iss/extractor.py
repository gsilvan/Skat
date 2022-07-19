#!/usr/bin/env python3
from iss import ISSGames, GameType


if __name__ == "__main__":
    suit_games = []
    grand_games = []
    null_games = []

    iss_games = ISSGames("/home/silvan/Downloads/iss-games-04-2021.sgf")
    full_sample = iss_games.sample(n=len(iss_games))
    for game in full_sample:
        if not game.is_valid:
            continue
        # if not game.is_won:
        #     continue
        match game.type:
            case GameType.DIAMONDS | GameType.HEARTS | GameType.SPADES | GameType.CLUBS:
                suit_games.append(game)
            case GameType.GRAND:
                grand_games.append(game)
            case GameType.NULL:
                null_games.append(game)
    for game in suit_games:
        with open("/home/silvan/iss_games/suit_win_loss.sgf", "a") as file:
            file.write(f"{str(game)}\n")
    for game in grand_games:
        with open("/home/silvan/iss_games/grand_win_loss.sgf", "a") as file:
            file.write(f"{str(game)}\n")
    for game in null_games:
        with open("/home/silvan/iss_games/null_win_loss.sgf", "a") as file:
            file.write(f"{str(game)}\n")
