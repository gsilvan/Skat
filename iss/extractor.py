#!/usr/bin/env python3
import sys
import os.path

from iss import GameType, ISSGames


if len(sys.argv) < 1:
    print("python extractor.py <path-to-sgf-file>")
    exit()
sgf_file = sys.argv[1]

full_path = os.path.abspath(sgf_file)
directory = os.path.dirname(sgf_file)

suit_games = []
grand_games = []
null_games = []

iss_games = ISSGames(full_path)
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
    with open(os.path.join(directory, "suit_win_loss.sgf"), "a") as file:
        file.write(f"{str(game)}\n")
for game in grand_games:
    with open(os.path.join(directory, "grand_win_loss.sgf"), "a") as file:
        file.write(f"{str(game)}\n")
for game in null_games:
    with open(os.path.join(directory, "null_win_loss.sgf"), "a") as file:
        file.write(f"{str(game)}\n")
