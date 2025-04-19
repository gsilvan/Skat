from iss import GameType
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame


def get_game(game_type: GameType):
    """Convert an ISS-GameType to a skat.game.Game instance."""
    match game_type:
        case GameType.DIAMONDS:
            print("game=diamonds")
            return SuitGame(0)
        case GameType.HEARTS:
            print("game=hearts")
            return SuitGame(1)
        case GameType.SPADES:
            print("game=spades")
            return SuitGame(2)
        case GameType.CLUBS:
            print("game=clubs")
            return SuitGame(3)
        case GameType.GRAND:
            return Grand()
        case GameType.NULL:
            return Null()
