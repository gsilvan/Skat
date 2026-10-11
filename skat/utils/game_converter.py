from iss import GameType
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame


def get_game(game_type: GameType):
    """Convert an ISS-GameType to a skat.game.Game instance."""
    match game_type:
        case GameType.DIAMONDS:
            return SuitGame(0)
        case GameType.HEARTS:
            return SuitGame(1)
        case GameType.SPADES:
            return SuitGame(2)
        case GameType.CLUBS:
            return SuitGame(3)
        case GameType.GRAND:
            return Grand()
        case GameType.NULL:
            return Null()
