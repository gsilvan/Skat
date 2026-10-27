"""Jointly choose a discard pair and declaration from calibrated win scores."""

from itertools import combinations

import numpy as np
import torch

from skat.agents.declaring import DeclarationChoice, HeuristicDeclarer, validate_pressed
from skat.agents.neural_declaring.checkpoint import load_declaration
from skat.agents.neural_declaring.features import declaration_state
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame
from skat.utils.reproducibility import resolve_device


def candidate_games(cards, highest_bid):
    """Only simple pickup games that safely cover the won bid."""
    games = [SuitGame(suit) for suit in range(4)] + [Grand(), Null()]
    return [game for game in games if game.spielwert(cards) >= highest_bid]


def candidates(view, *, pressed=None):
    cards = tuple(sorted(view.hand, key=lambda card: card.np_index))
    if len(cards) != 12 or len(view.skat) != 2:
        raise ValueError("declaration requires twelve cards and visible original skat")
    games = candidate_games(cards, view.highest_bid)
    pairs = (
        combinations(cards, 2) if pressed is None else [validate_pressed(view, pressed)]
    )
    for pair in pairs:
        kept = tuple(card for card in cards if card not in pair)
        for game in games:
            yield (
                DeclarationChoice(pair, game),
                declaration_state(
                    kept, pair, view.skat, game, view.seat_id, view.dealer
                ),
            )


class DeclarationScorer:
    """Immutable model weights, safe to share between agents."""

    def __init__(self, path, *, device="cpu"):
        self.device = resolve_device(device)
        self.model, self.temperature, self.metadata = load_declaration(
            path, self.device
        )

    def probabilities(self, features):
        states = torch.from_numpy(np.stack(features)).to(self.device)
        with torch.inference_mode():
            return (self.model(states) / self.temperature).sigmoid().cpu().numpy()


class NeuralDeclarer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.baseline = HeuristicDeclarer()

    def choose_pickup(self, view):
        return True

    def choose_after_pickup(self, view, *, pressed=None):
        options = list(candidates(view, pressed=pressed))
        if not options:
            return self.baseline.choose_after_pickup(view, pressed=pressed)
        scores = self.scorer.probabilities([state for _, state in options])
        if len(scores) != len(options) or not np.isfinite(scores).all():
            raise ValueError("invalid declaration model scores")
        return options[int(np.argmax(scores))][0]

    def choose_hand(self, view):
        return self.baseline.choose_hand(view)
