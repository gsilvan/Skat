from __future__ import annotations

import numpy as np

from skat.agents import Agent
from skat.agents.belief.samples import KNOWN
from skat.agents.search import infer_voids
from skat.card import Card
from skat.hand import FULL_HAND
from skat.models.belief import NEXT, PREVIOUS, SKAT


def void_block(round_, seat: int) -> np.ndarray:
    """
    The two opponents' void masks, `(64,)`, in the observer's own seat order:
    the next player's 32 first, then the previous player's.

    Built with the search's own `infer_voids` so the network is trained on
    exactly the evidence the sampler will be constraining with - and including
    the live trick, whose one or two cards are the freshest void evidence
    there is.
    """
    assert round_.game is not None
    voids = infer_voids(
        [*round_.trick_history.buffer, round_.game.trick], round_.game.trick
    )
    block = np.zeros(2 * len(FULL_HAND), dtype=np.float32)
    for offset, opponent in enumerate(((seat + 1) % 3, (seat + 2) % 3)):
        for card in voids.get(opponent, ()):
            block[offset * len(FULL_HAND) + card.np_index] = 1.0
    return block


def belief_input(round_, seat: int) -> np.ndarray:
    """The observation the belief network sees: `get_state` plus the voids."""
    return np.concatenate([round_.get_state(player_id=seat), void_block(round_, seat)])


def observer_view(round_, seat: int) -> np.ndarray:
    """
    Where every card really is, from `seat`'s chair: `NEXT`, `PREVIOUS`, `SKAT`
    or `KNOWN` for the ones it can already see.

    "Can already see" has to mean exactly what `MCTSAgent.determinize` means by
    it, or the network is trained on a different question from the one it gets
    asked. That is: the observer's own hand, every card played so far, and the
    skat only when the observer is the soloist and picked it up.
    """
    played = {card for trick in round_.trick_history.buffer for _, card in trick.buffer}
    assert round_.game is not None
    played |= {card for _, card in round_.game.trick.buffer}

    hand = set(round_.player[seat].hand)
    soloist = round_.solo_player_id
    known_skat = set(round_.skat) if seat == soloist and not round_.hand_game else set()

    next_seat, previous_seat = (seat + 1) % 3, (seat + 2) % 3
    where: dict[Card, int] = {}
    for card in round_.player[next_seat].hand:
        where[card] = NEXT
    for card in round_.player[previous_seat].hand:
        where[card] = PREVIOUS
    for card in round_.skat:
        where[card] = SKAT

    seen = played | hand | known_skat
    locations = np.full(len(FULL_HAND), KNOWN, dtype=np.int8)
    for card in FULL_HAND:
        if card not in seen and card in where:
            locations[card.np_index] = where[card]
    return locations


class BeliefRecorder(Agent):
    """
    Mixin that writes down, before every card this agent plays, what the
    position looked like and where the hidden cards actually were.

    A mixin rather than a subclass of one agent because the *labels* do not
    depend on who is playing - the engine knows the hands whoever is at the
    table. What the playing agent decides is the *distribution of positions*
    the corpus covers, and that is a reason to collect with a strong agent
    rather than a reason to tie the recorder to one.

    Mix it in before the agent, as with `HeuristicDeclarer`:

        class RecordingPolicyAgent(BeliefRecorder, PolicyAgent): ...

    It derives from `Agent` only so that the `super()` calls below resolve
    statically - `Agent` lands last in the MRO either way, after whichever real
    agent it is mixed with, so nothing about the dispatch changes.
    """

    def __init__(self, *args, recorder=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    def new_round_event(self) -> None:
        super().new_round_event()
        self.recorder.new_round()

    def choose_card(self, valid_moves):
        assert self.state is not None and self.state.public_state is not None
        round_ = self.state.public_state
        seat = self.state.seat_id
        # Taken before the card leaves the hand, so the observation is one the
        # agent could actually be holding when it asks the question.
        self.recorder.add(
            state=belief_input(round_, seat),
            locations=observer_view(round_, seat),
        )
        return super().choose_card(valid_moves)
