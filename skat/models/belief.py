import torch
from torch import nn

from skat.models.card_transformer import CardTransformer

#: Where an unseen card can be, from the observer's chair. The order matches
#: the egocentric seat order everywhere else in the codebase (`seats_from`),
#: with the skat appended: a card is held by the player who moves after me, by
#: the one who moves before me, or it is lying in the skat.
NEXT, PREVIOUS, SKAT = 0, 1, 2
LOCATIONS = 3

#: Two opponents, one 32-wide "cannot hold this card" mask each. Appended to
#: `Round.get_state` to make the belief input; see `BeliefNet`.
VOID_BLOCK = 2 * 32


class BeliefNet(CardTransformer):
    """
    Where are the cards I cannot see?

    Skat is an imperfect-information game, and `MCTSAgent` handles that by
    sampling complete deals consistent with what it has seen - but it samples
    them *uniformly*. Every consistent world is equally likely, which is wrong
    in a way that costs simulations: a soloist who declared Grand is far more
    likely to be holding jacks than not, and half the sampled worlds where they
    hold none are worlds nobody would ever reach.

    This network estimates the distribution the search should be sampling from.
    For each of the 32 cards it returns three logits - next player, previous
    player, skat - to be read under a mask of the cards that are genuinely
    unseen. Cards in the observer's own hand, or already played, are known and
    contribute nothing.

    It reuses `CardTransformer`'s tokeniser and encoder wholesale, and for the
    same reason they were built: the observation decomposes exactly into 32
    per-card tokens, and "where is this card" is a per-card question with a
    per-card answer.

    **Its input is wider than the observation.** `Round.get_state` reports the
    played cards as one 32-hot set per seat, ORed over all ten tricks, so which
    cards fell *together* is gone - and with it any way to reconstruct that a
    seat failed to follow suit. Void inference is the single strongest read in
    Skat and the observation structurally cannot express it, so the two
    opponents' void masks are computed with the search's own
    `search.determinize.infer_voids` and appended: `INPUT_SIZE` is
    `OBSERVATION_SIZE + VOID_BLOCK`. They arrive as two extra per-card token
    columns, which is exactly where a per-card fact belongs.

    Worth being clear about what that buys, since `deal_unseen` *already*
    treats voids as hard constraints and would never place a heart in a hand
    shown void in hearts. The network is not being asked to enforce them. It is
    being asked to learn their consequences - that a player void in two suits
    is holding more trumps than their share, and that the cards they discarded
    say something about what they kept.

    What it still cannot see: the winning bid, which is not in the observation
    either. That is the strongest remaining clue about the soloist's hand and
    the obvious next thing to add.
    """

    LOCATIONS = LOCATIONS
    VOID_BLOCK = VOID_BLOCK
    INPUT_SIZE = CardTransformer.OBSERVATION_SIZE + VOID_BLOCK

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Two more per-card columns than the policy tokeniser produces.
        self.token = nn.Linear(self.TOKEN_FEATURES[self.features] + 2, self.d_model)
        # Replaces the one-output policy head.
        self.head = nn.Linear(self.d_model, LOCATIONS)

    def features_of(self, x):
        """Split the observation from the void block and tokenise both."""
        if x.shape[-1] != self.INPUT_SIZE:
            raise RuntimeError(
                f"expected {self.INPUT_SIZE} features "
                f"({self.OBSERVATION_SIZE} observation + {VOID_BLOCK} void), "
                f"got {x.shape[-1]}"
            )
        observation = x[:, : self.OBSERVATION_SIZE]
        tokens, context = super().features_of(observation)
        voids = x[:, self.OBSERVATION_SIZE :].view(-1, 2, self.CARDS).transpose(1, 2)
        return torch.cat([tokens, voids], dim=-1), context

    def forward(self, x):
        """`(B, 304)` in, `(B, 32, 3)` location logits out."""
        return self.head(self.trunk(x))
