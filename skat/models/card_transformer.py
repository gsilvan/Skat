from typing import ClassVar

import torch
from torch import nn

from skat.models.observation import INPUT_SIZE, OUTPUT_SIZE


class CardTransformer(nn.Module):
    """
    Card-play policy as self-attention over the pack: 32 tokens, one per card,
    240 observation floats in and 32 logits out.

    Same contract as `PolicyNet` - `(B, 240) -> (B, 32)`, read under a
    legal-move mask - and a drop-in alternative to it. The difference is what
    the network is handed. `PolicyNet` sees a flat vector and has to discover
    thirty-two separate times that input slot 3 and output slot 3 are the same
    object, and has to learn "an ace is high" once per suit block.
    `Round.get_state` is egocentric and its per-card blocks are disjoint, so
    that structure is recoverable exactly: each card becomes a token carrying
    its own status, and the head is one shared `Linear(d_model, 1)` per token.
    Half a million parameters against three million, with the symmetry built in
    rather than learned - which is the trade that matters on a corpus this
    small.

    The tokenisation happens inside `forward`, deliberately: every caller
    (`PolicyAgent.select`, `distill.train`) passes a flat `(B, 240)` batch and
    knows nothing about architectures, and that is what lets one network be
    swapped for the other by checkpoint alone.

    There is no positional encoding. Cards are a set, and the suit/rank
    embedding *is* the position. There is no CLS token either - with 32 tokens
    and full attention every token already sees every other one in the first
    layer. A CLS token is what a *value* head would want, if the PUCT prior of
    expert-iteration stage 3 ever needs one; that is the reason to add it, and
    the only one.
    """

    INPUT_SIZE = INPUT_SIZE
    OUTPUT_SIZE = OUTPUT_SIZE

    #: How much of the input is `Round.get_state`. The same as `INPUT_SIZE`
    #: here, and smaller than it in `skat.models.belief.BeliefNet`, which
    #: appends a block the observation does not contain.
    OBSERVATION_SIZE = INPUT_SIZE

    CARDS = 32

    # Where each 32-wide block in which a card can appear starts, in the order
    # `Round.get_state` concatenates them: the observer's hand, then the three
    # completed-trick blocks, then the three live-trick blocks, both egocentric
    # (me, next, previous). The blocks are disjoint - `Round.step` files a full
    # trick into the history and calls `new_trick()` together - so a card sets
    # at most one of these seven.
    #
    # Restated here rather than imported from `observation.py` or the tests. This
    # module is the second reader of that layout, and two independent
    # statements of it are what catch a change to one of them.
    STATUS_BLOCKS = (0, 44, 76, 108, 140, 172, 204)
    STATUS = len(STATUS_BLOCKS)

    GAME = slice(32, 38)  # trump suit one-hot, is_grand, is_null
    CURRENT = slice(140, 236)  # 3 x 32, the live trick
    FRONT_HAND = slice(236, 239)  # my own seat in the trick order
    CONTEXT_HEAD = slice(32, 44)  # game, role, soloist seat, points
    CONTEXT_TAIL = slice(236, 240)  # front hand, trick value

    #: ...plus `led_is_trump`, which is derived rather than sliced.
    CONTEXT = 17

    #: "raw" is the control: status and nothing else, to separate "attention
    #: over a card set helped" from "the derived features helped".
    TOKEN_FEATURES: ClassVar[dict[str, int]] = {
        "raw": STATUS + 1,
        "full": STATUS + 6,
    }

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        features: str = "full",
    ) -> None:
        super().__init__()
        if features not in self.TOKEN_FEATURES:
            raise ValueError(
                f"features={features!r}, expected one of {sorted(self.TOKEN_FEATURES)}"
            )
        self.features = features
        self.d_model = d_model

        cards = torch.arange(self.CARDS)
        # `persistent=False` keeps these out of the state dict. They are
        # constants rebuilt at construction, so storing them would mean adding
        # or renaming one retires every checkpoint that came before.
        self.register_buffer(
            "status_index",
            (torch.tensor(self.STATUS_BLOCKS).view(-1, 1) + cards.view(1, -1)).reshape(
                -1
            ),
            persistent=False,
        )
        self.register_buffer("suit_of", cards // 8, persistent=False)
        self.register_buffer("rank_of", cards % 8, persistent=False)
        self.register_buffer(
            "suit_onehot", torch.eye(4).repeat_interleave(8, 0), persistent=False
        )
        self.register_buffer("is_jack", (cards % 8 == 3).float(), persistent=False)

        self.token = nn.Linear(self.TOKEN_FEATURES[features], d_model)
        self.suit_embedding = nn.Embedding(4, d_model)
        self.rank_embedding = nn.Embedding(8, d_model)
        self.context = nn.Linear(self.CONTEXT, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers,
            norm=nn.LayerNorm(d_model),
            # Pre-norm makes the nested-tensor fast path a no-op, and leaving
            # it on warns on every construction.
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)

    # --------------------------------------------------------------- features

    def status(self, x: torch.Tensor) -> torch.Tensor:
        """The seven mutually exclusive per-card blocks, as `(B, 32, 7)`."""
        return (
            x[:, self.status_index]
            .view(x.shape[0], self.STATUS, self.CARDS)
            .transpose(1, 2)
        )

    def trump(self, game: torch.Tensor) -> torch.Tensor:
        """
        Which cards are trumps, `(B, 32)`, from the 6-wide game block.

        `Game.to_numpy` only sets a suit bit when the game *has* a trump suit,
        so grand and null arrive with an all-zero suit one-hot and need no
        special case beyond null having no trumps at all. The clamp is what
        stops the jack of the trump suit counting twice.
        """
        trump_suit = game[:, :4] @ self.suit_onehot.t()
        is_null = game[:, 5:6]
        return (1.0 - is_null) * (self.is_jack + trump_suit).clamp(max=1.0)

    def trick_order(self, x: torch.Tensor) -> torch.Tensor:
        """
        One-hot of "this card was played 1st / 2nd / 3rd in the live trick",
        `(B, 32, 3)`, all-zero for the cards not on the table.

        The observation attributes the live trick by *relative seat* but does
        not order it, and which card led decides who is currently winning -
        most of what card play is about. It is recoverable: `FRONT_HAND` gives
        my own position `p` in the trick, and relative seat `r` plays at
        position `(r + p) % 3`.
        """
        b = x.shape[0]
        current = x[:, self.CURRENT].view(b, 3, self.CARDS)
        position = x[:, self.FRONT_HAND].argmax(-1)
        seat = torch.arange(3, device=x.device).view(1, 3)
        slot = ((seat + position.view(b, 1)) % 3).unsqueeze(-1)
        order = torch.zeros(b, 3, 3, device=x.device).scatter_(2, slot, 1.0)
        return torch.einsum("brc,bro->bco", current, order)

    def features_of(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Both halves at once: the per-card table `(B, 32, F)` and the global
        context `(B, 17)`.

        They are built together because both need the live trick's order, and
        deriving it twice is a tenth of the batch-1 latency. `tokens` and
        `context_vector` are the readable single-purpose views onto this.
        """
        if x.shape[-1] != self.OBSERVATION_SIZE:
            # Every block below is a fixed index into `Round.get_state`, so a
            # vector of the wrong width would be silently mis-sliced rather
            # than merely wrong. Say so instead.
            raise RuntimeError(
                f"expected {self.OBSERVATION_SIZE} observation features, got "
                f"{x.shape[-1]}"
            )
        status = self.status(x)
        # A card in none of the seven blocks is in the skat or in an opponent's
        # hand, and the two are indistinguishable from here.
        unseen = 1.0 - status.sum(-1, keepdim=True)
        is_trump = self.trump(x[:, self.GAME])
        order = self.trick_order(x)
        led = order[:, :, 0]

        if self.features == "raw":
            tokens = torch.cat([status, unseen], dim=-1)
        else:
            # Printed suit of the led card, spread back over every card of it.
            # Not the same as "legal to play" - that is `Trick.forced_cards`'
            # job, and reimplementing it here would let the two drift apart.
            # What this buys is what it says about the *opponents*' choices;
            # the agent's own legal moves arrive as the mask.
            same_suit = (led @ self.suit_onehot) @ self.suit_onehot.t()
            tokens = torch.cat(
                [
                    status,
                    unseen,
                    is_trump.unsqueeze(-1),
                    order,
                    same_suit.unsqueeze(-1),
                ],
                dim=-1,
            )

        led_is_trump = (led * is_trump).sum(-1, keepdim=True)
        context = torch.cat(
            [x[:, self.CONTEXT_HEAD], x[:, self.CONTEXT_TAIL], led_is_trump], dim=-1
        )
        return tokens, context

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """The per-card feature table, `(B, 32, TOKEN_FEATURES[features])`."""
        return self.features_of(x)[0]

    def context_vector(self, x: torch.Tensor) -> torch.Tensor:
        """The 17 global scalars, `(B, 17)`."""
        return self.features_of(x)[1]

    # ---------------------------------------------------------------- forward

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """
        One encoded vector per card, `(B, 32, d_model)`.

        Split out from `forward` so a different head can sit on the same
        tokeniser and encoder. `skat.models.belief.BeliefNet` is the second
        user: same 32 cards, same features, but asking where each *unseen* card
        is rather than which one to play.
        """
        # Adding the projected context to every token is the same function as
        # concatenating the raw context onto every token's features, since
        # Linear([t; g]) = W_t t + W_g g + b. It is written this way because it
        # keeps the per-card and global halves legible, not because it differs.
        tokens, context = self.features_of(x)
        identity = self.suit_embedding(self.suit_of) + self.rank_embedding(self.rank_of)
        h = (
            self.token(tokens)
            + identity.unsqueeze(0)
            + self.context(context).unsqueeze(1)
        )
        return self.encoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(-1)
