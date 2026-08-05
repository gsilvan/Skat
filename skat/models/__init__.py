from typing import Any

from .card_transformer import CardTransformer
from .policy import PolicyNet
from .suit_solo import SuitSoloNet  # noqa: F401

#: Policy architectures, by the name `--arch` takes and the name written into
#: every policy checkpoint. That makes these names permanent: rename one and
#: every file carrying it stops loading.
#:
#: Typed loosely because the two share no base beyond `nn.Module` - what makes
#: them interchangeable is the contract `(B, 240) -> (B, 32)` plus the
#: `INPUT_SIZE`/`OUTPUT_SIZE` class attributes, which is a duck type rather
#: than a hierarchy. `tests/models/test_card_transformer.py` is what checks it.
ARCHITECTURES: dict[str, Any] = {
    "mlp": PolicyNet,
    "transformer": CardTransformer,
}

#: What a checkpoint without an `arch` key means. The key was added once there
#: was a second architecture, so the files that predate it are all MLPs.
DEFAULT_ARCH = "mlp"
