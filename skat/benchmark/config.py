from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    arch: str
    target: str = "visits"
    features: str = "full"


def policy_variants(include_controls: bool = False) -> list[PolicyVariant]:
    """Policy models to train from the shared, fixed MCTS corpus."""
    primary = [
        PolicyVariant("policy-mlp", "mlp"),
        PolicyVariant("policy-transformer", "transformer"),
    ]
    if not include_controls:
        return primary
    return [
        *primary,
        PolicyVariant("policy-mlp-value", "mlp", target="value"),
        PolicyVariant("policy-transformer-value", "transformer", target="value"),
        PolicyVariant("policy-transformer-raw", "transformer", features="raw"),
        PolicyVariant(
            "policy-transformer-raw-value",
            "transformer",
            target="value",
            features="raw",
        ),
    ]


@dataclass
class BenchmarkConfig:
    profile: str
    seed: int = 20260829
    device: str = "cpu"
    deterministic: bool = True
    include_controls: bool = True

    dqn_rounds: int = 2_000
    dqn_batch_size: int = 256
    dqn_learn_start: int = 10_000
    dqn_buffer_size: int = 100_000

    distill_rounds: int = 650
    distill_simulations: int = 600
    distill_determinizations: int = 20
    policy_epochs: int | None = 10

    belief_rounds: int = 2_000
    belief_epochs: int = 10

    bidding_deals: int = 650
    bidding_simulations: int = 600
    bidding_determinizations: int = 20
    bidding_epochs: int = 30
    bidding_threshold: float = 0.5
    bidding_aggressive_threshold: float = 0.35
    bidding_defensive_threshold: float = 0.65

    evaluation_rounds: int = 200
    evaluation_simulations: int = 600
    evaluation_determinizations: int = 20

    @classmethod
    def from_profile(cls, name: str, **overrides: Any) -> BenchmarkConfig:
        profiles: dict[str, dict[str, Any]] = {
            "smoke": {
                "dqn_rounds": 4,
                "dqn_batch_size": 8,
                "dqn_learn_start": 16,
                "dqn_buffer_size": 128,
                "distill_rounds": 6,
                "distill_simulations": 4,
                "distill_determinizations": 1,
                "policy_epochs": 1,
                "belief_rounds": 6,
                "belief_epochs": 1,
                "bidding_deals": 4,
                "bidding_simulations": 4,
                "bidding_determinizations": 1,
                "bidding_epochs": 1,
                "evaluation_rounds": 2,
                "evaluation_simulations": 4,
                "evaluation_determinizations": 1,
            },
            "standard": {},
            "full": {
                "dqn_rounds": 20_000,
                "distill_rounds": 5_200,
                "distill_simulations": 2_000,
                "policy_epochs": None,
                "belief_rounds": 10_200,
                "belief_epochs": 20,
                "bidding_deals": 5_200,
                "bidding_simulations": 2_000,
                "bidding_epochs": 40,
                "evaluation_rounds": 1_000,
            },
        }
        if name not in profiles:
            raise ValueError(
                f"unknown benchmark profile {name!r}; known: "
                f"{', '.join(sorted(profiles))}"
            )
        values = {**profiles[name], **overrides, "profile": name}
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "dqn_rounds": self.dqn_rounds,
            "dqn_batch_size": self.dqn_batch_size,
            "dqn_learn_start": self.dqn_learn_start,
            "distill_rounds": self.distill_rounds,
            "distill_simulations": self.distill_simulations,
            "distill_determinizations": self.distill_determinizations,
            "belief_rounds": self.belief_rounds,
            "belief_epochs": self.belief_epochs,
            "bidding_deals": self.bidding_deals,
            "bidding_simulations": self.bidding_simulations,
            "bidding_determinizations": self.bidding_determinizations,
            "bidding_epochs": self.bidding_epochs,
            "evaluation_rounds": self.evaluation_rounds,
            "evaluation_simulations": self.evaluation_simulations,
            "evaluation_determinizations": self.evaluation_determinizations,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"benchmark values must be positive: {', '.join(invalid)}")
        if self.dqn_learn_start < self.dqn_batch_size:
            raise ValueError("dqn_learn_start must be at least dqn_batch_size")
        if self.policy_epochs is not None and self.policy_epochs <= 0:
            raise ValueError("policy_epochs must be positive or None")
        thresholds = (
            self.bidding_aggressive_threshold,
            self.bidding_threshold,
            self.bidding_defensive_threshold,
        )
        if not all(0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("bidding thresholds must be between 0 and 1")
        if thresholds != tuple(sorted(thresholds)):
            raise ValueError(
                "bidding thresholds must order aggressive <= balanced <= defensive"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
