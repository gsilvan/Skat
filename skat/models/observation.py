"""Versioned, egocentric public-information observation."""

OBSERVATION_SCHEMA = "skat-public-contract-v3"
INPUT_SIZE = (
    376  # 240 prefix + 5 contract + 96 exposed cards + 3 trick counts + 32 known skat
)
OUTPUT_SIZE = 32


def validate_observation_schema(schema, source):
    if schema != OBSERVATION_SCHEMA:
        raise RuntimeError(
            f"{source}: incompatible observation schema; recollect and retrain for {OBSERVATION_SCHEMA}"
        )
