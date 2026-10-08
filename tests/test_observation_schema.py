"""Equal widths do not make pre-fix, privileged observations compatible."""

import numpy as np
import pytest
import torch

from skat.agents.belief.checkpoint import load_belief, save_belief
from skat.agents.belief.samples import BeliefShardWriter, load_belief_shards
from skat.agents.distill.checkpoint import load_policy, save_policy
from skat.agents.distill.samples import ShardWriter, load_shards
from skat.hand import FULL_HAND
from skat.models.belief import BeliefNet
from skat.models.observation import OBSERVATION_SCHEMA
from skat.models.policy import PolicyNet


@pytest.mark.parametrize("kind", ["policy", "belief"])
@pytest.mark.parametrize("schema", [None, "future-schema"])
def test_training_refuses_unmarked_or_different_schema_shards(tmp_path, kind, schema):
    if kind == "policy":
        writer = ShardWriter(str(tmp_path), input_size=240)
        writer.new_round()
        move = FULL_HAND[0]
        writer.add(np.zeros(240), {move: [1, 1]}, [move], move)
        reader = load_shards
    else:
        writer = BeliefShardWriter(str(tmp_path))
        writer.new_round()
        writer.add(np.zeros(304), np.zeros(32, dtype=np.int8))
        reader = load_belief_shards
    writer.close()
    path = next(tmp_path.glob("*.npz"))
    with np.load(path) as shard:
        arrays = {key: shard[key] for key in shard.files if key != "observation_schema"}
        assert str(shard["observation_schema"]) == OBSERVATION_SCHEMA
    if schema is not None:
        arrays["observation_schema"] = np.asarray(schema)
    np.savez_compressed(path, **arrays)
    with pytest.raises(RuntimeError, match="hidden skat"):
        reader(str(tmp_path))


@pytest.mark.parametrize("kind", ["policy", "belief"])
def test_checkpoints_mark_schema_and_warn_when_loading_legacy(tmp_path, kind):
    save, load, model = (
        (save_policy, load_policy, PolicyNet())
        if kind == "policy"
        else (save_belief, load_belief, BeliefNet())
    )
    path = str(tmp_path / "model.pt")
    save(path, model)
    blob = torch.load(path, weights_only=False)
    assert blob.pop("observation_schema") == OBSERVATION_SCHEMA
    torch.save(blob, path)
    with pytest.warns(RuntimeWarning, match="hidden skat"):
        load(path, torch.device("cpu"))
    blob["observation_schema"] = "future-schema"
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="observation schema"):
        load(path, torch.device("cpu"))
