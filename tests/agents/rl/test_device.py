"""
`--device` has to reach further than the model.

The observation (`Round.get_state_t`) and the legal-move mask
(`Hand.as_tensor_mask`) are built on the cpu, because that is where the replay
buffer wants them. If nothing moved them, the nets would sit on the gpu and the
first forward pass would fail on mismatched devices - a break that never shows
up while the default is cpu.
"""

import unittest
from unittest.mock import patch

import torch

from skat.agents.rl.buffer import ReplayBuffer
from skat.agents.rl.dqn import describe_device, gpu_backend, resolve_device
from skat.models import QNet


class GpuBackendTest(unittest.TestCase):
    """
    A ROCm build reports `version.cuda is None` but still answers to the device
    string "cuda", so "cuda" must never be read as "NVIDIA".
    """

    def test_rocm_is_reported_as_rocm(self) -> None:
        with patch.object(torch.version, "hip", "7.2.53211"):
            with patch.object(torch.version, "cuda", None):
                self.assertEqual("ROCm 7.2.53211", gpu_backend())

    def test_cuda_is_reported_as_cuda(self) -> None:
        with patch.object(torch.version, "hip", None):
            with patch.object(torch.version, "cuda", "12.4"):
                self.assertEqual("CUDA 12.4", gpu_backend())

    def test_hip_wins_when_both_are_somehow_set(self) -> None:
        with patch.object(torch.version, "hip", "7.2"):
            with patch.object(torch.version, "cuda", "12.4"):
                self.assertEqual("ROCm 7.2", gpu_backend())

    def test_a_cpu_build_says_so(self) -> None:
        with patch.object(torch.version, "hip", None):
            with patch.object(torch.version, "cuda", None):
                self.assertEqual("no GPU support", gpu_backend())


class DescribeDeviceTest(unittest.TestCase):
    def test_cpu_touches_no_cuda_api(self) -> None:
        # calling into torch.cuda on a cpu-only build is how this would break
        with patch.object(torch.cuda, "get_device_name") as name:
            self.assertEqual("cpu", describe_device(torch.device("cpu")))
        name.assert_not_called()

    def test_a_gpu_is_named(self) -> None:
        with patch.object(
            torch.cuda, "get_device_name", return_value="Radeon RX 5600 XT"
        ):
            with patch.object(torch.version, "hip", "7.2"):
                described = describe_device(torch.device("cuda"))
        self.assertIn("Radeon RX 5600 XT", described)
        self.assertIn("ROCm 7.2", described)


class ResolveDeviceTest(unittest.TestCase):
    def test_explicit_cpu(self) -> None:
        self.assertEqual(torch.device("cpu"), resolve_device("cpu"))

    def test_auto_falls_back_to_cpu_without_cuda(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(torch.device("cpu"), resolve_device("auto"))

    def test_auto_takes_cuda_when_available(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(torch.device("cuda"), resolve_device("auto"))

    def test_auto_is_the_default(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), resolve_device())

    def test_an_indexed_device_survives(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(torch.device("cuda:1"), resolve_device("cuda:1"))

    def test_a_gpu_request_is_refused_with_the_facts(self) -> None:
        import sys

        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as caught:
                resolve_device("cuda")
        message = str(caught.exception)
        self.assertIn(torch.__version__, message)  # which build is loaded
        self.assertIn(sys.executable, message)  # which interpreter it came from
        self.assertIn("ROCm", message)  # AMD is not an afterthought
        self.assertIn("CUDA", message)

    def test_the_refusal_does_not_prescribe_the_venv_as_the_fix(self) -> None:
        # training here runs outside the venv against a system ROCm build, so
        # "re-pin pyproject.toml" must be offered as one cause, not the answer
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as caught:
                resolve_device("cuda")
        self.assertIn("usually", str(caught.exception))

    def test_a_nonsense_device_still_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_device("gpu")


class DeviceReachesTheTensorsTest(unittest.TestCase):
    """
    Guards the coupling without needing a gpu, by putting the model on torch's
    "meta" device: both inputs have to arrive there before use.
    """

    def test_select_action_moves_state_and_mask(self) -> None:
        from skat.agents.rl import dqn as dqn_module

        dqn = dqn_module.DQN.__new__(dqn_module.DQN)  # no nets, no writer, no io
        # "meta" is a real torch device every machine can produce, so a missing
        # .to() shows up as a cpu tensor here instead of passing unnoticed.
        dqn.device = torch.device("meta")
        dqn.epsilon = 0.0
        dqn.epsilon_decay = 1.0
        dqn.epsilon_min = 0.0
        # past the warm-up, so select_action takes the greedy branch under test
        dqn.replay_buffer = ReplayBuffer(buffer_size=1)
        dqn.learn_start = 0

        seen: dict[str, torch.device] = {}

        class SpyNet:
            def __call__(self, state):
                seen["state"] = state.device
                return torch.zeros((1, 32))

        dqn.policy_net = SpyNet()

        state = torch.zeros((1, 171))
        mask = torch.zeros(32, dtype=torch.bool)
        mask[0] = True

        with patch.object(dqn_module, "MaskedCategorical") as masked:
            masked.return_value.probs.argmax.return_value = torch.tensor(0)
            dqn.select_action(state, mask, explore=False)

        self.assertEqual(torch.device("meta"), seen["state"])
        self.assertEqual(torch.device("meta"), masked.call_args.kwargs["mask"].device)

    def test_a_checkpoint_is_mapped_with_the_resolved_device(self) -> None:
        """
        Regression: torch.load was handed the raw --device string, so resuming
        with the default "auto" died on 'don't know how to restore data
        location ... tagged with auto'.
        """
        import tempfile
        from pathlib import Path

        from skat.agents.rl import dqn as dqn_module

        net = QNet()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(net.state_dict(), path)
            # would raise if "auto" reached torch.load
            dqn_module.DQN(device="auto", model_path=str(path))

    def test_a_stale_checkpoint_says_why_it_does_not_fit(self) -> None:
        """A layout change retires every checkpoint; say so, not 'size mismatch'."""
        import tempfile
        from pathlib import Path

        from torch import nn

        from skat.agents.rl import dqn as dqn_module

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            stale = nn.Sequential(nn.Linear(171, 32))  # the previous input width
            torch.save(stale.state_dict(), path)
            with self.assertRaises(RuntimeError) as caught:
                dqn_module.DQN(device="cpu", model_path=str(path))
        message = str(caught.exception)
        self.assertIn("does not fit", message)
        self.assertIn(str(QNet.INPUT_SIZE), message)

    def test_the_replay_buffer_keeps_cpu_tensors(self) -> None:
        # states are stored as produced; only DQN moves copies to the device
        from skat.agents.random import RandomAgent
        from skat.game import GamePhase, Round
        from skat.games.suit import SuitGame
        from skat.hand import FULL_HAND

        deal = [FULL_HAND[0:10], FULL_HAND[10:20], FULL_HAND[20:30], FULL_HAND[30:32]]
        round_ = Round(
            skip_bidding=True,
            solo_player_id=0,
            hand_game=True,
            declare_game=SuitGame(3),
            agents=[RandomAgent() for _ in range(3)],
            start=False,
            initial_cards=deal,
        )
        round_.deal()
        round_.phase = GamePhase.PLAYING
        self.assertEqual(torch.device("cpu"), round_.get_state_t(0).device)
        self.assertEqual(
            torch.device("cpu"), round_.player[0].hand.as_tensor_mask.device
        )


if __name__ == "__main__":
    unittest.main()
