from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_tracker import ControlledAMTTracker  # noqa: E402
from fishmambatrack.data.samplers import RandomIdentitySampler  # noqa: E402
from fishmambatrack.tracking.amt_tracker import (  # noqa: E402
    AMTTracker,
    AMTTrackerConfig,
    Detection,
    Track,
)


class _Item:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class _Dataset:
    def __init__(self) -> None:
        self.items = [_Item(0) for _ in range(7)] + [_Item(1) for _ in range(2)]


class ReleaseSemanticsTests(unittest.TestCase):
    def test_crowd_count_one_is_off_sentinel(self) -> None:
        matrix = np.asarray([[0.9, 0.8]], dtype=np.float32)
        off = AMTTracker(AMTTrackerConfig(crowd_count_th=1))
        on = AMTTracker(AMTTrackerConfig(crowd_count_th=2))
        self.assertFalse(bool(off._crowd_mask(matrix)[0]))
        self.assertTrue(bool(on._crowd_mask(matrix)[0]))

    def test_crowd_damping_control_retains_fifo_freeze_mask(self) -> None:
        tracker = ControlledAMTTracker(
            AMTTrackerConfig(crowd_count_th=2),
            {
                "association": {
                    "crowd_appearance_suppression": False,
                    "geometry_confidence_scaling": False,
                },
                "write_policy": "reliability_first",
            },
        )
        matrix = np.asarray([[0.9, 0.8]], dtype=np.float32)
        self.assertTrue(bool(tracker._crowd_mask(matrix)[0]))
        self.assertAlmostEqual(
            float(tracker._appearance_weights(matrix, 1.25, "high")[0]), 1.25
        )

    def test_disabled_appearance_never_writes_identity_memory(self) -> None:
        tracker = ControlledAMTTracker(
            AMTTrackerConfig(),
            {
                "association": {"appearance": False, "geometry": "iou"},
                "write_policy": "disabled",
            },
        )
        detection = Detection(
            tlwh=np.asarray([0, 0, 10, 10], dtype=np.float32), score=0.9
        )
        track = Track(
            track_id=1,
            tlwh=np.asarray([0, 0, 10, 10], dtype=np.float32),
            score=0.9,
        )
        self.assertIsNone(tracker._similarity([detection], [track]))
        self.assertFalse(
            tracker._write_decision(
                stage="stage1",
                detection=detection,
                track=track,
                geometry=1.0,
                crowded=False,
            )
        )

    def test_sampler_length_matches_emitted_indices(self) -> None:
        sampler = RandomIdentitySampler(
            _Dataset(), num_instances=2, batch_size=4, seed=7
        )
        self.assertEqual(len(sampler), len(list(iter(sampler))))
        sampler.set_epoch(3)
        self.assertEqual(len(sampler), len(list(iter(sampler))))

    def test_multiseed_configs_select_seed_specific_checkpoints(self) -> None:
        for family in ("lengths", "encoders"):
            for path in (ROOT / "configs" / "experiments" / family).glob("*.yaml"):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["seeds"], [0, 1, 2])
                self.assertIn("{seed}", config["checkpoint"])

    def test_leave_one_out_controls_match_manuscript_definitions(self) -> None:
        root = ROOT / "configs" / "experiments" / "leave_one_out"
        temporal = yaml.safe_load(
            (root / "w_o_temporal_appearance.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(temporal["representation"], "appearance_disabled")
        self.assertFalse(temporal["association"]["appearance"])
        self.assertEqual(temporal["association"]["geometry"], "iou")
        crowd = yaml.safe_load(
            (root / "w_o_crowd_suppression.yaml").read_text(encoding="utf-8")
        )
        self.assertFalse(crowd["association"]["crowd_appearance_suppression"])
        self.assertEqual(crowd["write_policy"], "reliability_first")


if __name__ == "__main__":
    unittest.main()
