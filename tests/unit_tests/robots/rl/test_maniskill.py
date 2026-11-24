# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract checks for the native ManiSkill episode adapter."""

from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robots.rl.maniskill import ManiSkillPrimitives
from robots.rl.tools import Measurement


def test_mani_skill_absolute_command_uses_the_articulation_frame():
    backend = ManiSkillPrimitives.__new__(ManiSkillPrimitives)
    root = Rotation.from_euler("z", 90, degrees=True)
    backend._base = np.eye(4)
    backend._base[:3, :3] = root.as_matrix()
    backend._base[:3, 3] = [0, 2, 0]
    backend._gripper_target = 0.6
    current = Measurement(np.array([0, 2, 0]), root.as_matrix(), 0.8)
    local = Rotation.from_euler("XYZ", [0.1, 0.2, 0.3])

    action, distance, angle = backend._pose_action(
        current, np.array([0, 2.1, 0]), (root * local).as_matrix()
    )

    np.testing.assert_allclose(action, [0.1, 0, 0, 0.1, 0.2, 0.3, 0.6], atol=1e-7)
    np.testing.assert_allclose([distance, angle], [0.1, local.magnitude()])


def test_mani_skill_native_terminal_flags_and_verdict_are_separate():
    backend = ManiSkillPrimitives.__new__(ManiSkillPrimitives)
    observation = {}
    backend.env = SimpleNamespace(
        step=lambda _: (
            observation,
            0.0,
            torch.tensor([True]),
            False,
            {"success": torch.tensor([False])},
        )
    )
    backend.env_steps = 0
    backend._max_episode_steps = 4
    backend._success = False
    backend._check_cancelled = lambda: None

    backend._step(np.zeros(7))

    assert backend.episode_flags() == {"terminated": True, "truncated": False}
    assert backend.ended and not backend.native_success()
    assert backend.env_steps == 1 and backend._obs is observation
