"""Panda with wrist camera, with per-agent uid suffix to avoid sensor-uid collisions in multi-arm tasks.

The stock `PandaWristCam` hardcodes `uid="hand_camera"` for its sensor config.
When multiple arms are instantiated (robot_uids=("panda_wristcam",) * N), the
parsed sensor-config dict collides and only one camera actually renders.

This variant suffixes the uid with the agent index: hand_camera_0, hand_camera_1, ...
"""
import numpy as np
import sapien

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class PandaWristCamMulti(PandaWristCam):
    uid = "panda_wristcam_multi"

    @property
    def _sensor_configs(self):
        suffix = f"_{self._agent_idx}" if self._agent_idx is not None else ""
        return [
            CameraConfig(
                uid=f"hand_camera{suffix}",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=224,
                height=224,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
