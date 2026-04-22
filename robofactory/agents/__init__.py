from . import panda_wristcam_multi  # noqa: F401

# Patch MultiAgent to expose its aggregated sensor_configs via the `_sensor_configs`
# property that ManiSkill's BaseEnv reads. Without this, per-agent wrist cameras
# (registered in each BaseAgent's `_sensor_configs`) are silently ignored in
# multi-arm tasks because MultiAgent inherits BaseAgent's empty default.
from mani_skill.agents.multi_agent import MultiAgent as _MultiAgent

def _multi_agent_sensor_configs(self):
    return list(getattr(self, "sensor_configs", []))

_MultiAgent._sensor_configs = property(_multi_agent_sensor_configs)

# ManiSkill's `_setup_sensors` reads `self.agent.robot` when any agent sensor
# is registered, to pass articulation context for state indexing. MultiAgent
# has many robots, not one. For mount-based cameras (the wrist cams), the mount
# link references the correct articulation directly, so the passed articulation
# is only used for bookkeeping — returning agents[0].robot is safe.
def _multi_agent_robot(self):
    return self.agents[0].robot

_MultiAgent.robot = property(_multi_agent_robot)

# RoboFactory's motion planners, YAML configs, and stored demos all hardcode
# action dict keys as `panda-{i}` regardless of the underlying robot class.
# MultiAgent's default key is `{agent.uid}-{i}` which breaks when we swap in
# PandaWristCamMulti (uid=panda_wristcam_multi). Wrap __init__ to force the
# `panda-{i}` key scheme so the existing planners + action replay still work.
_orig_multi_agent_init = _MultiAgent.__init__
def _multi_agent_init_with_panda_keys(self, agents):
    _orig_multi_agent_init(self, agents)
    new_dict = {}
    for i, agent in enumerate(agents):
        new_dict[f"panda-{i}"] = agent
    self.agents_dict = new_dict

_MultiAgent.__init__ = _multi_agent_init_with_panda_keys



