from typing import Any, Dict, Tuple

import os.path as osp
import numpy as np
import sapien
import torch
import json
import yaml
import copy

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
# from mani_skill.utils.scene_builder.table import TableSceneBuilder
# from robofactory.utils.scenes import TableSceneBuilder, RobocasaSceneBuilder
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.utils.structs.pose import Pose
import robofactory.utils.scenes as scene_rf
from robofactory import CONFIG_DIR
from robofactory.utils.nested_dict_utils import nested_yaml_map, replace_dir

@register_env("LiftBarrier-rf", max_episode_steps=500)
class LiftBarrierEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    goal_thresh = 0.025
    cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [1]))
    light_cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [0.5]))
    cube_half_size = 0.02

    def __init__(
        self, *args, robot_uids=("panda", "panda",), robot_init_qpos_noise=0.02, **kwargs
    ):
        if 'config' in kwargs:
            with open(kwargs['config'], 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            del kwargs['config']
        else:
            if 'scene' in kwargs:
                scene = kwargs['scene']
                del kwargs['scene']
            else:
                scene = 'table'
            with open(osp.join(CONFIG_DIR, scene, 'lift_barrier.yaml'), 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = nested_yaml_map(replace_dir, cfg)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', {})
        all_camera_configs =[]
        if sensor_cfg:
            for sensor in sensor_cfg:
                pose = sensor['pose']
                if pose['type'] == 'pose':
                    sensor['pose'] = sapien.Pose(*pose['params'])
                elif pose['type'] == 'look_at':
                    sensor['pose'] = sapien_utils.look_at(*pose['params'])
                all_camera_configs.append(CameraConfig(**sensor))
        return all_camera_configs


    @property
    def _default_human_render_camera_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', {})
        all_camera_configs =[]
        if render_cfg:
            for render in render_cfg:
                pose = render['pose']
                if pose['type'] == 'pose':
                    render['pose'] = sapien.Pose(*pose['params'])
                elif pose['type'] == 'look_at':
                    render['pose'] = sapien_utils.look_at(*pose['params'])
                all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=20,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )
    
    def _load_agent(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        init_poses = []
        for agent_cfg in cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        scene_name = cfg['scene']['name']
        scene_builder = getattr(scene_rf, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            # Per-env consecutive-hold counter for the SUSTAINED success criterion
            # (see evaluate). Create on first reset (full num_envs), then reset ONLY
            # the env_idx rows so a partial reset (some envs mid-episode) does not wipe
            # the others' progress. env_idx is a 1-D long tensor of the envs being
            # (re)initialized this call.
            if getattr(self, "_lift_hold", None) is None:
                self._lift_hold = torch.zeros(self.num_envs, dtype=torch.long)
            self._lift_hold[env_idx] = 0


    def evaluate(self):
        # SUSTAINED GEOMETRIC success (replaces the grasp-blind centre-height check AND
        # the fragile is_grasping contact gate): a PER-FRAME condition C must hold for
        # HOLD_FRAMES_K CONSECUTIVE frames. C = BOTH grasp ENDS of the barrier clear
        # base_z + 0.25  AND  BOTH arms' grippers are closed (each arm's 2 finger joints
        # sum < GRIPPER_CLOSE_MAX). The grasp ends are
        # barrier.pose.p + R(barrier.pose.q) @ [+/-0.222, 0, 0.074] (long axis =
        # local-X; offsets verified against model_data.json + the annotation grasp
        # pipeline). is_grasping is DROPPED: the contact-force probe UNRELIABLY
        # registers a real load-bearing grasp on the thin (~0.09 m) bar ends (one arm
        # read False for 25 consecutive frames on a genuine clean held lift). The
        # sustained counter kills transient flings (single-frame z-teleports) and tips
        # (don't sustain BOTH ends). Shared with success_candidates so the two stay in
        # lock-step. Runs on the env's BATCHED GPU torch tensors (batch dim kept).
        from robofactory.tasks.success_candidates import (
            barrier_grasp_ends_world,
            arm_gripper_closed,
            BARRIER_LIFT_DZ,
            GRIPPER_CLOSE_MAX,  # noqa: F401  (documents the gripper-closed threshold)
            HOLD_FRAMES_K,
        )
        ends = barrier_grasp_ends_world(self.barrier)            # (num_envs, 2, 3)
        base_z = self.agent.agents[0].robot.pose.p[0, 2]
        ends_z = ends[..., 2]                                    # (num_envs, 2)
        height_ok = (ends_z > base_z + BARRIER_LIFT_DZ).all(dim=-1)  # (num_envs,)
        closed0 = arm_gripper_closed(self.agent.agents[0])       # (num_envs,)
        closed1 = arm_gripper_closed(self.agent.agents[1])       # (num_envs,)
        c = (height_ok & closed0 & closed1).bool()               # (num_envs,) per-frame
        # lazy-init the counter (evaluate can fire before _initialize_episode in some
        # harness paths); match c's batch shape / device.
        if getattr(self, "_lift_hold", None) is None:
            self._lift_hold = torch.zeros_like(c, dtype=torch.long)
        hold = self._lift_hold.to(c.device)
        # increment where C is True this frame, RESET to 0 where False.
        hold = torch.where(c, hold + 1, torch.zeros_like(hold))
        self._lift_hold = hold
        success = (hold >= HOLD_FRAMES_K)
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0