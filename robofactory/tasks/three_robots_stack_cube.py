from typing import Any, Dict, Tuple

import os.path as osp
import numpy as np
import sapien
import torch
import math
import yaml
from transforms3d.euler import euler2quat
import copy

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
# from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import robofactory.utils.scenes as scene_rf
from robofactory import CONFIG_DIR
from robofactory.utils.nested_dict_utils import nested_yaml_map, replace_dir


@register_env("ThreeRobotsStackCube-rf", max_episode_steps=800)
class ThreeRobotsStackCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda", "panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda, Panda]]

    goal_radius = 0.12
    # SUSTAINED-stack success: the geometric stack + placement + release must HOLD for
    # this many CONSECUTIVE frames before success fires. Replaces the velocity is_static
    # gate (which spuriously failed on persistent contact micro-drift). A tower that
    # collapses right after release resets the counter -> never a success. Matches the
    # lift_barrier HOLD_FRAMES_K convention.
    STACK_HOLD_K = 8

    def __init__(
        self, *args, robot_uids=("panda", "panda", "panda"), robot_init_qpos_noise=0.02, **kwargs
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
            with open(osp.join(CONFIG_DIR, scene, 'three_robots_stack_cube.yaml'), 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = nested_yaml_map(replace_dir, cfg)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs =[]
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
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs =[]
        for render in render_cfg:
            pose = render['pose']
            if pose['type'] == 'pose':
                render['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        init_poses = []
        for agent_cfg in cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        scene_name = cfg['scene']['name']
        scene_builder = getattr(scene_rf, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
        # Per-episode intended stack ORDER (bottom -> mid -> top), as cube indices
        # 0=cubeA, 1=cubeB, 2=cubeC. Default (0,1,2) = canonical A-B-C, so non-reorder
        # episodes are byte-identical to the original success. The data-gen runner (and
        # eventually the eval harness) OVERRIDES self.intended_order AFTER reset to drive
        # the reorder_stack variant; evaluate() reads it (lazy-init fallback below).
        self.intended_order = (0, 1, 2)
        # reset the sustained-stack hold counter for the (re)initialised envs.
        if getattr(self, "_stack_hold", None) is None:
            self._stack_hold = torch.zeros(self.num_envs, dtype=torch.long)
        self._stack_hold[env_idx] = 0


    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def middle_agent(self) -> Panda:
        return self.agent.agents[2]
    
    def evaluate(self):
        # PARAMETERIZED by the per-episode intended stack order (bottom -> mid -> top),
        # cube indices 0=A,1=B,2=C. Default (0,1,2) = canonical A-B-C (so non-reorder
        # episodes are byte-identical to the original criterion). Cube i is grasped by
        # agent i (A<->left/0, B<->right/1, C<->middle/2).
        order = tuple(int(x) for x in getattr(self, "intended_order", (0, 1, 2)))
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        agents = [self.left_agent, self.right_agent, self.middle_agent]
        bottom, mid, top = cubes[order[0]], cubes[order[1]], cubes[order[2]]

        def _on(upper, lower):
            off = upper.pose.p - lower.pose.p
            xy_flag = (
                torch.linalg.norm(off[..., :2], axis=1)
                <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(off[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
            return torch.logical_and(xy_flag, z_flag)

        # mid sits on bottom, top sits on mid (the commanded order)
        is_mid_on_bottom = _on(mid, bottom)
        is_top_on_mid = _on(top, mid)
        # the two upper cubes are over the goal region (the bottom is placed there first)
        mid_placed = torch.linalg.norm(
            mid.pose.p[:, :2] - self.goal_region.pose.p[..., :2], axis=1
        ) < self.goal_radius
        top_placed = torch.linalg.norm(
            top.pose.p[:, :2] - self.goal_region.pose.p[..., :2], axis=1
        ) < self.goal_radius
        # every cube released by its grasping agent
        grasped = [agents[k].is_grasping(cubes[k]) for k in range(3)]
        # SUSTAINED-stack success (replaces the velocity is_static gate): the per-frame
        # geometric condition C = stack-built + both-upper-placed + all-released must
        # hold for STACK_HOLD_K CONSECUTIVE frames. A tower that collapses right after
        # release breaks C -> the counter resets -> never a success; a standing tower
        # holds C (the persistent uniform contact drift never breaks the RELATIVE 5mm
        # geometry) and passes. No velocity threshold. Mirrors lift_barrier's _lift_hold.
        cond = (
            is_top_on_mid & is_mid_on_bottom & mid_placed & top_placed
            & (~grasped[0]) & (~grasped[1]) & (~grasped[2])
        ).bool()
        # lazy-init the counter (evaluate can fire before _initialize_episode in some
        # harness paths); match cond's batch shape / device.
        if getattr(self, "_stack_hold", None) is None:
            self._stack_hold = torch.zeros_like(cond, dtype=torch.long)
        hold = self._stack_hold.to(cond.device)
        hold = torch.where(cond, hold + 1, torch.zeros_like(hold))
        self._stack_hold = hold
        success = (hold >= self.STACK_HOLD_K)
        return {
            "is_cubeA_grasped": grasped[0],
            "is_cubeB_grasped": grasped[1],
            "is_cubeC_grasped": grasped[2],
            "is_mid_on_bottom": is_mid_on_bottom,
            "is_top_on_mid": is_top_on_mid,
            "stack_hold": hold,
            "intended_order": order,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
