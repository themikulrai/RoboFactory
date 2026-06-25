from typing import Any, Dict, Tuple

import os.path as osp
import numpy as np
import sapien
import torch
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
# from robofactory.utils.scenes import TableSceneBuilder, RobocasaSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import robofactory.utils.scenes as scene_rf
from robofactory import CONFIG_DIR
from robofactory.utils.nested_dict_utils import nested_yaml_map, replace_dir

# max_episode_steps bumped 500 -> 600 so the wait_hold causal variant (a single
# ~200-tick mid-place freeze injected on top of the ~300-tick natural stack) does NOT
# truncate before the stack completes + the 8-frame sustained-hold success fires. The
# non-wait variants auto-terminate on success ~300 steps, so the larger budget is a
# no-op for them. (TSC uses 800; 2SC's shorter episodes only need 600.)
@register_env("TwoRobotsStackCube-rf", max_episode_steps=600)
class TwoRobotsStackCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    goal_radius = 0.11
    # SUSTAINED-stack success: the geometric stack + placement + release must HOLD for
    # this many CONSECUTIVE frames before success fires. Replaces the instantaneous
    # criterion (which can blip on a tower that collapses right after release, and which
    # the velocity is_static gate spuriously failed on persistent contact micro-drift).
    # A tower that collapses right after release resets the counter -> never a success.
    # Matches the TSC STACK_HOLD_K / lift_barrier HOLD_FRAMES_K convention.
    STACK_HOLD_K = 8

    def __init__(
        self, *args, robot_uids=("panda", "panda"), robot_init_qpos_noise=0.02, **kwargs
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
            with open(osp.join(CONFIG_DIR, scene, 'two_robots_stack_cube.yaml'), 'r', encoding='utf-8') as f:
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
        for agent_cfg in cfg["agents"]:
            init_poses.append(sapien.Pose(p=agent_cfg["pos"]["ppos"]["p"]))
        super()._load_agent(options, init_poses)


    def _load_scene(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        scene_name = cfg["scene"]["name"]
        scene_builder = getattr(scene_rf, f"{scene_name}SceneBuilder")
        self.scene_builder = scene_builder(env=self, cfg=cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
        # Per-episode intended stack ORDER (bottom -> top), as cube indices 0=cubeA,
        # 1=cubeB. Default (0,1) = canonical A-B, so non-reorder episodes are
        # byte-identical to the original success. The data-gen runner (and eventually
        # the eval harness) OVERRIDES self.intended_order AFTER reset to drive the
        # reorder_stack variant; evaluate() reads it (lazy-init fallback below).
        self.intended_order = (0, 1)
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

    def evaluate(self):
        # PARAMETERIZED by the per-episode intended stack order (bottom -> top), cube
        # indices 0=cubeA, 1=cubeB. Default (0,1) = canonical A-B (so non-reorder
        # episodes are byte-identical to the original criterion). Cube i is grasped by
        # agent i (A<->left/0, B<->right/1).
        order = tuple(int(x) for x in getattr(self, "intended_order", (0, 1)))
        cubes = [self.cubeA, self.cubeB]
        agents = [self.left_agent, self.right_agent]
        bottom, top = cubes[order[0]], cubes[order[1]]

        # top sits on bottom (the commanded order)
        offset = top.pose.p - bottom.pose.p
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_top_on_bottom = torch.logical_and(xy_flag, z_flag)
        # the bottom cube is placed in the goal region
        bottom_placed = torch.linalg.norm(
            bottom.pose.p[:, :2] - self.goal_region.pose.p[..., :2], axis=1
        ) < self.goal_radius
        # every cube released by its grasping agent
        grasped = [agents[k].is_grasping(cubes[k]) for k in range(2)]
        # SUSTAINED-stack success (replaces the instantaneous criterion): the per-frame
        # geometric condition C = top-on-bottom + bottom-placed + both-released must hold
        # for STACK_HOLD_K CONSECUTIVE frames. A tower that collapses right after release
        # breaks C -> the counter resets -> never a success; a standing tower holds C
        # (the persistent uniform contact drift never breaks the RELATIVE 5mm geometry)
        # and passes. No velocity threshold. Mirrors TSC / lift_barrier.
        cond = (
            is_top_on_bottom & bottom_placed & (~grasped[0]) & (~grasped[1])
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
            "is_top_on_bottom": is_top_on_bottom,
            "bottom_placed": bottom_placed,
            "stack_hold": hold,
            "intended_order": order,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                goal_region_pos=self.goal_region.pose.p,
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                left_arm_tcp_to_cubeA_pos=self.cubeA.pose.p
                - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_cubeB_pos=self.cubeB.pose.p
                - self.right_agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
