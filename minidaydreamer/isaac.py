"""Isaac Lab adapter. Import after AppLauncher starts the Kit application."""

import random

import gymnasium as gym
import isaaclab_tasks  # noqa: F401 -- registers the task
from isaaclab_tasks.utils import parse_env_cfg
import numpy as np
import torch

from .stack import STEP


TASK_V2 = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
TASK_V3 = "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor"


def randomize_tabletop(env, env_ids, asset_cfg, textures, default_texture="", texture_rotation=(0.0, 0.0)):
    """Change the rendered tabletop's authored material, leaving its collider intact."""
    import omni.usd
    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    visuals = stage.GetPrimAtPath("/World/envs/env_0/Table/Visuals")
    if visuals.IsInstance():
        visuals.SetInstanceable(False)
    shader = stage.GetPrimAtPath(visuals.GetPath().AppendPath("Looks/table_base/Shader"))
    texture = shader.GetAttribute("inputs:diffuse_texture")
    texture.Set(Sdf.AssetPath(random.choice(textures)))


class IsaacStack:
    backend = "isaac"
    max_steps = 200
    block_height = .0468
    grip_delay = 7

    def __init__(self):
        task = TASK_V3 if TASK_V3 in gym.registry else TASK_V2
        options = {"overrides": ["physics=isaacsim_physx"]} if task == TASK_V3 else {}
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=1, **options)
        # The upstream lighting and material reset events use these flags in every split.
        cfg.eval_mode = True
        cfg.eval_type = "all"
        # These HDR domes saturate the white Franka at the upstream training intensities.
        cfg.events.randomize_light.params.update(intensity_range=(500.0, 1100.0), color_variation=0.15)
        cfg.events.randomize_robot_arm_visual_texture = None
        cfg.events.randomize_table_visual_material.func = randomize_tabletop
        # Keep the green cube as a distractor. The outer task scores red on blue.
        # Disable internal auto-resets so the final transition still contains its terminal scene.
        for term in ("success", "time_out", "cube_1_dropping", "cube_2_dropping", "cube_3_dropping"):
            setattr(cfg.terminations, term, None)
        cfg.scene.table_cam.width = 640
        cfg.scene.table_cam.height = 480
        self.env = gym.make(task, cfg=cfg)
        self.scale = cfg.actions.arm_action.scale
        self.frame_callback = None

    def close(self):
        self.env.close()

    def state(self, observation):
        p = observation["policy"]
        self.camera = p["table_cam"]
        tool = p["eef_pos"].reshape(1, 3)
        # One environment at the world origin. Express xyz relative to the table work area.
        blue, red, _ = p["cube_positions"].reshape(3, 3)
        xyz = torch.stack((tool[0], red, blue)) - tool.new_tensor([.5, 0, 0])
        self.aperture = p["gripper_pos"].abs().min().item()
        grasped = observation["subtask_terms"]["grasp_1"].reshape(1).float()
        return torch.cat((xyz.flatten() / .1, grasped, tool.new_tensor([float(self.closed)]))).cpu().numpy()

    def rgb(self):
        return self.camera[0, ..., :3].to(torch.uint8).cpu().numpy()

    def reset(self, seed):
        self.closed = False
        self.steps = self.stable = 0
        obs, _ = self.env.reset(seed=int(seed))
        return self.state(obs)

    def step(self, action):
        action = np.clip(action, -1, 1)
        device = self.env.unwrapped.device
        control = torch.zeros((1, 7), device=device)
        control[0, :3] = torch.as_tensor(action[:3], device=device) * (STEP / self.scale)
        # Keep wrist orientation fixed. Isaac relative IK uses positive values to open the fingers.
        self.closed = bool(action[3] > 0)
        control[0, 6] = -1 if self.closed else 1
        obs, _, terminated, truncated, _ = self.env.step(control)
        if (terminated | truncated).any().item():
            raise RuntimeError("Unexpected Isaac auto-reset at a dataset boundary")
        state = self.state(obs)
        tool, red, blue = state[:9].reshape(3, 3) * .1
        scene = self.env.unwrapped.scene
        velocities = (scene[k].data.root_lin_vel_w for k in ("cube_1", "cube_2"))
        slow = all(getattr(v, "torch", v)[0].norm().item() < .025 for v in velocities)
        stacked = (not self.closed and self.aperture > .035 and np.linalg.norm(tool - red) > .05
                   and np.linalg.norm(red[:2] - blue[:2]) < .012
                   and abs(red[2] - blue[2] - self.block_height) < .008
                   and .014 < blue[2] < .032 and slow)
        self.stable = self.stable + 1 if stacked else 0
        self.steps += 1
        if self.frame_callback is not None:
            self.frame_callback()
        success = self.stable >= 8
        return state, success, success or self.steps >= self.max_steps or min(red[2], blue[2]) < -.05
