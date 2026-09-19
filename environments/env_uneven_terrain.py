"""
Task 2 — Unitree A1 terrain-robust locomotion environment.

Architecture:
    PPO -> normalized action -> rate-limited desired joint position
        -> explicit joint-level PD -> torque -> PyBullet A1

Task 1 is intentionally left untouched. This environment keeps the same
55-D observation and 12-D normalized desired-position action interface so
the validated Task-1 PPO policy can be transferred/fine-tuned.

Terrain:
    A smooth randomized heightfield is regenerated on reset.
    difficulty controls terrain amplitude and spatial variation.
"""
from __future__ import annotations
from networkx.generators import spectral_graph_forge




import argparse
import os
from typing import Optional

import gymnasium as gym
import time
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from controllers.pd_controller import PDController

NUM_JOINTS = 12
ACTION_DIM = 12
PHYSICS_HZ = 240
CONTROL_HZ = 60
REALTIME_RENDER = True
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ

TORQUE_LIMIT = 33.5
ACTION_SCALE = 0.30
ACTION_RATE_LIMIT = 0.08
MAX_EPISODE_STEPS = 1000

STANDING_POSE = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float32)
JOINT_LOWER = np.array([-0.802, -1.047, -2.696] * 4, dtype=np.float32)
JOINT_UPPER = np.array([0.802, 4.189, -0.916] * 4, dtype=np.float32)

FALL_HEIGHT = 0.18
FALL_ROLL = np.deg2rad(50.0)
FALL_PITCH = np.deg2rad(50.0)


class QuadrupedTerrainEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render: bool = False,
        difficulty: float = 0.15,
        target_velocity: float = 0.5,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        super().__init__()
        self.render_enabled = bool(render)
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.target_velocity = float(target_velocity)
        self.max_episode_steps = int(max_episode_steps)

        self.physics_client: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.terrain_id: Optional[int] = None
        self.joint_indices = []
        self.foot_link_indices = []

        self.standing_pose = STANDING_POSE.copy()
        self.controller = PDController(kp=40.0, kd=1.0, torque_limit=TORQUE_LIMIT)

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        obs_lim = np.full(55, 1e6, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_lim, obs_lim, (55,), dtype=np.float32)

        self.current_q_des = self.standing_pose.copy()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.episode_step = 0
        self.prev_x = 0.0
        self.initial_yaw = 0.0
        self.last_torque = np.zeros(NUM_JOINTS, dtype=np.float32)

    def set_difficulty(self, difficulty: float):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self._reset_simulation()
        self._load_terrain()
        self._load_robot()
        self._initialize_joints()
        self._settle_robot(seconds=3.0)

        self.current_q_des = self.standing_pose.copy()
        self.prev_action.fill(0.0)
        self.episode_step = 0
        self.last_torque.fill(0.0)

        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        self.prev_x = float(pos[0])
        self.initial_yaw = float(p.getEulerFromQuaternion(orn)[2])

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        target_q_des = self.standing_pose + ACTION_SCALE * action
        target_q_des = np.clip(target_q_des, JOINT_LOWER, JOINT_UPPER)

        max_target_step = ACTION_RATE_LIMIT * ACTION_SCALE  # 0.024 rad/control step
        delta = np.clip(target_q_des - self.current_q_des,
                        -max_target_step, max_target_step)
        self.current_q_des = np.clip(
            self.current_q_des + delta, JOINT_LOWER, JOINT_UPPER
        ).astype(np.float32)

        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            q, qd = self._get_joint_state()
            torque = self.controller.compute_torque(self.current_q_des, q, qd)
            self.last_torque = torque.copy()
            for i, joint_id in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id, joint_id, p.TORQUE_CONTROL,
                    force=float(torque[i]), physicsClientId=self.physics_client
                )
            p.stepSimulation(physicsClientId=self.physics_client)

        # Keep the GUI playback synchronized with the 60 Hz control rate.
        # Physics still runs at 240 Hz (4 substeps per control step).
        if self.render_enabled and REALTIME_RENDER:
            time.sleep(1.0 / CONTROL_HZ)

        self.episode_step += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)

        terminated = bool(self._is_fallen())
        truncated = bool(self.episode_step >= self.max_episode_steps)

        info = self._get_info()
        info["is_fallen"] = bool(terminated)
        info["terrain_difficulty"] = self.difficulty

        self.prev_action = action.copy()

        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None
            self.robot_id = None
            self.terrain_id = None

    def _connect(self):
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render_enabled else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _reset_simulation(self):
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self.physics_client)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)

    def _load_terrain(self):
        rng = self.np_random
        rows, cols = 41, 21
        cell = 0.4

        # Smooth low-frequency terrain. Amplitude rises with difficulty.
        amplitude = 0.01 + 0.07 * self.difficulty
        raw = rng.normal(0.0, 1.0, (rows, cols)).astype(np.float32)

        # Repeated neighbor averaging creates hills rather than sharp noise.
        h = raw
        for _ in range(3):
            h = (
                0.40 * h
                + 0.15 * np.roll(h, 1, 0)
                + 0.15 * np.roll(h, -1, 0)
                + 0.15 * np.roll(h, 1, 1)
                + 0.15 * np.roll(h, -1, 1)
            )

        h = h / max(float(np.max(np.abs(h))), 1e-6) * amplitude

        # Keep the starting region close to flat so transfer begins safely.
        # The heightfield is centered at x=5.0. The A1 starts at x=0.0,
        # so flatten the initial ~1 m region around the robot's start.
        start_r = int(round((rows - 1) / 2 - 5.0 / cell))
        center_c = cols // 2
        for r in range(max(0, start_r - 2), min(rows, start_r + 3)):
            for c in range(center_c - 2, center_c + 3):
                h[r, c] *= 0.10

        # Heightfield is centered at its origin; shift it so x spans roughly
        # -3 to +13 m, covering the Task-1 evaluation distance.
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[cell, cell, 1.0],
            heightfieldData=h.flatten().tolist(),
            numHeightfieldRows=rows,
            numHeightfieldColumns=cols,
            physicsClientId=self.physics_client,
        )
        self.terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=[5.0, 0.0, 0.0],
            physicsClientId=self.physics_client,
        )

    def _load_robot(self):
        a1_path = os.path.join(pybullet_data.getDataPath(), "a1", "a1.urdf")
        self.robot_id = p.loadURDF(
            a1_path, basePosition=[0.0, 0.0, 0.35], useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.physics_client
        )

        self.joint_indices = []
        for jid in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            ji = p.getJointInfo(self.robot_id, jid, physicsClientId=self.physics_client)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(jid)

        if len(self.joint_indices) != NUM_JOINTS:
            raise RuntimeError(f"Expected {NUM_JOINTS} revolute joints, found {len(self.joint_indices)}.")

        for jid in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id, jid, p.VELOCITY_CONTROL, force=0.0,
                physicsClientId=self.physics_client
            )

        candidates = []
        for lid in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            ji = p.getJointInfo(self.robot_id, lid, physicsClientId=self.physics_client)
            name = ji[12].decode("utf-8").lower()
            if "foot" in name or "toe" in name:
                candidates.append(lid)

        self.foot_link_indices = candidates[:4]
        if len(self.foot_link_indices) != 4:
            self.foot_link_indices = [
                self.joint_indices[2], self.joint_indices[5],
                self.joint_indices[8], self.joint_indices[11]
            ]

    def _initialize_joints(self):
        for i, jid in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id, jid, float(self.standing_pose[i]), 0.0,
                physicsClientId=self.physics_client
            )
            p.setJointMotorControl2(
                self.robot_id, jid, p.VELOCITY_CONTROL, force=0.0,
                physicsClientId=self.physics_client
            )

    def _settle_robot(self, seconds=3.0):
        for _ in range(int(seconds * PHYSICS_HZ)):
            q, qd = self._get_joint_state()
            tau = self.controller.compute_torque(self.standing_pose, q, qd)
            for i, jid in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id, jid, p.TORQUE_CONTROL, force=float(tau[i]),
                    physicsClientId=self.physics_client
                )
            p.stepSimulation(physicsClientId=self.physics_client)

    def _get_joint_state(self):
        states = [
            p.getJointState(self.robot_id, jid, physicsClientId=self.physics_client)
            for jid in self.joint_indices
        ]
        return (
            np.asarray([s[0] for s in states], dtype=np.float32),
            np.asarray([s[1] for s in states], dtype=np.float32),
        )

    def _get_base_state(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        lv, av = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        rpy = p.getEulerFromQuaternion(orn)
        return (
            np.asarray(pos, dtype=np.float32),
            np.asarray(lv, dtype=np.float32),
            np.asarray(av, dtype=np.float32),
            np.asarray(rpy, dtype=np.float32),
            orn,
        )

    def _get_foot_contacts(self):
        out = []
        for lid in self.foot_link_indices:
            pts = p.getContactPoints(
                bodyA=self.robot_id, bodyB=self.terrain_id,
                linkIndexA=lid, physicsClientId=self.physics_client
            )
            out.append(float(bool(pts)))
        return np.asarray(out, dtype=np.float32)

    def _get_obs(self):
        pos, lv, av, rpy, orn = self._get_base_state()
        q, qd = self._get_joint_state()
        rot = np.asarray(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3,3)
        projected_g = rot.T @ np.array([0,0,-1], dtype=np.float32)
        target = np.array([self.target_velocity, 0, 0], dtype=np.float32)
        obs = np.concatenate([
            lv, av, rpy, q, qd, self._get_foot_contacts(),
            projected_g, target, self.current_q_des
        ]).astype(np.float32)
        if obs.shape != (55,) or not np.all(np.isfinite(obs)):
            raise FloatingPointError("Invalid Task-2 observation.")
        return obs

    def _compute_reward(self, action):
        pos, lv, av, rpy, _ = self._get_base_state()
        vx, vy = float(lv[0]), float(lv[1])
        roll, pitch, yaw = map(float, rpy)

        velocity_tracking = np.exp(-6.0 * (vx - self.target_velocity) ** 2)
        dx = float(pos[0] - self.prev_x)
        self.prev_x = float(pos[0])

        orientation_penalty = 1.5 * roll**2 + 1.5 * pitch**2
        lateral_penalty = 0.8 * abs(vy)
        yaw_penalty = 0.6 * abs(yaw - self.initial_yaw)
        torque_penalty = 0.002 * float(np.mean(self.last_torque ** 2))
        action_change_penalty = 0.02 * float(np.mean((action - self.prev_action) ** 2))
        angular_penalty = 0.02 * (abs(float(av[0])) + abs(float(av[1])))

        return (
            2.5 * velocity_tracking + 8.0 * dx + 0.5
            - orientation_penalty - lateral_penalty - yaw_penalty
            - torque_penalty - action_change_penalty - angular_penalty
        )

    def _is_fallen(self):
        pos, _, _, rpy, _ = self._get_base_state()
        return (
            float(pos[2]) < FALL_HEIGHT
            or abs(float(rpy[0])) > FALL_ROLL
            or abs(float(rpy[1])) > FALL_PITCH
        )

    def _get_info(self):
        pos, lv, av, rpy, _ = self._get_base_state()
        q, qd = self._get_joint_state()
        err = np.abs(self.current_q_des - q)
        return {
            "base_height": float(pos[2]),
            "base_x": float(pos[0]),
            "base_y": float(pos[1]),
            "base_velocity_x": float(lv[0]),
            "base_velocity_y": float(lv[1]),
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "mean_joint_error": float(np.mean(err)),
            "max_joint_error": float(np.max(err)),
            "max_torque": float(np.max(np.abs(self.last_torque))),
            "mean_joint_velocity": float(np.mean(np.abs(qd))),
            "terrain_difficulty": self.difficulty,
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--difficulty", type=float, default=0.15)
    a = ap.parse_args()

    env = QuadrupedTerrainEnv(render=a.render, difficulty=a.difficulty)
    obs, info = env.reset(seed=0)
    print("=" * 68)
    print("TASK 2 TERRAIN ENVIRONMENT SMOKE TEST")
    print("=" * 68)
    print("Observation:", obs.shape, obs.dtype)
    print("Action:", env.action_space.shape, env.action_space.dtype)
    print("Difficulty:", env.difficulty)
    print("Finite reset:", bool(np.all(np.isfinite(obs))))
    for i in range(a.steps):
        obs, rew, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
        if term or trunc:
            print("Ended at step:", i)
            break
    print("Final z:", info["base_height"], "roll:", info["roll"], "pitch:", info["pitch"])
    env.close()
