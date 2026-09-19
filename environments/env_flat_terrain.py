"""
Unitree A1 PPO Environment - Task 1
===================================

Flat-ground locomotion environment.

Control:
    PPO action
        -> desired joint position
        -> explicit PD controller
        -> torque
        -> PyBullet TORQUE_CONTROL
        -> Unitree A1

This version is deliberately kept clean and close to the validated
standalone PD experiment. It does not use PyBullet POSITION_CONTROL.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from controllers.pd_controller import PDController


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS = 12
ACTION_DIM = 12

PHYSICS_HZ = 240
CONTROL_HZ = 60
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ

TORQUE_LIMIT = 33.5

STANDING_POSE = np.array(
    [0.0, 0.9, -1.8] * 4,
    dtype=np.float32,
)

JOINT_LOWER = np.array(
    [-0.802, -1.047, -2.696] * 4,
    dtype=np.float32,
)

JOINT_UPPER = np.array(
    [0.802, 4.189, -0.916] * 4,
    dtype=np.float32,
)

ACTION_SCALE = 0.30

# Maximum normalized PPO-action change per 60 Hz control step.
# 0.08 * 0.30 = 0.024 rad maximum q_des change.
ACTION_RATE_LIMIT = 0.08

MAX_EPISODE_STEPS = 1000

FALL_HEIGHT = 0.18
FALL_ROLL = np.deg2rad(50.0)
FALL_PITCH = np.deg2rad(50.0)


class QuadrupedEnv(gym.Env):
    """
    Unitree A1 flat-ground RL environment.

    Observation: 55 values
        base linear velocity       3
        base angular velocity      3
        roll/pitch/yaw             3
        joint positions           12
        joint velocities          12
        foot contacts              4
        projected gravity          3
        target velocity             3
        desired joint positions    12
                                   --
                                   55

    Action:
        12 normalized desired-position offsets in [-1, 1].

    Zero action:
        desired joint position = STANDING_POSE.

    The nominal standing pose remains the PD reference. The steady-state
    joint error caused by gravity is intentional and generates holding torque.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        terrain_id: int = 0,
        render: bool = False,
        difficulty: float = 0.0,
        target_velocity: float = 0.5,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        super().__init__()

        if terrain_id != 0:
            raise ValueError(
                "Task 1 currently supports terrain_id=0 (flat terrain) only."
            )

        self.terrain_id = terrain_id
        self.render_enabled = bool(render)
        self.difficulty = float(difficulty)
        self.target_velocity = float(target_velocity)
        self.max_episode_steps = int(max_episode_steps)

        self.physics_client: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.plane_id: Optional[int] = None

        self.joint_indices: list[int] = []
        self.foot_link_indices: list[int] = []

        self.standing_pose = STANDING_POSE.copy()

        # Same controller values as the validated standalone PD test.
        self.controller = PDController(
            kp=40.0,
            kd=1.0,
            torque_limit=TORQUE_LIMIT,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        obs_limit = np.full(55, 1e6, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_limit,
            high=obs_limit,
            shape=(55,),
            dtype=np.float32,
        )

        self.current_q_des = self.standing_pose.copy()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        self.episode_step = 0
        self.prev_x = 0.0
        self.initial_yaw = 0.0

        self.last_torque = np.zeros(NUM_JOINTS, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._connect()
        self._reset_simulation()
        self._load_world()
        self._load_robot()
        self._initialize_joints()

        # Exactly the same explicit PD settling concept as the standalone
        # diagnostic: 3 seconds at 240 Hz.
        self._settle_robot(seconds=3.0)

        self.episode_step = 0
        self.current_q_des = self.standing_pose.copy()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_torque = np.zeros(NUM_JOINTS, dtype=np.float32)

        pos, _, _, rpy = self._get_base_state()
        self.prev_x = float(pos[0])
        self.initial_yaw = float(rpy[2])

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected action shape {(ACTION_DIM,)}, got {action.shape}"
            )

        action = np.clip(action, -1.0, 1.0)

        # PPO action -> desired joint-position target.
        # Rate-limit the target before sending it to the explicit PD loop.
        target_q_des = self.standing_pose + ACTION_SCALE * action
        target_q_des = np.clip(
            target_q_des,
            JOINT_LOWER,
            JOINT_UPPER,
        ).astype(np.float32)

        max_q_change = ACTION_RATE_LIMIT * ACTION_SCALE
        q_delta = target_q_des - self.current_q_des
        q_delta = np.clip(
            q_delta,
            -max_q_change,
            max_q_change,
        )

        self.current_q_des = (
            self.current_q_des + q_delta
        ).astype(np.float32)

        self.current_q_des = np.clip(
            self.current_q_des,
            JOINT_LOWER,
            JOINT_UPPER,
        ).astype(np.float32)

        # Explicit PD -> torque.
        q, qd = self._get_joint_state()

        torque = self.controller.compute_torque(
            self.current_q_des,
            q,
            qd,
        )

        torque = np.clip(
            torque,
            -TORQUE_LIMIT,
            TORQUE_LIMIT,
        ).astype(np.float32)

        self.last_torque = torque.copy()

        # PyBullet TORQUE_CONTROL.
        # Recompute PD torque at every 240 Hz physics substep.
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            q, qd = self._get_joint_state()

            torque = self.controller.compute_torque(
                self.current_q_des,
                q,
                qd,
            )

            torque = np.clip(
                torque,
                -TORQUE_LIMIT,
                TORQUE_LIMIT,
            ).astype(np.float32)

            self.last_torque = torque.copy()

            for i, joint_id in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=p.TORQUE_CONTROL,
                    force=float(torque[i]),
                    physicsClientId=self.physics_client,
                )

            p.stepSimulation(
                physicsClientId=self.physics_client
            )

        self.episode_step += 1

        obs = self._get_obs()

        reward = self._compute_reward(
            action,
            self.prev_action,
        )

        terminated = self._is_fallen()
        truncated = self.episode_step >= self.max_episode_steps

        info = self._get_info()
        info["is_fallen"] = bool(terminated)

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
            self.plane_id = None

    # -----------------------------------------------------------------------
    # Simulation setup
    # -----------------------------------------------------------------------

    def _connect(self):
        if self.physics_client is not None:
            return

        mode = p.GUI if self.render_enabled else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _reset_simulation(self):
        p.resetSimulation(
            physicsClientId=self.physics_client
        )

        p.setGravity(
            0,
            0,
            -9.81,
            physicsClientId=self.physics_client,
        )

        p.setTimeStep(
            1.0 / PHYSICS_HZ,
            physicsClientId=self.physics_client,
        )

        p.setRealTimeSimulation(
            0,
            physicsClientId=self.physics_client,
        )

    def _load_world(self):
        self.plane_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=self.physics_client,
        )

    def _load_robot(self):
        # Same A1 loading style as the validated standalone PD test.
        a1_path = os.path.join(
            pybullet_data.getDataPath(),
            "a1",
            "a1.urdf",
        )

        self.robot_id = p.loadURDF(
            a1_path,
            basePosition=[0.0, 0.0, 0.35],
            useFixedBase=False,
            physicsClientId=self.physics_client,
        )

        self.joint_indices = []

        for joint_id in range(
            p.getNumJoints(
                self.robot_id,
                physicsClientId=self.physics_client,
            )
        ):
            info = p.getJointInfo(
                self.robot_id,
                joint_id,
                physicsClientId=self.physics_client,
            )

            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(joint_id)

        if len(self.joint_indices) != NUM_JOINTS:
            raise RuntimeError(
                f"Expected {NUM_JOINTS} revolute joints, "
                f"found {len(self.joint_indices)}."
            )

        # Disable default motors exactly as in standalone test.
        for joint_id in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.physics_client,
            )

        self._find_foot_links()

    def _initialize_joints(self):
        # Exact nominal initialization. No randomization for Task 1.
        for i, joint_id in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=float(self.standing_pose[i]),
                targetVelocity=0.0,
                physicsClientId=self.physics_client,
            )

        for joint_id in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.physics_client,
            )

    def _settle_robot(self, seconds: float = 3.0):
        steps = int(seconds * PHYSICS_HZ)

        for _ in range(steps):
            q, qd = self._get_joint_state()

            torque = self.controller.compute_torque(
                self.standing_pose,
                q,
                qd,
            )

            for i, joint_id in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_id,
                    p.TORQUE_CONTROL,
                    force=float(torque[i]),
                    physicsClientId=self.physics_client,
                )

            p.stepSimulation(
                physicsClientId=self.physics_client
            )

    # -----------------------------------------------------------------------
    # State
    # -----------------------------------------------------------------------

    def _get_joint_state(self):
        states = [
            p.getJointState(
                self.robot_id,
                joint_id,
                physicsClientId=self.physics_client,
            )
            for joint_id in self.joint_indices
        ]

        q = np.asarray(
            [s[0] for s in states],
            dtype=np.float32,
        )

        qd = np.asarray(
            [s[1] for s in states],
            dtype=np.float32,
        )

        return q, qd

    def _get_base_state(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client,
        )

        linear_vel, angular_vel = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.physics_client,
        )

        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        return (
            np.asarray(pos, dtype=np.float32),
            np.asarray(linear_vel, dtype=np.float32),
            np.asarray(angular_vel, dtype=np.float32),
            np.asarray([roll, pitch, yaw], dtype=np.float32),
        )

    def _find_foot_links(self):
        candidates = []

        for link_id in range(
            p.getNumJoints(
                self.robot_id,
                physicsClientId=self.physics_client,
            )
        ):
            info = p.getJointInfo(
                self.robot_id,
                link_id,
                physicsClientId=self.physics_client,
            )

            link_name = info[12].decode(
                "utf-8",
                errors="ignore",
            ).lower()

            if "foot" in link_name or "toe" in link_name:
                candidates.append(link_id)

        if len(candidates) >= 4:
            self.foot_link_indices = candidates[:4]
        else:
            self.foot_link_indices = [
                self.joint_indices[2],
                self.joint_indices[5],
                self.joint_indices[8],
                self.joint_indices[11],
            ]

    def _get_foot_contacts(self):
        contacts = []

        for link_id in self.foot_link_indices:
            points = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=link_id,
                physicsClientId=self.physics_client,
            )

            contacts.append(
                1.0 if len(points) > 0 else 0.0
            )

        return np.asarray(
            contacts,
            dtype=np.float32,
        )

    def _get_projected_gravity(self, orn):
        rotation = np.asarray(
            p.getMatrixFromQuaternion(orn),
            dtype=np.float32,
        ).reshape(3, 3)

        gravity_world = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )

        return (
            rotation.T @ gravity_world
        ).astype(np.float32)

    def _get_obs(self):
        pos, linear_vel, angular_vel, rpy = (
            self._get_base_state()
        )

        _, orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client,
        )

        q, qd = self._get_joint_state()

        contacts = self._get_foot_contacts()
        gravity = self._get_projected_gravity(orn)

        target_velocity = np.array(
            [self.target_velocity, 0.0, 0.0],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                linear_vel,
                angular_vel,
                rpy,
                q,
                qd,
                contacts,
                gravity,
                target_velocity,
                self.current_q_des,
            ]
        ).astype(np.float32)

        if obs.shape != (55,):
            raise RuntimeError(
                f"Expected observation shape (55,), got {obs.shape}"
            )

        if not np.all(np.isfinite(obs)):
            raise FloatingPointError(
                "Observation contains NaN or Inf."
            )

        return obs

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _compute_reward(self, action, previous_action):
        pos, linear_vel, angular_vel, rpy = (
            self._get_base_state()
        )

        vx = float(linear_vel[0])
        vy = float(linear_vel[1])

        roll = float(rpy[0])
        pitch = float(rpy[1])
        yaw = float(rpy[2])

        velocity_error = vx - self.target_velocity

        velocity_tracking = np.exp(
            -6.0 * velocity_error * velocity_error
        )

        dx = float(pos[0] - self.prev_x)
        self.prev_x = float(pos[0])

        forward_progress = 8.0 * dx
        alive_reward = 0.5

        orientation_penalty = (
            1.5 * roll * roll
            + 1.5 * pitch * pitch
        )

        lateral_penalty = 0.8 * abs(vy)

        yaw_error = yaw - self.initial_yaw
        yaw_penalty = 0.6 * abs(yaw_error)

        torque_penalty = 0.002 * float(
            np.mean(self.last_torque ** 2)
        )

        action_change_penalty = 0.02 * float(
            np.mean(
                (action - previous_action) ** 2
            )
        )

        angular_penalty = 0.02 * (
            abs(float(angular_vel[0]))
            + abs(float(angular_vel[1]))
        )

        reward = (
            2.5 * velocity_tracking
            + forward_progress
            + alive_reward
            - orientation_penalty
            - lateral_penalty
            - yaw_penalty
            - torque_penalty
            - action_change_penalty
            - angular_penalty
        )

        return float(reward)

    # -----------------------------------------------------------------------
    # Termination / diagnostics
    # -----------------------------------------------------------------------

    def _is_fallen(self):
        pos, _, _, rpy = self._get_base_state()

        if float(pos[2]) < FALL_HEIGHT:
            return True

        if abs(float(rpy[0])) > FALL_ROLL:
            return True

        if abs(float(rpy[1])) > FALL_PITCH:
            return True

        return False

    def _get_info(self):
        pos, linear_vel, angular_vel, rpy = (
            self._get_base_state()
        )

        q, qd = self._get_joint_state()

        joint_error = np.abs(
            self.current_q_des - q
        )

        return {
            "base_height": float(pos[2]),
            "base_x": float(pos[0]),
            "base_velocity_x": float(linear_vel[0]),
            "base_velocity_y": float(linear_vel[1]),
            "base_angular_velocity": float(
                np.linalg.norm(angular_vel)
            ),
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "mean_joint_error": float(
                np.mean(joint_error)
            ),
            "max_joint_error": float(
                np.max(joint_error)
            ),
            "max_torque": float(
                np.max(np.abs(self.last_torque))
            ),
            "mean_joint_velocity": float(
                np.mean(np.abs(qd))
            ),
            "episode_step": int(self.episode_step),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("A1 RL ENVIRONMENT - PD HOLD SMOKE TEST")
    print("=" * 60)

    env = QuadrupedEnv(
        terrain_id=0,
        render=args.render,
        target_velocity=0.5,
    )

    obs, info = env.reset()

    print(
        f"Observation shape: {obs.shape}"
    )

    print(
        f"Action shape: {env.action_space.shape}"
    )

    print(
        f"Initial observation finite: "
        f"{np.all(np.isfinite(obs))}"
    )

    print(
        f"Reset state | "
        f"z={info['base_height']:.3f} | "
        f"roll={info['roll']:+.3f} | "
        f"pitch={info['pitch']:+.3f} | "
        f"mean_err={info['mean_joint_error']:.4f} | "
        f"max_err={info['max_joint_error']:.4f}"
    )

    print()
    print("Testing ZERO action.")
    print("Zero action -> standing pose -> PD -> torque.")
    print()

    zero_action = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = (
            env.step(zero_action)
        )

        if step % 10 == 0:
            print(
                f"step={step:3d} | "
                f"reward={reward:+7.3f} | "
                f"x={info['base_x']:+.3f} | "
                f"z={info['base_height']:.3f} | "
                f"roll={info['roll']:+.3f} | "
                f"pitch={info['pitch']:+.3f} | "
                f"mean_err={info['mean_joint_error']:.4f} | "
                f"max_tau={info['max_torque']:.2f}"
            )

        if terminated or truncated:
            print(
                f"Episode ended at step: {step}"
            )
            break

    env.close()

    print()
    print("Environment validation finished.")


if __name__ == "__main__":
    main()
