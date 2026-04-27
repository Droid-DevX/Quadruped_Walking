"""
env.py — Unified Quadruped Locomotion Environment
==================================================
Single ``QuadrupedEnv(gym.Env)`` class that supports three terrain modes
selected by ``terrain_id``:

    0  — Flat ground (baseline)
    1  — 3-segment procedural slope  (downhill → plateau → uphill)
    2  — Obstacle-avoidance corridor (random boxes/cylinders + lidar + goal)

Usage
-----
    from env import QuadrupedEnv
    env = QuadrupedEnv(terrain_id=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Observation vector layout (terrain 0 / 1):
    [ base_vel(3) | base_ang_vel(3) | rpy(3) | joint_pos(12) | joint_vel(12)
    | foot_contacts(4) | gravity_proj(3) | target_vel(3) | slope_sin(1)
    | lidar(16) ]                                               total = 60

Observation vector layout (terrain 2):
    [ proprioception(44) | lidar_scan(16) | goal_relative(3) ]  total = 60
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

# Absolute path to the project root (directory containing this file).
# Used so PyBullet can find 'a1/urdf/a1.urdf' regardless of CWD or n_envs.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

NUM_JOINTS        = 12
ACT_DIM           = NUM_JOINTS
LIDAR_RAYS        = 16          # 1-D lidar for all terrains
PROP_DIM          = 44          # base_vel(3)+ang(3)+rpy(3)+jpos(12)+jvel(12)+contacts(4)+grav(3)+tvec(3)+slope(1)
OBS_DIM           = PROP_DIM + LIDAR_RAYS  # terrain 0/1: goal_rel replaced by zeros; terrain 2: + goal_rel → still 60
# terrain 2 uses the last 16 floats as lidar AND appends 3 goal floats keeping total = 60 (16+3=19 sensor → 44+16=60, goal replaces 3 lidar slots)
# Simpler: always 60 dims — proprioception 44, lidar 13, goal_relative 3
LIDAR_RAYS_T2     = 13          # terrain 2: 13-ray lidar (rest filled by goal 3)
GOAL_DIM          = 3           # dx, dy, dist in robot frame
_SENS_DIM         = LIDAR_RAYS  # 16 total sensor floats (13 lidar + 3 goal for T2, 16 lidar for T0/T1)

MAX_EPISODE_STEPS = 1000
CONTROL_HZ        = 60
PHYSICS_SUBSTEPS  = 4
TORQUE_LIMIT      = 33.5

# Obstacle-avoidance (terrain 2) settings
LIDAR_RANGE       = 5.0         # metres — max raycasting distance
T2_LIDAR_RAYS     = 13          # rays for terrain-2 lidar (±45° FOV)
T2_GOAL_APPROACH  = 0.5         # metres — goal-reached radius
T2_GOAL_MIN_DIST  = 5.0         # metres — min goal spawn distance
T2_GOAL_MAX_DIST  = 15.0        # metres — max goal spawn distance
T2_REWARD_GOAL    = 10.0        # bonus per goal reached
T2_PENALTY_COL    = 2.0         # penalty per step in collision with obstacle
T2_PENALTY_FALL   = 50.0        # terminal fall penalty
T2_BASE_HEIGHT_MIN = 0.12       # metres — fall threshold for obstacle terrain

# Real Unitree A1 joint limits (radians)
A1_JOINT_LIMITS: List[Tuple[float, float]] = [
    (-0.802,  0.802),   # FL_hip
    (-1.047,  4.189),   # FL_thigh
    (-2.696, -0.916),   # FL_calf
    (-0.802,  0.802),   # FR_hip
    (-1.047,  4.189),   # FR_thigh
    (-2.696, -0.916),   # FR_calf
    (-0.802,  0.802),   # RL_hip
    (-1.047,  4.189),   # RL_thigh
    (-2.696, -0.916),   # RL_calf
    (-0.802,  0.802),   # RR_hip
    (-1.047,  4.189),   # RR_thigh
    (-2.696, -0.916),   # RR_calf
]

STANDING_POSE = [0.0, 0.9, -1.8] * 4   # default standing joint targets


# ─────────────────────────────────────────────────────────────────────────────

class QuadrupedEnv(gym.Env):
    """Unified quadruped locomotion environment for MuJoCo/PyBullet.

    Parameters
    ----------
    terrain_id : int
        0 = flat ground, 1 = procedural slope, 2 = obstacle-avoidance corridor.
    render : bool
        Open the PyBullet GUI if True.
    difficulty : float [0, 1]
        Curriculum difficulty (used for scaling parameters in some terrains).
    target_velocity : float
        Forward velocity target (m/s) used in the reward function.
    energy_penalty : float
        Scaling coefficient for the torque-energy penalty term.
    alive_bonus : float
        Per-step alive bonus when the robot is moving forward.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    # Camera defaults (used by test_sac.py)
    _CAM_DISTANCE   = 2.5
    _CAM_PITCH      = -12.0
    _CAM_YAW_OFFSET = 180.0

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        terrain_id: int      = 0,
        render: bool         = False,
        difficulty: float    = 1.0,
        target_velocity: float = 0.5,
        energy_penalty: float  = 0.008,
        alive_bonus: float     = 1.5,
        # legacy alias kept for backward-compat with train_curriculum.py
        terrain_level: Optional[int] = None,
    ) -> None:
        super().__init__()
        # honour legacy kwarg
        if terrain_level is not None:
            terrain_id = int(terrain_level)

        self.terrain_id   = int(np.clip(terrain_id, 0, 2))
        self.render_mode  = render
        self.difficulty   = float(np.clip(difficulty, 0.0, 1.0))
        self.target_vel   = np.array([target_velocity, 0.0, 0.0], dtype=np.float32)
        self.energy_pen   = energy_penalty
        self.alive_bonus  = alive_bonus

        obs_high = np.inf * np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

        # internal state
        self._physics_client: Optional[int] = None
        self._robot_id:       Optional[int] = None
        self._step_count  = 0
        self._prev_pos    = np.zeros(3)
        self._prev_action = np.zeros(ACT_DIM)
        self._joint_ids:   List[int]             = []
        self._joint_limits: List[Tuple[float, float]] = []
        self._init_yaw    = 0.0
        self._obstacle_ids: List[int] = []

        # terrain-2 goal state
        self._goal_pos: np.ndarray = np.zeros(3)
        self._goal_body_id: Optional[int] = None
        self._goals_reached: int = 0

        # terrain-1 slope parameters (re-sampled each reset)
        self._slope_L1: float = 5.0
        self._slope_L2: float = 6.0
        self._slope_L3: float = 5.0
        self._slope_a1: float = -np.deg2rad(12.0)   # downhill
        self._slope_a3: float =  np.deg2rad(12.0)   # uphill
        # cumulative X breakpoints (set by _sample_slope_params)
        self._slope_x0: float = 0.0    # world-X where downhill ends / plateau starts
        self._slope_x1: float = 0.0    # world-X where plateau ends / uphill starts
        self._slope_z0: float = 0.0    # height at start of plateau
        self._slope_z1: float = 0.0    # height at start of uphill

        self._connect()

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the simulation and return the initial observation."""
        if seed is not None:
            np.random.seed(seed)

        # reconnect if session dropped
        try:
            p.getConnectionInfo(self._physics_client)
        except Exception:
            self._connect()

        p.resetSimulation(physicsClientId=self._physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        p.setTimeStep(1.0 / CONTROL_HZ, physicsClientId=self._physics_client)
        # pybullet_data contains plane.urdf, etc.
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._physics_client)

        self._obstacle_ids  = []
        self._goal_body_id  = None
        self._goals_reached = 0

        # re-sample terrain parameters BEFORE building terrain
        if self.terrain_id == 1:
            self._sample_slope_params()

        self._build_terrain()

        # ── spawn robot ──────────────────────────────────────────────────────
        # Terrain 1: spawn in flat lead-in (x=-1) so robot starts level.
        # Terrain 0/2: spawn at origin.
        if self.terrain_id == 1:
            spawn_x, spawn_y = -1.0, 0.0
        else:
            spawn_x, spawn_y = 0.0, 0.0
        ground_z  = self._get_ground_height_at(spawn_x, spawn_y)
        start_pos = [spawn_x, spawn_y, ground_z + 0.48]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

        urdf_path = os.path.join(pybullet_data.getDataPath(), "a1", "a1.urdf")
        self._robot_id = p.loadURDF(
            urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn,
            physicsClientId=self._physics_client,
        )

        self._cache_joint_info()
        self._randomise_init_pose()

        self._step_count  = 0
        self._prev_pos    = np.array(start_pos)
        self._prev_action = np.zeros(ACT_DIM)

        _, orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._physics_client)
        _, _, self._init_yaw = p.getEulerFromQuaternion(orn)

        # Settle robot into standing pose.
        # Fewer steps for non-flat terrains to keep reset cost low when
        # episodes are short during early fine-tuning.
        settle_steps = 200 if self.terrain_id == 0 else 80
        for i, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot_id, jid, STANDING_POSE[i], 0.0,
                              physicsClientId=self._physics_client)
        for _ in range(settle_steps):
            for i, jid in enumerate(self._joint_ids):
                p.setJointMotorControl2(
                    self._robot_id, jid, p.POSITION_CONTROL,
                    targetPosition=STANDING_POSE[i], force=TORQUE_LIMIT,
                    positionGain=1.0, velocityGain=0.2,
                    physicsClientId=self._physics_client,
                )
            p.stepSimulation(physicsClientId=self._physics_client)

        # spawn goal marker for terrain 2 AFTER robot is settled
        if self.terrain_id == 2:
            self._spawn_goal()

        if self.render_mode:
            self._follow_camera()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Advance simulation by one control step."""
        action = np.clip(action, -1.0, 1.0)
        action = 0.2 * self._prev_action + 0.8 * action
        self._prev_action = action.copy()

        self._apply_action(action)
        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._physics_client)

        self._step_count += 1
        if self.render_mode:
            time.sleep(PHYSICS_SUBSTEPS / CONTROL_HZ)
            self._follow_camera()

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_done()
        truncated  = self._step_count >= MAX_EPISODE_STEPS
        info       = self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Disconnect physics client."""
        if not hasattr(self, "_physics_client") or self._physics_client is None:
            return
        try:
            p.disconnect(physicsClientId=self._physics_client)
        except Exception:
            pass
        self._physics_client = None

    def __del__(self) -> None:
        self.close()

    # ── legacy attribute alias ────────────────────────────────────────────────

    @property
    def terrain_level(self) -> int:
        """Backward-compatible alias for ``terrain_id``."""
        return self.terrain_id

    @terrain_level.setter
    def terrain_level(self, value: int) -> None:
        self.terrain_id = int(np.clip(value, 0, 2))

    def set_terrain_level(self, level: int) -> None:
        """Convenience setter (kept for backward-compat)."""
        self.terrain_id = int(np.clip(level, 0, 2))

    # ─────────────────────────────────────────────────────────────────────────
    # TERRAIN BUILDING
    # ─────────────────────────────────────────────────────────────────────────

    def _build_terrain(self) -> None:
        """Dispatch terrain construction based on ``self.terrain_id``."""
        if self.terrain_id == 0:
            self._build_terrain_flat()
        elif self.terrain_id == 1:
            self._build_terrain_slope()
        else:
            self._build_terrain_obstacles()

    # ── terrain 0: flat ───────────────────────────────────────────────────────

    def _build_terrain_flat(self) -> None:
        """Load a simple flat ground plane."""
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self._physics_client,
        )

    # ── terrain 1: 3-segment procedural slope ────────────────────────────────

    def _sample_slope_params(self) -> None:
        """Sample slope segment lengths and angles; compute breakpoint heights.

        Segment layout along the X axis (robot spawns at X=0):
            [−L1 .. 0]       — flat lead-in
            [0  .. L1]       — downhill  (angle a1, negative = down)
            [L1 .. L1+L2]    — plateau   (flat)
            [L1+L2 .. L1+L2+L3] — uphill (angle a3, positive = up)
        """
        scale  = max(0.2, self.difficulty)
        # lengths
        self._slope_L1 = float(np.random.uniform(2.0, 8.0))
        self._slope_L2 = float(np.random.uniform(3.0, 10.0))
        self._slope_L3 = float(np.random.uniform(2.0, 8.0))
        # angles (scale by difficulty so curriculum starts gentle)
        max_a = np.deg2rad(5.0 + 15.0 * scale)   # 5°..20° depending on difficulty
        self._slope_a1 = float(np.random.uniform(-max_a, -np.deg2rad(5.0)))  # downhill
        self._slope_a3 = float(np.random.uniform( np.deg2rad(5.0),  max_a))  # uphill

        # C0 height continuity:
        # Downhill ends at X = L1 with height h0 = tan(a1)*L1
        # Plateau is flat at that height
        # Uphill starts there and climbs
        self._slope_x0 = self._slope_L1                      # end of downhill
        self._slope_x1 = self._slope_L1 + self._slope_L2     # end of plateau
        self._slope_z0 = np.tan(self._slope_a1) * self._slope_L1   # height at plateau (negative)
        self._slope_z1 = self._slope_z0                             # plateau is flat
        # (uphill height at X > x1 is: z1 + tan(a3)*(X - x1))

    def _slope_height_at(self, x: float) -> float:
        """Return terrain height at world-X for the 3-segment slope."""
        if x < 0.0:
            # flat lead-in before downhill
            return 0.0
        if x < self._slope_x0:
            # downhill segment
            return float(np.tan(self._slope_a1) * x)
        if x < self._slope_x1:
            # plateau
            return float(self._slope_z0)
        # uphill segment
        return float(self._slope_z1 + np.tan(self._slope_a3) * (x - self._slope_x1))

    def _slope_angle_at(self, x: float) -> float:
        """Return the terrain pitch angle (radians) at world-X for terrain 1."""
        if x < 0.0:
            return 0.0
        if x < self._slope_x0:
            return self._slope_a1
        if x < self._slope_x1:
            return 0.0
        return self._slope_a3

    def _build_terrain_slope(self) -> None:
        """Build the 3-segment slope as a heightfield.

        A heightfield long enough to cover all three segments plus margins is
        generated analytically from ``_slope_height_at`` and loaded as a
        PyBullet GEOM_HEIGHTFIELD.
        """
        total_x  = self._slope_L1 + self._slope_L2 + self._slope_L3 + 4.0  # extra margin
        lead_in  = 2.0    # flat lead-in before X=0
        x_min    = -lead_in
        x_max    = total_x

        rows     = 512    # resolution along X
        cols     = 16     # resolution along Y (wide enough to walk)
        dx       = (x_max - x_min) / rows
        dy       = 4.0 / cols          # corridor is 4 m wide

        mesh_scale_x = dx
        mesh_scale_y = dy
        mesh_scale_z = 1.0

        heights = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            x = x_min + r * dx
            h = self._slope_height_at(x)
            heights[r, :] = h

        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[mesh_scale_x, mesh_scale_y, mesh_scale_z],
            heightfieldTextureScaling=256,
            heightfieldData=heights.flatten(order='F').tolist(),
            numHeightfieldRows=rows,
            numHeightfieldColumns=cols,
            physicsClientId=self._physics_client,
        )
        terrain_id = p.createMultiBody(0, shape, physicsClientId=self._physics_client)

        centre_x = x_min + (x_max - x_min) / 2.0
        centre_y = 0.0
        min_h    = float(heights.min())
        max_h    = float(heights.max())
        centre_z = (min_h + max_h) / 2.0

        p.resetBasePositionAndOrientation(
            terrain_id,
            [centre_x, centre_y, centre_z],
            [0, 0, 0, 1],
            physicsClientId=self._physics_client,
        )
        p.changeVisualShape(
            terrain_id, -1, rgbaColor=[0.75, 0.75, 0.75, 1],
            physicsClientId=self._physics_client,
        )

    # ── terrain 2: obstacle corridor ─────────────────────────────────────────

    def _build_terrain_obstacles(self) -> None:
        """Build flat corridor and spawn N random box/cylinder obstacles.

        Obstacle count, positions, sizes, and types are re-randomised on each
        reset.  Obstacles are solid collidable bodies. The difficulty controls
        the number and size of the obstacles.
        """
        # flat ground
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self._physics_client,
        )

        n_obs_max = max(1, int(15 * self.difficulty))
        n_obs_min = int(8 * self.difficulty)
        n_obs = 0 if self.difficulty < 0.05 else int(np.random.randint(n_obs_min, n_obs_max + 1))
        
        scale = max(0.1, self.difficulty) # scale dimensions from 10% to 100%

        for _ in range(n_obs):
            x = float(np.random.uniform(2.0, 25.0))
            y = float(np.random.uniform(-1.5, 1.5))

            if np.random.random() < 0.5:
                # box obstacle
                half_x = float(np.random.uniform(0.05, 0.25)) * scale
                half_y = float(np.random.uniform(0.05, 0.35)) * scale
                half_z = float(np.random.uniform(0.10, 0.35)) * scale
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[half_x, half_y, half_z],
                    physicsClientId=self._physics_client,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[half_x, half_y, half_z],
                    rgbaColor=[0.85, 0.25, 0.20, 1.0],
                    physicsClientId=self._physics_client,
                )
                body = p.createMultiBody(
                    0, col, vis, basePosition=[x, y, half_z],
                    physicsClientId=self._physics_client,
                )
            else:
                # cylinder obstacle
                radius = float(np.random.uniform(0.05, 0.20)) * scale
                height = float(np.random.uniform(0.20, 0.50)) * scale
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height,
                    physicsClientId=self._physics_client,
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=[0.20, 0.50, 0.85, 1.0],
                    physicsClientId=self._physics_client,
                )
                body = p.createMultiBody(
                    0, col, vis, basePosition=[x, y, height / 2.0],
                    physicsClientId=self._physics_client,
                )

            self._obstacle_ids.append(body)

    # ─────────────────────────────────────────────────────────────────────────
    # GOAL MANAGEMENT (terrain 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _spawn_goal(self) -> None:
        """Spawn (or move) the visual goal sphere ahead of the robot."""
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        dist  = float(np.random.uniform(T2_GOAL_MIN_DIST, T2_GOAL_MAX_DIST))
        angle = float(np.random.uniform(-np.pi / 6, np.pi / 6))   # ±30° off forward
        gx    = base_pos[0] + dist * np.cos(angle)
        gy    = base_pos[1] + dist * np.sin(angle)
        self._goal_pos = np.array([gx, gy, 0.1], dtype=np.float32)

        if self._goal_body_id is not None:
            # move existing marker
            p.resetBasePositionAndOrientation(
                self._goal_body_id,
                [gx, gy, 0.1],
                [0, 0, 0, 1],
                physicsClientId=self._physics_client,
            )
        else:
            # create visual-only sphere (no collision)
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.25,
                rgbaColor=[0.0, 1.0, 0.2, 0.7],
                physicsClientId=self._physics_client,
            )
            self._goal_body_id = p.createMultiBody(
                0, -1, vis,
                basePosition=[gx, gy, 0.1],
                physicsClientId=self._physics_client,
            )

    def _check_goal(self) -> float:
        """Return goal bonus if the robot reached the current goal; 0 otherwise."""
        if self.terrain_id != 2 or self._robot_id is None:
            return 0.0
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        dx   = base_pos[0] - self._goal_pos[0]
        dy   = base_pos[1] - self._goal_pos[1]
        dist = float(np.sqrt(dx * dx + dy * dy))
        if dist < T2_GOAL_APPROACH:
            self._goals_reached += 1
            self._spawn_goal()
            return T2_REWARD_GOAL
        return 0.0

    def _goal_relative(self) -> np.ndarray:
        """Return goal vector (dx, dy, dist) in the robot's local frame."""
        if self.terrain_id != 2 or self._robot_id is None:
            return np.zeros(GOAL_DIM, dtype=np.float32)
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        world_delta = np.array([
            self._goal_pos[0] - base_pos[0],
            self._goal_pos[1] - base_pos[1],
            0.0,
        ])
        local_delta = rot_mat.T @ world_delta
        dist = float(np.linalg.norm(local_delta[:2]))
        return np.array([local_delta[0], local_delta[1], dist], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # HEIGHT & SLOPE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ground_height_at(self, x: float, y: float) -> float:
        """Return the ground height (z) at world position (x, y)."""
        if self.terrain_id == 1:
            return self._slope_height_at(x)
        return 0.0

    def _get_ground_height(self) -> float:
        """Return terrain height below the robot's current base position."""
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        return self._get_ground_height_at(base_pos[0], base_pos[1])

    def _get_slope_pitch_offset(self) -> float:
        """Return the terrain pitch angle (rad) at the robot's X position."""
        if self.terrain_id == 1 and self._robot_id is not None:
            base_pos, _ = p.getBasePositionAndOrientation(
                self._robot_id, physicsClientId=self._physics_client
            )
            return self._slope_angle_at(base_pos[0])
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # SENSORS & OBSERVATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _cache_joint_info(self) -> None:
        """Populate ``_joint_ids`` and ``_joint_limits`` from the loaded URDF."""
        self._joint_ids, self._joint_limits = [], []
        for j in range(p.getNumJoints(self._robot_id, physicsClientId=self._physics_client)):
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self._physics_client)
            if info[2] == p.JOINT_REVOLUTE:
                lo, hi = info[8], info[9]
                idx = len(self._joint_ids)
                if abs(hi - lo) < 1e-6 and idx < len(A1_JOINT_LIMITS):
                    lo, hi = A1_JOINT_LIMITS[idx]
                self._joint_ids.append(j)
                self._joint_limits.append((lo, hi))
        self._joint_ids    = self._joint_ids[:NUM_JOINTS]
        self._joint_limits = self._joint_limits[:NUM_JOINTS]

    def _randomise_init_pose(self) -> None:
        """Add small random noise to each joint's initial state."""
        noise = 0.05
        for i, jid in enumerate(self._joint_ids):
            target = STANDING_POSE[i] + float(np.random.uniform(-noise, noise))
            p.resetJointState(self._robot_id, jid, target,
                              physicsClientId=self._physics_client)

    def _get_lidar_obs(self) -> np.ndarray:
        """Compute normalised 1-D lidar scan from the robot's head.

        For terrain 2: 13 rays over ±45° horizontal FOV, range ``LIDAR_RANGE``
        metres, returned as fractions in [0, 1] (1 = no hit).
        For other terrains: returns np.ones(LIDAR_RAYS).

        Returns
        -------
        np.ndarray of shape (LIDAR_RAYS,)  — terrain 0/1 return all-ones (16 rays).
                                             terrain 2 returns 13-ray array.
        """
        if self.terrain_id != 2 or self._robot_id is None:
            return np.ones(LIDAR_RAYS, dtype=np.float32)

        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)

        # head-mounted sensor offset (forward 0.2 m, up 0.1 m in body frame)
        head_local = np.array([0.2, 0.0, 0.1])
        head_world = np.array(base_pos) + rot_mat @ head_local

        fov    = np.deg2rad(90.0)   # ±45°
        angles = np.linspace(-fov / 2.0, fov / 2.0, T2_LIDAR_RAYS)

        ray_starts, ray_ends = [], []
        for angle in angles:
            local_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
            world_dir = rot_mat @ local_dir
            ray_starts.append(head_world.tolist())
            ray_ends.append((head_world + world_dir * LIDAR_RANGE).tolist())

        results   = p.rayTestBatch(ray_starts, ray_ends, physicsClientId=self._physics_client)
        distances = []
        for res in results:
            hit_id, _, hit_frac, _, _ = res
            # ignore self-hits and ground plane (body id 0)
            if hit_id <= 0 or hit_id == self._robot_id:
                distances.append(1.0)
            else:
                distances.append(float(hit_frac))

        # debug visualisation
        if self.render_mode:
            for i, (rs, re) in enumerate(zip(ray_starts, ray_ends)):
                color = [1, 0, 0] if distances[i] < 1.0 else [0, 1, 0]
                end_pt = [rs[j] + (re[j] - rs[j]) * distances[i] for j in range(3)]
                p.addUserDebugLine(rs, end_pt, lineColorRGB=color, lifeTime=0.1,
                                   physicsClientId=self._physics_client)

        return np.array(distances, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build and return the full observation vector.

        Layout (always 60 floats):
            [0:3]   base_vel
            [3:6]   base_ang_vel
            [6:9]   rpy (roll, pitch, yaw)
            [9:21]  joint_pos (12)
            [21:33] joint_vel (12)
            [33:37] foot_contacts (4)
            [37:40] projected gravity vector (3)
            [40:43] target_vel (3)
            [43]    slope_sin
            [44:57] lidar (terrain 0/1: ones; terrain 2: 13 rays)
            [57:60] goal_relative dx,dy,dist (terrain 0/1: zeros; terrain 2: real)
        """
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        base_vel, base_ang = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._physics_client
        )
        rpy     = p.getEulerFromQuaternion(base_orn)
        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        gravity_base = rot_mat.T @ np.array([0, 0, -1.0])

        joint_pos, joint_vel = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot_id, jid, physicsClientId=self._physics_client)
            joint_pos.append(js[0])
            joint_vel.append(js[1])

        # foot contacts
        contacts = np.zeros(4, dtype=np.float32)
        cps = p.getContactPoints(self._robot_id, physicsClientId=self._physics_client)
        foot_ids: List[int] = []
        for j in range(p.getNumJoints(self._robot_id, physicsClientId=self._physics_client)):
            name = p.getJointInfo(self._robot_id, j, physicsClientId=self._physics_client)[12].decode()
            if "foot" in name.lower() or "toe" in name.lower():
                foot_ids.append(j)
        foot_ids = (foot_ids if len(foot_ids) >= 4 else self._joint_ids[-4:])[:4]
        if cps:
            for cp in cps:
                if cp[3] in foot_ids:
                    contacts[foot_ids.index(cp[3])] = 1.0

        self._prev_pos = np.array(base_pos)
        slope_angle = self._get_slope_pitch_offset()

        # sensor block: 13 lidar + 3 goal = 16 floats total
        lidar_raw = self._get_lidar_obs()          # shape (LIDAR_RAYS,) for T0/1, (T2_LIDAR_RAYS,) for T2

        if self.terrain_id == 2:
            lidar_block = lidar_raw[:T2_LIDAR_RAYS]   # 13 floats
            goal_block  = self._goal_relative()        # 3 floats
        else:
            lidar_block = lidar_raw[:LIDAR_RAYS]       # 16 floats
            goal_block  = np.zeros(GOAL_DIM, dtype=np.float32)

        # pad so total sensor floats = LIDAR_RAYS (16)
        pad = LIDAR_RAYS - len(lidar_block) - len(goal_block)
        sensor_block = np.concatenate([lidar_block,
                                       np.zeros(max(pad, 0), dtype=np.float32),
                                       goal_block])[:LIDAR_RAYS]

        return np.concatenate([
            base_vel, base_ang, rpy,
            joint_pos, joint_vel, contacts,
            gravity_base, self.target_vel, [np.sin(slope_angle)],
            sensor_block,
        ]).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        """Map normalised actions to joint position targets."""
        ACTION_SCALE = 0.30
        for i, jid in enumerate(self._joint_ids):
            lo, hi = self._joint_limits[i]
            target = float(np.clip(STANDING_POSE[i] + action[i] * ACTION_SCALE, lo, hi))
            p.setJointMotorControl2(
                self._robot_id, jid, p.POSITION_CONTROL,
                targetPosition=target, force=TORQUE_LIMIT,
                positionGain=0.30, velocityGain=0.12,
                physicsClientId=self._physics_client,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # REWARD  (flat-terrain formulation used for all terrains; terrain-2 extras)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute the per-step reward.

        Core reward (unchanged flat-terrain formulation):
          + forward_reward   — velocity tracking via Gaussian kernel
          + alive_bonus      — gated on forward motion
          − orient_penalty   — roll / relative-pitch penalties
          − yaw_drift        — heading deviation from initialisation
          − lateral          — side drift + lateral position
          − energy           — torque RMS
          − stillness        — penalise stationary robot
          − height_penalty   — penalise body too low above ground
          − action_smooth    — penalise jerky actions

        Terrain-2 extras:
          + goal_bonus       — +10 per goal reached this step
          + approach_reward  — negative delta-distance to goal (dense)
          − collision_penalty — −2 per step in contact with an obstacle
          − fall_penalty     — −50 on terminal fall
        """
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        base_vel, _ = p.getBaseVelocity(self._robot_id, physicsClientId=self._physics_client)
        rpy = p.getEulerFromQuaternion(base_orn)

        vx, vy, vz = base_vel[0], base_vel[1], base_vel[2]
        target_vx   = self.target_vel[0]

        pitch_offset = self._get_slope_pitch_offset()
        slope_fwd_x  = np.cos(pitch_offset)
        slope_fwd_z  = -np.sin(pitch_offset)

        vx_eff      = vx * slope_fwd_x + vz * slope_fwd_z
        vel_error   = vx_eff - target_vx
        prog_scale  = float(np.clip(vx_eff / (target_vx + 1e-6), 0.0, 1.5))
        fwd_reward  = 3.0 * np.exp(-5.0 * vel_error ** 2) * prog_scale

        # The excessive stillness/alive penalties caused a "suicide policy".
        # We restore a moderate alive bonus so that staying alive (even when standing)
        # is slightly net-positive compared to dying, but moving forward is greatly rewarded.
        alive = self.alive_bonus  # Removed the strict vx_eff > 0.1 gate

        roll, pitch_val, yaw = rpy
        rel_pitch      = pitch_val - pitch_offset
        roll_penalty   = 4.0 * (np.exp(2.0 * abs(roll)) - 1.0)
        pitch_penalty  = 1.5 * (np.exp(1.0 * abs(rel_pitch)) - 1.0)
        orient_penalty = roll_penalty + pitch_penalty

        yaw_drift      = 2.0 * abs(yaw - self._init_yaw)
        lateral        = 0.5 * abs(vy) + 0.8 * abs(base_pos[1])

        torques = np.array([
            p.getJointState(self._robot_id, jid, physicsClientId=self._physics_client)[3]
            for jid in self._joint_ids
        ])
        energy_pen   = self.energy_pen * np.sum(np.square(torques)) / NUM_JOINTS
        # Revert stillness penalty to a smaller value (0.5) to avoid the suicide loop
        stillness    = 0.5 if abs(vx_eff) < 0.05 else 0.0

        ground_z     = self._get_ground_height()
        height_above = base_pos[2] - ground_z
        height_pen   = 3.0 * max(0.0, 0.18 - height_above)

        action_smooth = 0.1 * float(np.sum(np.square(action - self._prev_action)))

        total = (fwd_reward + alive
                 - orient_penalty
                 - yaw_drift
                 - lateral
                 - energy_pen
                 - stillness
                 - height_pen
                 - action_smooth)

        # terrain-2 extras
        if self.terrain_id == 2:
            # goal bonus + dense approach reward
            goal_bonus = self._check_goal()
            total += goal_bonus

            # delta-distance to goal (negative → reward getting closer)
            goal_rel = self._goal_relative()
            cur_dist = goal_rel[2]
            prev_dist_attr = getattr(self, "_prev_goal_dist", cur_dist)
            approach = float(np.clip(prev_dist_attr - cur_dist, -2.0, 2.0))
            self._prev_goal_dist = cur_dist  # type: ignore[attr-defined]
            total += approach

            # collision penalty
            cps = p.getContactPoints(self._robot_id, physicsClientId=self._physics_client)
            if cps:
                for cp in cps:
                    if cp[2] in self._obstacle_ids:
                        total -= T2_PENALTY_COL

        # universal terminal penalty (prevents suicide exploit)
        if self._is_done():
            total -= 50.0

        return float(total)

    # ─────────────────────────────────────────────────────────────────────────
    # TERMINATION
    # ─────────────────────────────────────────────────────────────────────────

    def _is_done(self) -> bool:
        """Return True if the episode should terminate."""
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        rpy = p.getEulerFromQuaternion(base_orn)

        ground_z     = self._get_ground_height()
        height_above = base_pos[2] - ground_z

        # terrain-specific height thresholds
        if self.terrain_id == 0:
            z_min = 0.18
        elif self.terrain_id == 1:
            z_min = 0.10
        else:
            z_min = T2_BASE_HEIGHT_MIN

        if height_above < z_min:
            return True
        if abs(rpy[0]) > np.deg2rad(50):   # roll
            return True

        slope_pitch  = self._get_slope_pitch_offset()
        rel_pitch    = abs(rpy[1] - slope_pitch)
        if rel_pitch > np.deg2rad(50):
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # INFO & CAMERA
    # ─────────────────────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        """Return diagnostic info dict."""
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        info = {
            "x_position":    base_pos[0],
            "step":          self._step_count,
            "terrain_id":    self.terrain_id,
            "terrain_level": self.terrain_id,   # backward-compat
        }
        if self.terrain_id == 2:
            info["goals_reached"] = self._goals_reached
        return info

    def _follow_camera(self) -> None:
        """Move the debug camera to follow the robot."""
        if self._robot_id is None:
            return
        pos, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client
        )
        _, _, yaw_rad = p.getEulerFromQuaternion(orn)
        yaw_deg = float(np.degrees(yaw_rad))
        p.resetDebugVisualizerCamera(
            cameraDistance=self._CAM_DISTANCE,
            cameraYaw=yaw_deg + self._CAM_YAW_OFFSET,
            cameraPitch=self._CAM_PITCH,
            cameraTargetPosition=[pos[0], pos[1], pos[2] + 0.05],
            physicsClientId=self._physics_client,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PHYSICS CLIENT
    # ─────────────────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Connect to a PyBullet physics server."""
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
        mode = p.GUI if self.render_mode else p.DIRECT
        self._physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._physics_client)
        if self.render_mode:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                       physicsClientId=self._physics_client)


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test QuadrupedEnv")
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--terrain", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--steps",   type=int, default=1000)
    args = parser.parse_args()

    env = QuadrupedEnv(terrain_id=args.terrain, render=args.render)
    obs, info = env.reset()
    print(f"obs shape: {obs.shape}, terrain_id: {args.terrain}")
    try:
        for t in range(args.steps):
            action = env.action_space.sample()
            obs, r, done, trunc, info = env.step(action)
            if done or trunc:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
