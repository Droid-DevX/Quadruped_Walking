"""
train_curriculum.py  —  Curriculum Learning for Quadruped SAC
==============================================================
Continues from your existing flat-terrain best model (3 M steps already done).

Curriculum:
  Stage 0  →  flat    (terrain_level=0)  — already done, used as warm-start
  Stage 1  →  slope   (terrain_level=1)  — 10 L steps fine-tune
  Stage 2  →  rough   (terrain_level=2)  — 15 L steps fine-tune
                        rough difficulty increases progressively:
                        0-33%  steps → height 0.00–0.03  (gentle bumps)
                        33-66% steps → height 0.03–0.06  (medium bumps)
                        66-100% steps → height 0.06–0.08 (full difficulty)

Reward fixes applied (standing exploit patch):
  • stillness_penalty raised  1.0 → 3.0
  • alive_bonus gated on vx   (1.5 if vx>0.1 else 0.3)
  • forward_reward multiplied by clipped(vx / target_vx) factor

Usage:
  python train_curriculum.py                        # full curriculum from best model
  python train_curriculum.py --start_stage 2        # jump straight to rough
  python train_curriculum.py --stage1_steps 500000  # custom step counts
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecNormalize,
                                              VecMonitor)

from environment import (NUM_JOINTS, OBS_DIM, ACT_DIM, MAX_EPISODE_STEPS,
                         CONTROL_HZ, PHYSICS_SUBSTEPS, TORQUE_LIMIT,
                         A1_JOINT_LIMITS)

# ---------------------------------------------------------------------------
# PATCHED QuadrupedEnv  (reward fixes + progressive rough terrain)
# ---------------------------------------------------------------------------

class QuadrupedEnvPatched(gym.Env):
    """
    Drop-in replacement for QuadrupedEnv with fixes:
      1. stillness_penalty  1.0 → 3.0
      2. alive_bonus gated on forward velocity
      3. forward_reward scaled by actual vx progress
      4. rough_difficulty (0.0-1.0) controls heightfield max height
         0.0 → max_height=0.03 (gentle)
         1.0 → max_height=0.08 (full difficulty, original)
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    _CAM_DISTANCE   = 2.5
    _CAM_PITCH      = -12.0
    _CAM_YAW_OFFSET = 180.0

    def __init__(self, render=False, terrain_level=0,
                 target_velocity=0.5, energy_penalty=0.008, alive_bonus=1.5,
                 rough_difficulty=0.0):
        super().__init__()
        self.render_mode      = render
        self.terrain_level    = terrain_level
        self.target_vel       = np.array([target_velocity, 0.0, 0.0])
        self.energy_pen       = energy_penalty
        self.alive_bonus      = alive_bonus
        self.rough_difficulty = float(np.clip(rough_difficulty, 0.0, 1.0))

        obs_high = np.inf * np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1., 1., shape=(ACT_DIM,), dtype=np.float32)

        self._physics_client = None
        self._robot_id       = None
        self._step_count     = 0
        self._prev_pos       = np.zeros(3)
        self._prev_action    = np.zeros(ACT_DIM)
        self._joint_ids      = []
        self._joint_limits   = []
        self._init_yaw       = 0.0
        self._connect()

    # ── connect ──────────────────────────────────────────────────────────────
    def _connect(self):
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
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,           0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,       1,
                                       physicsClientId=self._physics_client)
            p.resetDebugVisualizerCamera(
                cameraDistance=self._CAM_DISTANCE, cameraYaw=180.0,
                cameraPitch=self._CAM_PITCH,
                cameraTargetPosition=[0, 0, 0.27],
                physicsClientId=self._physics_client)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        try:
            p.getConnectionInfo(self._physics_client)
        except Exception:
            self._connect()

        p.resetSimulation(physicsClientId=self._physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        p.setTimeStep(1.0 / CONTROL_HZ, physicsClientId=self._physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._physics_client)

        self._load_terrain()

        start_pos = [0, 0, 0.48]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self._robot_id = p.loadURDF(
            'a1/urdf/a1.urdf',
            basePosition=start_pos, baseOrientation=start_orn,
            physicsClientId=self._physics_client)

        self._cache_joint_info()
        self._randomise_init_pose()
        self._step_count  = 0
        self._prev_pos    = np.array(start_pos)
        self._prev_action = np.zeros(ACT_DIM)

        _, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        _, _, self._init_yaw = p.getEulerFromQuaternion(orn)

        STANDING_POSE = [0.0, 0.9, -1.8] * 4
        for i, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot_id, jid, STANDING_POSE[i], 0.0,
                              physicsClientId=self._physics_client)
        for _ in range(200):
            for i, jid in enumerate(self._joint_ids):
                p.setJointMotorControl2(
                    self._robot_id, jid, p.POSITION_CONTROL,
                    targetPosition=STANDING_POSE[i], force=TORQUE_LIMIT,
                    positionGain=1.0, velocityGain=0.2,
                    physicsClientId=self._physics_client)
            p.stepSimulation(physicsClientId=self._physics_client)

        if self.render_mode:
            self._follow_camera()

        return self._get_obs(), {}

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, -1., 1.)
        action = 0.2 * self._prev_action + 0.8 * action
        self._prev_action = action.copy()
        self._apply_action(action)
        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._physics_client)
        self._step_count += 1
        if self.render_mode:
            time.sleep(1.5 / CONTROL_HZ)
            self._follow_camera()
        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_done()
        truncated  = self._step_count >= MAX_EPISODE_STEPS
        info       = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    # ── camera ────────────────────────────────────────────────────────────────
    def _follow_camera(self):
        if self._robot_id is None:
            return
        pos, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        _, _, yaw_rad = p.getEulerFromQuaternion(orn)
        yaw_deg = np.degrees(yaw_rad)
        target  = [pos[0], pos[1], pos[2] + 0.05]
        p.resetDebugVisualizerCamera(
            cameraDistance=self._CAM_DISTANCE,
            cameraYaw=yaw_deg + self._CAM_YAW_OFFSET,
            cameraPitch=self._CAM_PITCH,
            cameraTargetPosition=target,
            physicsClientId=self._physics_client)

    def close(self):
        if self._physics_client is not None:
            try:
                p.disconnect(physicsClientId=self._physics_client)
            except Exception:
                pass
            self._physics_client = None

    def __del__(self):
        self.close()

    def set_terrain_level(self, level):
        self.terrain_level = int(np.clip(level, 0, 2))

    # ── terrain ───────────────────────────────────────────────────────────────
    def _load_terrain(self):
        if self.terrain_level == 0:
            p.loadURDF('plane.urdf', physicsClientId=self._physics_client)
        elif self.terrain_level == 1:
            orn = p.getQuaternionFromEuler([0, np.deg2rad(10), 0])
            p.loadURDF('plane.urdf', [0, 0, 0], orn,
                       physicsClientId=self._physics_client)
        else:
            self._create_heightfield()

    def _create_heightfield(self):
        """
        Progressive difficulty:
          rough_difficulty=0.0  → max bump 0.03 m  (gentle, robot can handle)
          rough_difficulty=0.5  → max bump 0.055 m (medium)
          rough_difficulty=1.0  → max bump 0.08 m  (original full difficulty)
        """
        rows, cols = 64, 64
        min_h      = 0.03
        max_h_full = 0.08
        max_h      = min_h + (max_h_full - min_h) * self.rough_difficulty
        heights    = np.random.uniform(0, max_h,
                                       size=(rows * cols,)).astype(np.float32)
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD, meshScale=[0.1, 0.1, 1.0],
            heightfieldTextureScaling=rows, heightfieldData=heights,
            numHeightfieldRows=rows, numHeightfieldColumns=cols,
            physicsClientId=self._physics_client)
        tid = p.createMultiBody(0, shape,
                                physicsClientId=self._physics_client)
        p.resetBasePositionAndOrientation(
            tid, [0, 0, 0], [0, 0, 0, 1],
            physicsClientId=self._physics_client)

    # ── joints ────────────────────────────────────────────────────────────────
    def _cache_joint_info(self):
        self._joint_ids, self._joint_limits = [], []
        for j in range(p.getNumJoints(self._robot_id,
                                      physicsClientId=self._physics_client)):
            info = p.getJointInfo(self._robot_id, j,
                                  physicsClientId=self._physics_client)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_ids.append(j)
                lo, hi = info[8], info[9]
                idx = len(self._joint_ids) - 1
                if abs(hi - lo) < 1e-6 and idx < len(A1_JOINT_LIMITS):
                    lo, hi = A1_JOINT_LIMITS[idx]
                self._joint_limits.append((lo, hi))
        self._joint_ids    = self._joint_ids[:NUM_JOINTS]
        self._joint_limits = self._joint_limits[:NUM_JOINTS]

    def _randomise_init_pose(self):
        STANDING_POSE = [0.0, 0.9, -1.8] * 4
        noise_scale   = 0.05
        for i, jid in enumerate(self._joint_ids):
            target = STANDING_POSE[i] + np.random.uniform(-noise_scale, noise_scale)
            p.resetJointState(self._robot_id, jid, target,
                              physicsClientId=self._physics_client)

    # ── observation ───────────────────────────────────────────────────────────
    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        base_vel, base_ang = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._physics_client)
        rpy          = p.getEulerFromQuaternion(base_orn)
        rot_mat      = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        gravity_base = rot_mat.T @ np.array([0, 0, -1.])
        joint_pos, joint_vel = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot_id, jid,
                                 physicsClientId=self._physics_client)
            joint_pos.append(js[0])
            joint_vel.append(js[1])
        contacts = self._get_foot_contacts()
        self._prev_pos = np.array(base_pos)
        return np.concatenate([
            base_vel, base_ang, rpy,
            joint_pos, joint_vel, contacts,
            gravity_base, self.target_vel, [self.terrain_level]
        ]).astype(np.float32)

    def _get_foot_contacts(self):
        contacts = np.zeros(4)
        cps = p.getContactPoints(self._robot_id,
                                 physicsClientId=self._physics_client)
        if cps:
            foot_ids = self._get_foot_link_ids()
            for cp in cps:
                if cp[3] in foot_ids:
                    contacts[foot_ids.index(cp[3])] = 1.0
        return contacts

    def _get_foot_link_ids(self):
        foot_ids = []
        for j in range(p.getNumJoints(self._robot_id,
                                      physicsClientId=self._physics_client)):
            name = p.getJointInfo(self._robot_id, j,
                physicsClientId=self._physics_client)[12].decode('utf-8')
            if 'foot' in name.lower() or 'toe' in name.lower():
                foot_ids.append(j)
        return (foot_ids if len(foot_ids) >= 4
                else self._joint_ids[-4:])[:4]

    # ── PATCHED reward ────────────────────────────────────────────────────────
    def _compute_reward(self, action):
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        base_vel, _ = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._physics_client)
        rpy = p.getEulerFromQuaternion(base_orn)

        vx = base_vel[0]
        vy = base_vel[1]
        target_vx = self.target_vel[0]

        # forward reward scaled by actual vx progress
        vel_error      = vx - target_vx
        progress_scale = float(np.clip(vx / (target_vx + 1e-6), 0.0, 1.5))
        forward_reward = 3.0 * np.exp(-5.0 * vel_error ** 2) * progress_scale

        # alive bonus only when moving forward
        alive = self.alive_bonus if vx > 0.1 else 0.3

        # orientation penalty
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        roll_penalty   = 4.0 * (np.exp(2.0 * abs(roll))  - 1.0)
        pitch_penalty  = 1.5 * (np.exp(1.0 * abs(pitch)) - 1.0)
        orient_penalty = roll_penalty + pitch_penalty

        # yaw drift
        yaw_drift_penalty = 2.0 * abs(yaw - self._init_yaw)

        # lateral drift
        lateral_penalty = 0.5 * abs(vy) + 0.8 * abs(base_pos[1])

        # energy
        torques = np.array([
            p.getJointState(self._robot_id, jid,
                physicsClientId=self._physics_client)[3]
            for jid in self._joint_ids])
        energy_penalty = self.energy_pen * np.sum(np.square(torques)) / NUM_JOINTS

        # stronger stillness penalty
        stillness_penalty = 3.0 if abs(vx) < 0.05 else 0.0

        # height
        height         = base_pos[2]
        height_penalty = 3.0 * max(0.0, 0.18 - height)

        # action smoothness
        action_smoothness = 0.1 * np.sum(np.square(action - self._prev_action))

        total = (forward_reward + alive
                 - orient_penalty
                 - yaw_drift_penalty
                 - lateral_penalty
                 - energy_penalty
                 - stillness_penalty
                 - height_penalty
                 - action_smoothness)

        return float(total)

    # ── action ────────────────────────────────────────────────────────────────
    STANDING_POSE_TARGETS = [0.0, 0.9, -1.8] * 4
    ACTION_SCALE          = 0.25

    def _apply_action(self, action):
        for i, jid in enumerate(self._joint_ids):
            lo, hi = self._joint_limits[i]
            target = self.STANDING_POSE_TARGETS[i] + action[i] * self.ACTION_SCALE
            target = float(np.clip(target, lo, hi))
            p.setJointMotorControl2(
                self._robot_id, jid, p.POSITION_CONTROL,
                targetPosition=target, force=TORQUE_LIMIT,
                positionGain=0.25, velocityGain=0.1,
                physicsClientId=self._physics_client)

    # ── termination ───────────────────────────────────────────────────────────
    def _is_done(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        rpy = p.getEulerFromQuaternion(base_orn)
        if base_pos[2] < 0.15:
            return True
        if abs(rpy[0]) > np.deg2rad(50) or abs(rpy[1]) > np.deg2rad(50):
            return True
        return False

    def _get_info(self):
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        return {'x_position': base_pos[0], 'step': self._step_count,
                'terrain_level': self.terrain_level}


# ---------------------------------------------------------------------------
# PROGRESSIVE DIFFICULTY CALLBACK
# ---------------------------------------------------------------------------

class ProgressiveDifficultyCallback(BaseCallback):
    """
    Gradually increases rough_difficulty on all envs as training progresses.

    Schedule:
      0%–33%   → difficulty = 0.0  (max bump 0.03m, gentle, stays flat)
      33%–100% → difficulty ramps linearly 0.0 → 1.0
      100%     → difficulty = 1.0  (max bump 0.08m, full original)
    """
    def __init__(self, total_steps: int, log_freq: int = 50_000, verbose=1):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.log_freq    = log_freq
        self._last_log   = 0

    def _get_difficulty(self) -> float:
        progress = self.num_timesteps / self.total_steps
        if progress < 0.33:
            return 0.0
        return float(np.clip((progress - 0.33) / 0.67, 0.0, 1.0))

    def _on_step(self) -> bool:
        diff = self._get_difficulty()

        # update difficulty on all training envs
        for env in self.training_env.envs:
            inner = env
            while hasattr(inner, 'env'):
                inner = inner.env
            if hasattr(inner, 'rough_difficulty'):
                inner.rough_difficulty = diff

        if (self.num_timesteps - self._last_log) >= self.log_freq:
            max_h = 0.03 + (0.08 - 0.03) * diff
            print(f'  [ProgressiveDifficulty] '
                  f'steps={self.num_timesteps:>8,} | '
                  f'difficulty={diff:.2f} | '
                  f'max_bump={max_h:.3f}m')
            self._last_log = self.num_timesteps
        return True


class CurriculumCallback(BaseCallback):
    """Logs per-episode stats every log_freq steps."""
    def __init__(self, stage_name: str, log_freq: int = 10_000, verbose=1):
        super().__init__(verbose)
        self.stage_name  = stage_name
        self.log_freq    = log_freq
        self._ep_rewards = []
        self._ep_lengths = []
        self._last_log   = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self._ep_rewards.append(info['episode']['r'])
                self._ep_lengths.append(info['episode']['l'])

        if (self.num_timesteps - self._last_log) >= self.log_freq:
            if self._ep_rewards:
                mean_r = np.mean(self._ep_rewards[-20:])
                mean_l = np.mean(self._ep_lengths[-20:])
                print(f'  [{self.stage_name}] '
                      f'steps={self.num_timesteps:>8,} | '
                      f'mean_reward(20ep)={mean_r:>8.1f} | '
                      f'mean_ep_len={mean_l:>6.0f}')
            self._last_log = self.num_timesteps
        return True


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

STAGE_DIRS = {
    1: 'sac_models/curriculum/stage1_slope',
    2: 'sac_models/curriculum/stage2_rough',
}


def make_env(terrain_level: int, seed: int = 0, rough_difficulty: float = 0.0):
    def _init():
        env = QuadrupedEnvPatched(terrain_level=terrain_level,
                                  rough_difficulty=rough_difficulty)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(terrain_level: int, n_envs: int = 1, seed: int = 0,
                  norm_path: str = None, rough_difficulty: float = 0.0):
    vec = DummyVecEnv([make_env(terrain_level, seed + i, rough_difficulty)
                       for i in range(n_envs)])
    vec = VecMonitor(vec)
    if norm_path and os.path.exists(norm_path):
        print(f'  Loading VecNormalize stats from: {norm_path}')
        vec = VecNormalize.load(norm_path, vec)
        vec.training    = True
        vec.norm_reward = True
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec


def load_model_for_stage(stage: int, vec_env, base_model_path: str, device: str):
    if stage == 1:
        model_path = base_model_path
        print(f'\n  [Stage 1] Loading flat-terrain base model: {model_path}')
    else:
        prev_dir   = STAGE_DIRS[stage - 1]
        model_path = os.path.join(prev_dir, 'best_model', 'best_model')
        if not os.path.exists(model_path + '.zip'):
            model_path = os.path.join(prev_dir, f'stage{stage-1}_final')
            print(f'  [Stage {stage}] best_model not found, using: {model_path}')
        else:
            print(f'\n  [Stage {stage}] Loading stage-{stage-1} best model: {model_path}')

    model = SAC.load(model_path, env=vec_env, device=device)
    model.learning_rate = 1e-4
    return model


# ---------------------------------------------------------------------------
# STAGE TRAINING
# ---------------------------------------------------------------------------

def train_stage(stage:           int,
                terrain_level:   int,
                total_steps:     int,
                base_model_path: str,
                base_norm_path:  str,
                device:          str,
                n_envs:          int = 1):

    out_dir = STAGE_DIRS[stage]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'eval'), exist_ok=True)

    stage_name = f'Stage{stage}({"slope" if terrain_level==1 else "rough"})'
    print(f'\n{"="*60}')
    print(f'  {stage_name}  |  terrain={terrain_level}  |  steps={total_steps:,}')
    if terrain_level == 2:
        print(f'  Progressive difficulty: 0.03m → 0.08m bump height')
        print(f'    0%-33%  steps: gentle bumps (0.03m)')
        print(f'    33%-100% steps: ramps up to full (0.08m)')
    print(f'{"="*60}')

    # norm path from previous stage
    if stage == 1:
        prev_norm = base_norm_path
    else:
        prev_dir  = STAGE_DIRS[stage - 1]
        prev_norm = os.path.join(prev_dir, 'vec_normalize.pkl')
        if not os.path.exists(prev_norm):
            prev_norm = base_norm_path
            print(f'  Warning: using fallback norm: {prev_norm}')

    train_env = build_vec_env(terrain_level, n_envs=n_envs,
                              norm_path=prev_norm, rough_difficulty=0.0)
    eval_env  = build_vec_env(terrain_level, n_envs=1,
                              norm_path=prev_norm, rough_difficulty=0.5)
    eval_env.training    = False
    eval_env.norm_reward = False

    model = load_model_for_stage(stage, train_env, base_model_path, device)

    checkpoint_cb = CheckpointCallback(
        save_freq        = max(50_000 // n_envs, 1),
        save_path        = os.path.join(out_dir, 'checkpoints'),
        name_prefix      = f'stage{stage}',
        save_vecnormalize= True,
        verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(out_dir, 'best_model'),
        log_path             = os.path.join(out_dir, 'eval'),
        eval_freq            = max(25_000 // n_envs, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        verbose              = 1)

    curriculum_cb = CurriculumCallback(stage_name, log_freq=10_000)
    callbacks     = [checkpoint_cb, eval_cb, curriculum_cb]

    if terrain_level == 2:
        prog_cb = ProgressiveDifficultyCallback(
            total_steps=total_steps, log_freq=50_000)
        callbacks.append(prog_cb)

    t0 = time.time()
    model.learn(
        total_timesteps     = total_steps,
        callback            = callbacks,
        reset_num_timesteps = False,
        progress_bar        = True)

    elapsed = time.time() - t0
    print(f'\n  {stage_name} finished in {elapsed/60:.1f} min')

    final_path = os.path.join(out_dir, f'stage{stage}_final')
    model.save(final_path)
    train_env.save(os.path.join(out_dir, 'vec_normalize.pkl'))
    print(f'  Saved final model  : {final_path}.zip')
    print(f'  Saved VecNormalize : {out_dir}/vec_normalize.pkl')

    train_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',   default='sac_models/best/best_model')
    parser.add_argument('--base_norm',    default='sac_models/best/vec_normalize.pkl')
    parser.add_argument('--start_stage',  type=int, default=1)
    parser.add_argument('--stage1_steps', type=int, default=1_000_000)
    parser.add_argument('--stage2_steps', type=int, default=1_500_000)
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--n_envs',       type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.base_model + '.zip'):
        raise FileNotFoundError(f'Base model not found: {args.base_model}.zip')
    if not os.path.exists(args.base_norm):
        raise FileNotFoundError(f'Norm stats not found: {args.base_norm}')

    print(f'\n  Base model   : {args.base_model}.zip')
    print(f'  Norm stats   : {args.base_norm}')
    print(f'  Start stage  : {args.start_stage}')
    print(f'  Stage 1 steps: {args.stage1_steps:,}  (slope)')
    print(f'  Stage 2 steps: {args.stage2_steps:,}  (rough, progressive difficulty)')
    print(f'  Device       : {args.device}')

    for stage in [1, 2]:
        if stage < args.start_stage:
            print(f'\n  Skipping stage {stage}')
            continue
        terrain_level = stage  # stage 1 = terrain 1, stage 2 = terrain 2
        steps         = args.stage1_steps if stage == 1 else args.stage2_steps
        train_stage(
            stage           = stage,
            terrain_level   = terrain_level,
            total_steps     = steps,
            base_model_path = args.base_model,
            base_norm_path  = args.base_norm,
            device          = args.device,
            n_envs          = args.n_envs)

    print('\n  ✓  Curriculum training complete!')
    print('\n  Test with:')
    print('    python test_sac.py --terrain 2 --episodes 5 '
          '--model sac_models/curriculum/stage2_rough/best_model/best_model '
          '--norm  sac_models/curriculum/stage2_rough/vec_normalize.pkl')


if __name__ == '__main__':
    main()