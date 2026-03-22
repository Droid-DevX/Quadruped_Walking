import argparse
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

# =============================================================================
# CONSTANTS
# =============================================================================

NUM_JOINTS        = 12
OBS_DIM           = 44       # 3+3+3+12+12+4+3+3+1 = 44
ACT_DIM           = NUM_JOINTS
MAX_EPISODE_STEPS = 1000
CONTROL_HZ        = 60       # control loop frequency
PHYSICS_SUBSTEPS  = 4        # physics steps per control step -> effective 240 Hz
TORQUE_LIMIT      = 33.5     # Unitree A1 motor max torque (Nm)

# Real Unitree A1 joint limits (radians).
A1_JOINT_LIMITS = [
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



# ENVIRONMENT


class QuadrupedEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False, terrain_level=0,
                 target_velocity=0.5, energy_penalty=0.008, alive_bonus=1.5):
        super().__init__()
        self.render_mode   = render
        self.terrain_level = terrain_level
        
        self.target_vel    = np.array([target_velocity, 0.0, 0.0])
        self.energy_pen    = energy_penalty
        
        self.alive_bonus   = alive_bonus

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

    # ── CONNECT 
    def _connect(self):
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
        mode = p.GUI if self.render_mode else p.DIRECT
        self._physics_client = p.connect(mode)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._physics_client)

        if self.render_mode:
            # Disable GUI panels and mouse-camera override so our
            # _follow_camera() is never hijacked by user mouse drag.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,           0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,       1,
                                       physicsClientId=self._physics_client)
            # Lock initial view to TPP behind-view before anything loads
            p.resetDebugVisualizerCamera(
                cameraDistance=self._CAM_DISTANCE,
                cameraYaw=180.0,
                cameraPitch=self._CAM_PITCH,
                cameraTargetPosition=[0, 0, 0.27],
                physicsClientId=self._physics_client)

    # ── RESET 
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

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._physics_client)

        self._load_terrain()

        start_pos = [0, 0, 0.48]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self._robot_id = p.loadURDF(
            'a1/urdf/a1.urdf',
            basePosition=start_pos,
            baseOrientation=start_orn,
            physicsClientId=self._physics_client)

        self._cache_joint_info()
        self._randomise_init_pose()
        self._step_count  = 0
        self._prev_pos    = np.array(start_pos)
        self._prev_action = np.zeros(ACT_DIM)

        # Record the yaw at reset so we can penalise drift from it
        _, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        _, _, self._init_yaw = p.getEulerFromQuaternion(orn)

      
        STANDING_POSE = [0.0, 0.9, -1.8] * 4
        # Step 1: teleport joints to standing pose instantly (no physics)
        for i, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot_id, jid, STANDING_POSE[i], 0.0,
                              physicsClientId=self._physics_client)
        # Step 2: hold with strong PD and simulate until stable
        for _ in range(200):
            for i, jid in enumerate(self._joint_ids):
                p.setJointMotorControl2(
                    self._robot_id, jid, p.POSITION_CONTROL,
                    targetPosition=STANDING_POSE[i], force=TORQUE_LIMIT,
                    positionGain=1.0, velocityGain=0.2,
                    physicsClientId=self._physics_client)
            p.stepSimulation(physicsClientId=self._physics_client)

        pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        print(f'  [env] Reset height: {pos[2]:.3f} m')

        if self.render_mode:
            self._follow_camera()

        return self._get_obs(), {}

    # ── STEP 
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

  
    _CAM_DISTANCE   = 2.5    # metres behind robot (wider view to see whole body)
    _CAM_PITCH      = -12.0  # slight downward angle — natural game TPP feel
    _CAM_YAW_OFFSET = 180.0  

    def _follow_camera(self):
        """TPP camera that locks strictly behind the robot every single step.
        Yaw tracks the robot's own heading so the view always faces forward."""
        if self._robot_id is None:
            return
        pos, orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        _, _, yaw_rad = p.getEulerFromQuaternion(orn)
        yaw_deg = np.degrees(yaw_rad)
        # Target slightly above the robot center so body is in mid-frame
        target = [pos[0], pos[1], pos[2] + 0.05]
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

    # ── TERRAIN ───────────────────────────────────────────────────────────────
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
        rows, cols = 64, 64
        heights = np.random.uniform(0, 0.08, size=(rows * cols,)).astype(np.float32)
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD, meshScale=[0.1, 0.1, 1.0],
            heightfieldTextureScaling=rows, heightfieldData=heights,
            numHeightfieldRows=rows, numHeightfieldColumns=cols,
            physicsClientId=self._physics_client)
        tid = p.createMultiBody(0, shape, physicsClientId=self._physics_client)
        p.resetBasePositionAndOrientation(
            tid, [0, 0, 0], [0, 0, 0, 1],
            physicsClientId=self._physics_client)

    # ── JOINTS ────────────────────────────────────────────────────────────────
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
        STANDING_POSE = [
             0.0,  0.9, -1.8,   # Front-Left
             0.0,  0.9, -1.8,   # Front-Right
             0.0,  0.9, -1.8,   # Rear-Left
             0.0,  0.9, -1.8,   # Rear-Right
        ]
        noise_scale = 0.05
        for i, jid in enumerate(self._joint_ids):
            target = STANDING_POSE[i] + np.random.uniform(-noise_scale, noise_scale)
            p.resetJointState(self._robot_id, jid, target,
                              physicsClientId=self._physics_client)

    # ── OBSERVATION ───────────────────────────────────────────────────────────
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
        return (foot_ids if len(foot_ids) >= 4 else self._joint_ids[-4:])[:4]

    # ── REWARD ────────────────────────────────────────────────────────────────
    def _compute_reward(self, action):
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        base_vel, _ = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._physics_client)
        rpy = p.getEulerFromQuaternion(base_orn)

        vx = base_vel[0]
        vy = base_vel[1]

        # ── Forward velocity reward 
    
        target_vx      = self.target_vel[0]
        vel_error      = vx - target_vx
        forward_reward = 3.0 * np.exp(-5.0 * vel_error ** 2)

        # ── Alive bonus 
    
        alive = self.alive_bonus

        # ── Orientation penalty 
    
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        roll_penalty  = 4.0 * (np.exp(2.0 * abs(roll))  - 1.0)
        pitch_penalty = 1.5 * (np.exp(1.0 * abs(pitch)) - 1.0)
        orient_penalty = roll_penalty + pitch_penalty

        # ── Yaw drift penalty 
        # Penalise curving/spinning so robot walks straight.
        yaw_drift_penalty = 2.0 * abs(yaw - self._init_yaw)

        # ── Lateral drift penalty 
        lateral_penalty = 0.5 * abs(vy) + 0.8 * abs(base_pos[1])

        # ── Energy penalty 
        torques = np.array([
            p.getJointState(self._robot_id, jid,
                physicsClientId=self._physics_client)[3]
            for jid in self._joint_ids])
        energy_penalty = self.energy_pen * np.sum(np.square(torques)) / NUM_JOINTS

        # ── Stillness penalty 
        # Stronger penalty for not moving forward at all.
        stillness_penalty = 1.0 if abs(vx) < 0.05 else 0.0

        # ── Height penalty 
        # Penalise crouching. A1 settles at ~0.27-0.35 m with pose [0,0.9,-1.8].
        # Penalise if body drops below 0.22 m (clearly crouching/falling).
        height = base_pos[2]
        height_penalty = 3.0 * max(0.0, 0.18 - height)

        # ── Action smoothness penalty 
       
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

    # ── ACTION 
  
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
              
                positionGain=0.25,
                velocityGain=0.1,
                physicsClientId=self._physics_client)

    # ── TERMINATION 
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



# SANITY CHECK


def run_sanity_check(render=True, n_steps=20000):
    env = QuadrupedEnv(render=render)
    obs, info = env.reset()

    print(f'Observation shape : {obs.shape}')
    print(f'Action space      : {env.action_space}')
    print(f'Obs space         : {env.observation_space}')
    print(f'(Ctrl+C to stop)\n')

    rewards = []
    ep_len  = 0
    ep_num  = 1

    try:
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            ep_len += 1

            if terminated or truncated:
                reason = 'truncated (time limit)' if truncated else 'terminated (fell)'
                print(f'  Episode {ep_num:3d} | length: {ep_len:4d} steps '
                      f'| reward: {reward:7.3f} | {reason}')
                ep_len = 0
                ep_num += 1
                obs, info = env.reset()

    except KeyboardInterrupt:
        print('\n  Stopped.')

    env.close()

    if not rewards:
        print('No rewards collected.')
        return

    print(f'\n  Done! Mean reward (random policy): {np.mean(rewards):.3f}')

    plt.figure(figsize=(10, 3))
    plt.plot(rewards, color='#4dabf7', linewidth=1)
    plt.axhline(np.mean(rewards), color='#ff6b6b', linestyle='--',
                label=f'Mean: {np.mean(rewards):.2f}')
    plt.title('Rewards — Random Policy')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    run_sanity_check(render=args.render, n_steps=args.steps)