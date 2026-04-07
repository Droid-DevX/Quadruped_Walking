import argparse
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


# CONSTANTS


NUM_JOINTS        = 12
OBS_DIM           = 44
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

    # Default standing joint positions (radians) 
    DEFAULT_JOINT_POS = [0.0, 0.9, -1.8] * 4

    # Reward weights 
    # Positive terms
    W_LIN_VEL   = 2.0    # exponential XY velocity tracking
    W_ANG_VEL   = 0.5    # exponential yaw-rate tracking
    # Penalty terms (subtracted)
    W_HEIGHT     = 2.0   # squared height deviation from target
    W_POSE_SIM   = 0.15  # squared joint deviation from default pose
    W_ACT_RATE   = 0.08  # squared consecutive action difference (smoothness)
    W_LIN_VEL_Z  = 1.5   # squared vertical base velocity
    W_ROLL_PITCH = 1.0   # squared roll + squared pitch
    W_ENERGY     = 0.005 # torque^2 energy cost per joint
    W_ALIVE      = 0.5   # small alive bonus per step

    #  Termination thresholds 
    ROLL_TH   = np.deg2rad(45)   # 45 deg
    PITCH_TH  = np.deg2rad(45)   # FIX: raised 40→45 so slope pitch doesn't falsely terminate
    Z_MIN_FLAT  = 0.18           # metres  — below this = fallen (flat terrain)
    Z_MIN_SLOPE = 0.10           # FIX: lowered 0.12→0.10 for tilted-plane spawns

    def __init__(self, render=False, terrain_level=0,
                 cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0,
                 target_height=0.30):
        super().__init__()
        self.render_mode    = render
        self.terrain_level  = terrain_level

        self.cmd_vx         = cmd_vx
        self.cmd_vy         = cmd_vy
        self.cmd_wz         = cmd_wz
        self.target_height  = target_height

        self.target_vel     = np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32)

        obs_high = np.inf * np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1., 1., shape=(ACT_DIM,), dtype=np.float32)

        self._physics_client = None
        self._robot_id       = None
        self._step_count     = 0
        self._prev_pos       = np.zeros(3)
        self._prev_action    = np.zeros(ACT_DIM)
        self._last_raw_action = np.zeros(ACT_DIM)
        self._joint_ids      = []
        self._joint_limits   = []
        self._slope_angle    = 0.0
        self._connect()

    #  CONNECT 
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
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,           0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0,
                                       physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,       1,
                                       physicsClientId=self._physics_client)
            p.resetDebugVisualizerCamera(
                cameraDistance=self._CAM_DISTANCE,
                cameraYaw=180.0,
                cameraPitch=self._CAM_PITCH,
                cameraTargetPosition=[0, 0, 0.27],
                physicsClientId=self._physics_client)

    #  RESET 
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

        spawn_x, spawn_y = 0.0, 0.0
        ground_z_at_spawn = self._get_ground_height_at(spawn_x, spawn_y)

        
        if self.terrain_level == 1:
            # Lean the robot forward to match the slope so feet land flat on surface
            start_orn = p.getQuaternionFromEuler([0, -self._slope_angle, 0])
            # FIX: Spawn higher on slope because tilted body needs more clearance
            start_pos = [spawn_x, spawn_y, ground_z_at_spawn + 0.55]
        else:
            start_orn = p.getQuaternionFromEuler([0, 0, 0])
            start_pos = [spawn_x, spawn_y, ground_z_at_spawn + 0.48]

        self._robot_id = p.loadURDF(
            'a1/urdf/a1.urdf',
            basePosition=start_pos,
            baseOrientation=start_orn,
            physicsClientId=self._physics_client)

        self._cache_joint_info()
        self._randomise_init_pose()
        self._step_count      = 0
        self._prev_pos        = np.array(start_pos)
        self._prev_action     = np.zeros(ACT_DIM)
        self._last_raw_action = np.zeros(ACT_DIM)

        STANDING_POSE = [0.0, 0.9, -1.8] * 4
        for i, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot_id, jid, STANDING_POSE[i], 0.0,
                              physicsClientId=self._physics_client)

        # FIX: Use more settle steps on slope to let the robot stabilise
        settle_steps = 300 if self.terrain_level == 1 else 200
        for _ in range(settle_steps):
            for i, jid in enumerate(self._joint_ids):
                p.setJointMotorControl2(
                    self._robot_id, jid, p.POSITION_CONTROL,
                    targetPosition=STANDING_POSE[i], force=TORQUE_LIMIT,
                    positionGain=1.0, velocityGain=0.2,
                    physicsClientId=self._physics_client)
            p.stepSimulation(physicsClientId=self._physics_client)

        pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        print(f'  [env] Reset height: {pos[2]:.3f} m  (ground_z={ground_z_at_spawn:.3f} m)')

        return self._get_obs(), {}

    #  STEP 
    def step(self, action):
        raw_action = np.clip(action, -1., 1.)

        smoothed_action = 0.2 * self._prev_action + 0.8 * raw_action

        action_rate_delta = raw_action - self._last_raw_action
        self._last_raw_action = raw_action.copy()
        self._prev_action     = smoothed_action.copy()

        self._apply_action(smoothed_action)

        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._physics_client)

        self._step_count += 1

        if self.render_mode:
            time.sleep(PHYSICS_SUBSTEPS / CONTROL_HZ)

        obs        = self._get_obs()
        reward, reward_info = self._compute_reward(action_rate_delta)
        terminated = self._is_done()
        truncated  = self._step_count >= MAX_EPISODE_STEPS
        info       = self._get_info()
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    _CAM_DISTANCE   = 2.5
    _CAM_PITCH      = -12.0
    _CAM_YAW_OFFSET = 180.0

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

    #  GROUND HEIGHT 
    def _slope_ground_z(self, x):
        """Analytical ground height on the tilted plane at x-position."""
        return -x * np.tan(self._slope_angle)

    def _get_ground_height_at(self, x, y):
        if self.terrain_level == 1:
            return self._slope_ground_z(x)
        ray_start = [x, y, 10.0]
        ray_end   = [x, y, -10.0]
        result = p.rayTest(ray_start, ray_end,
                           physicsClientId=self._physics_client)
        if result:
            for hit in result:
                if hit[0] >= 0:
                    return hit[3][2]
        return 0.0

    def _get_ground_height(self):
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        if self.terrain_level == 1:
            return self._slope_ground_z(base_pos[0])
        ray_start = [base_pos[0], base_pos[1], base_pos[2] + 2.0]
        ray_end   = [base_pos[0], base_pos[1], base_pos[2] - 5.0]
        result = p.rayTest(ray_start, ray_end,
                           physicsClientId=self._physics_client)
        if result:
            for hit in result:
                if hit[0] != self._robot_id and hit[0] >= 0:
                    return hit[3][2]
        return 0.0

    #  TERRAIN 
    def _load_terrain(self):
        self._slope_angle = 0.0
        if self.terrain_level == 0:
            p.loadURDF('plane.urdf', physicsClientId=self._physics_client)
        elif self.terrain_level == 1:
            
            self._slope_angle = np.deg2rad(10)
            orn = p.getQuaternionFromEuler([0, self._slope_angle, 0])
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

    # ── JOINTS 
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
             0.0,  0.9, -1.8,
             0.0,  0.9, -1.8,
             0.0,  0.9, -1.8,
             0.0,  0.9, -1.8,
        ]
        noise_scale = 0.05
        for i, jid in enumerate(self._joint_ids):
            target = STANDING_POSE[i] + np.random.uniform(-noise_scale, noise_scale)
            p.resetJointState(self._robot_id, jid, target,
                              physicsClientId=self._physics_client)

    # ── OBSERVATION 
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

        # Encode slope as sin/cos — replaces the old terrain_level scalar.
        # Flat:  sin=0, cos=1  |  10° slope: sin≈0.174, cos≈0.985
        # Keeps OBS_DIM=44 so stage-1 weights load without errors.
        slope_sin = np.sin(self._slope_angle)  # 0 on flat, sin(θ) on slope

        return np.concatenate([
            base_vel, base_ang, rpy,
            joint_pos, joint_vel, contacts,
            gravity_base, self.target_vel,
            [slope_sin],             
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

    # ── REWARD 
    def _compute_reward(self, action_rate_delta):
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        base_vel, base_ang_vel = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._physics_client)
        rpy = p.getEulerFromQuaternion(base_orn)

        vx, vy, vz = base_vel[0], base_vel[1], base_vel[2]
        wx, wy, wz = base_ang_vel[0], base_ang_vel[1], base_ang_vel[2]
        roll, pitch = rpy[0], rpy[1]

        ground_z     = self._get_ground_height()
        height_above = base_pos[2] - ground_z

        if self.terrain_level == 1:
            # Forward direction along the slope surface (unit vector)
            slope_fwd_x = np.cos(self._slope_angle)
            slope_fwd_z = np.sin(self._slope_angle)
            # Project world velocity onto slope-forward direction
            vx_slope = vx * slope_fwd_x + vz * slope_fwd_z
            v_xy_ref = np.array([self.cmd_vx, self.cmd_vy])
            v_xy     = np.array([vx_slope, vy])
        else:
            v_xy_ref = np.array([self.cmd_vx, self.cmd_vy])
            v_xy     = np.array([vx, vy])

        lin_vel_err    = np.sum((v_xy_ref - v_xy) ** 2)
        r_lin_vel      = self.W_LIN_VEL * np.exp(-2.0 * lin_vel_err)

        # ── 2. Angular velocity tracking (yaw rate) 
        ang_vel_err    = (self.cmd_wz - wz) ** 2
        r_ang_vel      = self.W_ANG_VEL * np.exp(-2.0 * ang_vel_err)

        # ── 3. Height penalty (relative to ground) 
        p_height       = self.W_HEIGHT * (height_above - self.target_height) ** 2

        # ── 4. Pose similarity penalty 
        joint_pos = np.array([
            p.getJointState(self._robot_id, jid,
                physicsClientId=self._physics_client)[0]
            for jid in self._joint_ids])
        q_default      = np.array(self.DEFAULT_JOINT_POS)
        p_pose_sim     = self.W_POSE_SIM * np.sum((joint_pos - q_default) ** 2)

        # ── 5. Action rate penalty 
        p_action_rate  = self.W_ACT_RATE * np.sum(action_rate_delta ** 2)

        if self.terrain_level == 1:
            # Expected vz from climbing the slope at cmd_vx speed
            expected_vz = self.cmd_vx * np.sin(self._slope_angle)
            excess_vz   = vz - expected_vz
            p_lin_vel_z = self.W_LIN_VEL_Z * excess_vz ** 2
        else:
            p_lin_vel_z = self.W_LIN_VEL_Z * vz ** 2

        slope_pitch    = self._get_slope_pitch_offset()
        relative_pitch = pitch - slope_pitch
        p_roll_pitch   = self.W_ROLL_PITCH * (roll ** 2 + relative_pitch ** 2)

        # ── 8. Energy efficiency penalty 
        torques = np.array([
            p.getJointState(self._robot_id, jid,
                physicsClientId=self._physics_client)[3]
            for jid in self._joint_ids])
        p_energy       = self.W_ENERGY * np.sum(torques ** 2)

        # ── 9. Alive bonus 
        r_alive        = self.W_ALIVE

        # ── Combine 
        total = (r_lin_vel + r_ang_vel + r_alive
                 - p_height
                 - p_pose_sim
                 - p_action_rate
                 - p_lin_vel_z
                 - p_roll_pitch
                 - p_energy)

        reward_info = {
            'r_lin_vel'    : float(r_lin_vel),
            'r_ang_vel'    : float(r_ang_vel),
            'r_alive'      : float(r_alive),
            'p_height'     : float(-p_height),
            'p_pose_sim'   : float(-p_pose_sim),
            'p_action_rate': float(-p_action_rate),
            'p_lin_vel_z'  : float(-p_lin_vel_z),
            'p_roll_pitch' : float(-p_roll_pitch),
            'p_energy'     : float(-p_energy),
            'reward_total' : float(total),
        }
        return float(total), reward_info

    # ── ACTION 
    STANDING_POSE_TARGETS = [0.0, 0.9, -1.8] * 4
    
    ACTION_SCALE          = 0.30

    def _apply_action(self, action):
        for i, jid in enumerate(self._joint_ids):
            lo, hi = self._joint_limits[i]
            target = self.STANDING_POSE_TARGETS[i] + action[i] * self.ACTION_SCALE
            target = float(np.clip(target, lo, hi))
            p.setJointMotorControl2(
                self._robot_id, jid, p.POSITION_CONTROL,
                targetPosition=target, force=TORQUE_LIMIT,
               
                positionGain=0.30,
                velocityGain=0.12,
                physicsClientId=self._physics_client)

    # ── TERMINATION 
    def _get_slope_pitch_offset(self):
        """Return the terrain slope angle (pitch) under the robot."""
        if self.terrain_level == 1:
            return -self._slope_angle
        if self.terrain_level == 0:
            return 0.0
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        dx = 0.3
        fwd_start = [base_pos[0] + dx, base_pos[1], base_pos[2] + 2.0]
        fwd_end   = [base_pos[0] + dx, base_pos[1], base_pos[2] - 5.0]
        fwd_result = p.rayTest(fwd_start, fwd_end,
                               physicsClientId=self._physics_client)
        fwd_z = 0.0
        if fwd_result:
            for hit in fwd_result:
                if hit[0] != self._robot_id and hit[0] >= 0:
                    fwd_z = hit[3][2]; break
        bwd_start = [base_pos[0] - dx, base_pos[1], base_pos[2] + 2.0]
        bwd_end   = [base_pos[0] - dx, base_pos[1], base_pos[2] - 5.0]
        bwd_result = p.rayTest(bwd_start, bwd_end,
                               physicsClientId=self._physics_client)
        bwd_z = 0.0
        if bwd_result:
            for hit in bwd_result:
                if hit[0] != self._robot_id and hit[0] >= 0:
                    bwd_z = hit[3][2]; break
        return np.arctan2(fwd_z - bwd_z, 2 * dx)

    def _is_done(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        rpy = p.getEulerFromQuaternion(base_orn)

        ground_z     = self._get_ground_height()
        height_above = base_pos[2] - ground_z

        z_min = self.Z_MIN_FLAT if self.terrain_level == 0 else self.Z_MIN_SLOPE

        if height_above < z_min:
            return True
        if abs(rpy[0]) > self.ROLL_TH:
            return True

        slope_pitch    = self._get_slope_pitch_offset()
        relative_pitch = abs(rpy[1] - slope_pitch)
        if relative_pitch > self.PITCH_TH:
            return True
        return False

    def _get_info(self):
        base_pos, _ = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._physics_client)
        base_vel, base_ang = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._physics_client)
        ground_z = self._get_ground_height()
        return {
            'x_position'      : base_pos[0],
            'height'          : base_pos[2],
            'height_above_gnd': base_pos[2] - ground_z,
            'ground_z'        : ground_z,
            'vx'              : base_vel[0],
            'vy'              : base_vel[1],
            'wz'              : base_ang[2],
            'step'            : self._step_count,
            'terrain_level'   : self.terrain_level,
        }



# SANITY CHECK


def run_sanity_check(render=True, n_steps=20000):
    env = QuadrupedEnv(render=render, cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0)
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
