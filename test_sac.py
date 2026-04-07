import argparse
import math
import os
import threading
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import CONTROL_HZ, QuadrupedEnv



# INTERACTIVE CAMERA CONTROLLER


class InteractiveCameraController:
    """
    Keyboard + mouse camera controller for PyBullet GUI.

    Controls
    --------
      A / D  or  ← / →   yaw  left / right        (2 deg/frame)
      W / S  or  ↑ / ↓   pitch up / down           (2 deg/frame)
      Q / E              zoom in / out             (0.1 m/frame)
      Scroll wheel       zoom in / out             (0.5 m/tick)
      Left-drag mouse    free orbit                (0.3 deg/px)
      Right-drag mouse   pan camera target XY      (0.005 m/px)
      R                  reset to defaults
    """

    def __init__(
        self,
        physics_client: int,
        distance: float          = 1.5,
        yaw: float               = 45.0,
        pitch: float             = -20.0,
        target: list             = None,
        yaw_speed: float         = 2.0,
        pitch_speed: float       = 2.0,
        zoom_speed: float        = 0.1,
        mouse_sensitivity: float = 0.3,
        pan_sensitivity: float   = 0.005,
        pitch_limits: tuple      = (-89.0, 0.0),
        distance_limits: tuple   = (0.3, 10.0),
        poll_hz: float           = 60.0,
    ):
        self._pc      = physics_client
        self.distance = distance
        self.yaw      = yaw
        self.pitch    = pitch
        self.target   = list(target or [0.0, 0.0, 0.15])

        self._yaw_speed   = yaw_speed
        self._pitch_speed = pitch_speed
        self._zoom_speed  = zoom_speed
        self._mouse_sens  = mouse_sensitivity
        self._pan_sens    = pan_sensitivity

        self._pitch_min, self._pitch_max = pitch_limits
        self._dist_min,  self._dist_max  = distance_limits

        # mouse drag state
        self._left_drag    = False
        self._right_drag   = False
        self._prev_mouse_x = 0
        self._prev_mouse_y = 0

        # key -> (attribute, delta)
        self._KEY_MAP = {
            p.B3G_LEFT_ARROW:  ('yaw',      -self._yaw_speed),
            ord('a'):          ('yaw',      -self._yaw_speed),
            p.B3G_RIGHT_ARROW: ('yaw',       self._yaw_speed),
            ord('d'):          ('yaw',       self._yaw_speed),
            p.B3G_UP_ARROW:    ('pitch',    -self._pitch_speed),
            ord('w'):          ('pitch',    -self._pitch_speed),
            p.B3G_DOWN_ARROW:  ('pitch',     self._pitch_speed),
            ord('s'):          ('pitch',     self._pitch_speed),
            ord('q'):          ('distance', -self._zoom_speed),
            ord('e'):          ('distance',  self._zoom_speed),
        }

        self._running       = False
        self._thread        = None
        self._poll_interval = 1.0 / poll_hz

        # store defaults for R-reset
        self._defaults = dict(
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            target=list(self.target),
        )

        self._apply()   # push initial view immediately

    #  public API 

    def start(self):
        """Spawn a daemon thread that polls input at poll_hz."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Shut down the background thread cleanly."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def update(self):
        """
        Synchronous alternative to start() — call once per sim step:

            cam.update()
        """
        self._poll_once()

    def reset(self):
        """Restore camera to the values passed at construction time."""
        self.distance = self._defaults['distance']
        self.yaw      = self._defaults['yaw']
        self.pitch    = self._defaults['pitch']
        self.target   = list(self._defaults['target'])
        self._apply()

    def follow(self, position: list, height_offset: float = 0.15):
        """
        Keep the camera target centred on a moving body while still letting
        the user orbit and zoom freely.  Call every sim step:

            pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=pc)
            cam.follow(pos)
        """
        self.target = [
            position[0],
            position[1],
            position[2] + height_offset,
        ]
        self._apply()

    # ── internals 

    def _apply(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target,
            physicsClientId=self._pc,
        )

    def _clamp(self):
        self.pitch    = max(self._pitch_min, min(self._pitch_max, self.pitch))
        self.distance = max(self._dist_min,  min(self._dist_max,  self.distance))

    def _poll_once(self):
        # ── keyboard 
        try:
            keys = p.getKeyboardEvents(physicsClientId=self._pc)
        except Exception:
            return

        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            self.reset()
            return

        changed = False
        for key, (attr, delta) in self._KEY_MAP.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                setattr(self, attr, getattr(self, attr) + delta)
                changed = True

        # ── mouse 
        try:
            mouse_events = p.getMouseEvents(physicsClientId=self._pc)
        except Exception:
            mouse_events = []

        for event in mouse_events:
            event_type, mx, my, btn, state = event

            # scroll → zoom
            if event_type == p.MOUSE_WHEEL_SCROLL and btn == 0:
                self.distance -= state * 0.5
                changed = True

            # button press / release
            if event_type == p.MOUSE_BUTTON_EVENT:
                if btn == 0:
                    self._left_drag = bool(state)
                elif btn == 2:
                    self._right_drag = bool(state)
                self._prev_mouse_x = mx
                self._prev_mouse_y = my

            # drag movement
            if event_type == p.MOUSE_MOVE_EVENT:
                dx = mx - self._prev_mouse_x
                dy = my - self._prev_mouse_y

                if self._left_drag:          # orbit
                    self.yaw   += dx * self._mouse_sens
                    self.pitch += dy * self._mouse_sens
                    changed = True

                if self._right_drag:         # pan target XY
                    yaw_rad = math.radians(self.yaw)
                    self.target[0] -= (
                        dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
                    ) * self._pan_sens
                    self.target[1] -= (
                        dx * math.sin(yaw_rad) - dy * math.cos(yaw_rad)
                    ) * self._pan_sens
                    changed = True

                self._prev_mouse_x = mx
                self._prev_mouse_y = my

        if changed:
            self._clamp()
            self._apply()

    def _poll_loop(self):
        while self._running:
            try:
                self._poll_once()
            except Exception:
                pass
            time.sleep(self._poll_interval)



# HELPERS


def _unwrap_raw(eval_env):
    """Peel VecNormalize / DummyVecEnv wrappers to reach QuadrupedEnv."""
    env = eval_env
    for _ in range(10):
        if hasattr(env, 'envs'):
            raw = env.envs[0]
            while hasattr(raw, 'env'):
                raw = raw.env
            return raw
        elif hasattr(env, 'venv'):
            env = env.venv
    return None


def _capture_frame(raw_env, width=960, height=540):
    if raw_env is None or raw_env._robot_id is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    pos, orn = p.getBasePositionAndOrientation(
        raw_env._robot_id, physicsClientId=raw_env._physics_client)
    _, _, yaw_rad = p.getEulerFromQuaternion(orn)

    target = [pos[0], pos[1], pos[2] + 0.05]
    yaw    = np.degrees(yaw_rad) + raw_env._CAM_YAW_OFFSET

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=raw_env._CAM_DISTANCE,
        yaw=yaw,
        pitch=raw_env._CAM_PITCH,
        roll=0,
        upAxisIndex=2,
        physicsClientId=raw_env._physics_client)

    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height,
        nearVal=0.1, farVal=100,
        physicsClientId=raw_env._physics_client)

    _, _, rgb, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=raw_env._physics_client)

    return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]



# EVALUATE


def evaluate_agent(model_path: str = 'sac_models/best/best_model',
                   norm_path:  str = 'sac_models/best/vec_normalize.pkl',
                   n_episodes: int = 5,
                   terrain_level: int = 0,
                   render: bool = False,
                   record_gif: str = None,
                   gif_width: int = 960,
                   gif_height: int = 540,
                   gif_fps: int = 30):

    eval_env = DummyVecEnv(
        [lambda: QuadrupedEnv(render=render, terrain_level=terrain_level)])

    if os.path.exists(norm_path):
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training    = False
        eval_env.norm_reward = False
        print(f'Loaded model : {model_path}')
        print(f'   Norm stats  : {norm_path}')
    else:
        print(f'Loaded model : {model_path}')
        print(f'   WARNING: No vec_normalize.pkl found at {norm_path}')

    model   = SAC.load(model_path, env=eval_env)
    raw_env = _unwrap_raw(eval_env)

    print(f'   Episodes    : {n_episodes}')
    print(f'   Terrain     : {terrain_level}  (0=flat, 1=slope, 2=rough)')
    print(f'   Device      : {model.device}')
    if render:
        print(f'   Camera      : interactive  '
              f'(WASD/arrows=orbit  QE/scroll=zoom  drag=mouse  R=reset)')
    if record_gif:
        print(f'   Recording   : {record_gif}  ({gif_width}x{gif_height} @ {gif_fps} fps)\n')
    else:
        print()

    # ── interactive camera ────────────────────────────────────────────────────
    cam = None
    if render and raw_env is not None:
        cam = InteractiveCameraController(
            physics_client=raw_env._physics_client,
            distance=getattr(raw_env, '_CAM_DISTANCE',   1.5),
            yaw     =getattr(raw_env, '_CAM_YAW_OFFSET', 45.0),
            pitch   =getattr(raw_env, '_CAM_PITCH',      -20.0),
        )
        cam.start()   # 60 Hz background thread — no changes needed in step loop

    all_foot_contacts: list = []
    all_joint_pos:     list = []
    episode_rewards:   list = []
    episode_distances: list = []
    frames:            list = []

    for ep in range(n_episodes):
        obs       = eval_env.reset()
        ep_reward = 0.0
        ep_steps  = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1

            # keep camera target locked to robot (user can still orbit freely)
            if cam is not None and raw_env._robot_id is not None:
                pos, _ = p.getBasePositionAndOrientation(
                    raw_env._robot_id,
                    physicsClientId=raw_env._physics_client)
                cam.follow(pos)

            # capture frame for GIF (every other step to keep file size down)
            if record_gif and ep_steps % 2 == 0:
                frames.append(_capture_frame(raw_env, gif_width, gif_height))

            all_foot_contacts.append(obs[0][33:37].copy())
            all_joint_pos.append(obs[0][9:21].copy())

            if done[0]:
                break

        x_dist = info[0].get('x_position', 0.0)
        episode_rewards.append(ep_reward)
        episode_distances.append(x_dist)

        ep_width = len(str(n_episodes))
        print(f'  Episode {ep + 1:>{ep_width}}/{n_episodes} '
              f'| Steps: {ep_steps:5d} '
              f'| Reward: {ep_reward:8.1f} '
              f'| Distance: {x_dist:.3f} m')

    # ── cleanup 
    if cam is not None:
        cam.stop()
    eval_env.close()

    mean_r, std_r = np.mean(episode_rewards), np.std(episode_rewards)
    mean_d, std_d = np.mean(episode_distances), np.std(episode_distances)
    print(f'\n  Mean reward  : {mean_r:.2f} +/- {std_r:.2f}')
    print(f'  Mean distance: {mean_d:.3f} +/- {std_d:.3f} m')

    # ── save GIF 
    if record_gif and frames:
        try:
            import imageio
            print(f'\n  Saving {len(frames)} frames -> {record_gif} ...')
            imageio.mimsave(record_gif, frames, fps=gif_fps // 2, loop=0)
            size_mb = os.path.getsize(record_gif) / 1e6
            print(f'  Done! GIF saved: {record_gif}  ({size_mb:.1f} MB)')
            print(f'\n  To embed in README:')
            print(f'    ![SAC Quadruped]({record_gif})')
        except ImportError:
            print('\n  ERROR: imageio not installed. Run:  pip install imageio')

    return np.array(all_foot_contacts), np.array(all_joint_pos)



# GAIT ANALYSIS PLOTS


def plot_gait_analysis(foot_contacts: np.ndarray,
                       joint_pos:     np.ndarray,
                       save_path:     str = 'sac_gait_analysis.png'):
    if len(foot_contacts) == 0:
        print('No data to plot.')
        return

    n_steps = len(foot_contacts)
    time    = np.arange(n_steps) / float(CONTROL_HZ)

    COLORS    = ['#00ff88', '#ff6b6b', '#4dabf7', '#ffd43b']
    LEG_NAMES = ['Front-Left', 'Front-Right', 'Rear-Left', 'Rear-Right']

    fig = plt.figure(figsize=(15, 11), facecolor='#0d1117')
    fig.suptitle('SAC Quadruped Gait Analysis', fontsize=16,
                 color='white', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#161b22')
    ax1.set_title('Foot Contact Pattern (Gait Diagram)', color='white', fontsize=11)
    t_disp = min(time[-1], 10)
    mask   = time <= t_disp
    for i in range(4):
        ax1.fill_between(time[mask], i + foot_contacts[mask, i] * 0.8, i,
                         color=COLORS[i], alpha=0.85, label=LEG_NAMES[i])
        ax1.axhline(i, color='#333', linewidth=0.5)
    ax1.set_yticks(np.arange(4) + 0.4)
    ax1.set_yticklabels(LEG_NAMES, color='#aaa', fontsize=9)
    ax1.set_xlabel('Time (s)', color='#aaa')
    ax1.set_xlim(0, t_disp)
    ax1.tick_params(colors='#aaa')
    ax1.spines[:].set_color('#333')
    ax1.legend(loc='upper right', fontsize=8,
               facecolor='#161b22', labelcolor='white', edgecolor='#333')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#161b22')
    ax2.set_title('Contact Frequency per Leg', color='white', fontsize=11)
    freqs = foot_contacts.mean(axis=0) * 100
    bars  = ax2.bar(LEG_NAMES, freqs, color=COLORS, alpha=0.85, edgecolor='#333')
    ax2.axhline(50, color='#555', linestyle='--', linewidth=1, label='50% baseline')
    ax2.set_ylabel('% time in contact', color='#aaa')
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='x', rotation=15, colors='#aaa')
    ax2.tick_params(axis='y', colors='#aaa')
    ax2.spines[:].set_color('#333')
    ax2.legend(fontsize=8, facecolor='#161b22', labelcolor='white', edgecolor='#333')
    for bar, val in zip(bars, freqs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f'{val:.0f}%', ha='center', color='white', fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#161b22')
    ax3.set_title('Gait Symmetry (Diagonal Pairs)', color='white', fontsize=11)
    sym = [
        np.mean(foot_contacts[:, 0] == foot_contacts[:, 3]) * 100,
        np.mean(foot_contacts[:, 1] == foot_contacts[:, 2]) * 100,
    ]
    bars2 = ax3.bar(['FL <-> RR\n(Diagonal 1)', 'FR <-> RL\n(Diagonal 2)'],
                    sym, color=['#00ff88', '#ff6b6b'], alpha=0.85, edgecolor='#333')
    ax3.axhline(50, color='#555', linestyle='--', linewidth=1, label='50% (random)')
    ax3.axhline(80, color='#ffd43b', linestyle='--', linewidth=1, label='80% (good trot)')
    ax3.set_ylabel('% synchronisation', color='#aaa')
    ax3.set_ylim(0, 105)
    ax3.tick_params(colors='#aaa')
    ax3.spines[:].set_color('#333')
    ax3.legend(fontsize=8, facecolor='#161b22', labelcolor='white', edgecolor='#333')
    for bar, val in zip(bars2, sym):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f'{val:.0f}%', ha='center', color='white', fontsize=9)

    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#161b22')
    ax4.set_title('Hip Joint Trajectories x4 Legs  (first 5 s)', color='white', fontsize=11)
    t_slice = min(int(5 * CONTROL_HZ), n_steps)
    for i, (j_idx, c) in enumerate(zip([0, 3, 6, 9], COLORS)):
        if j_idx < joint_pos.shape[1]:
            ax4.plot(time[:t_slice], joint_pos[:t_slice, j_idx],
                     color=c, alpha=0.9, linewidth=1.3, label=f'{LEG_NAMES[i]} Hip')
    for i, (j_idx, c) in enumerate(zip([1, 4, 7, 10], COLORS)):
        if j_idx < joint_pos.shape[1]:
            ax4.plot(time[:t_slice], joint_pos[:t_slice, j_idx],
                     color=c, alpha=0.45, linewidth=0.9, linestyle='--',
                     label=f'{LEG_NAMES[i]} Thigh')
    ax4.set_xlabel('Time (s)', color='#aaa')
    ax4.set_ylabel('Joint Position (normalised obs)', color='#aaa')
    ax4.tick_params(colors='#aaa')
    ax4.spines[:].set_color('#333')
    ax4.legend(loc='upper right', fontsize=7, ncol=2,
               facecolor='#161b22', labelcolor='white', edgecolor='#333')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f'Saved gait plot -> {save_path}')



# ENTRY POINT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test / evaluate a trained SAC model')
    parser.add_argument('--model',      type=str, default='sac_models/best/best_model')
    parser.add_argument('--norm',       type=str, default='sac_models/best/vec_normalize.pkl')
    parser.add_argument('--episodes',   type=int, default=5)
    parser.add_argument('--terrain',    type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--render',     action='store_true',
                        help='Open PyBullet GUI with interactive camera (WASD + mouse)')
    parser.add_argument('--gait',       action='store_true',
                        help='Generate gait analysis plots')
    parser.add_argument('--gait-out',   type=str, default='sac_gait_analysis.png')
    parser.add_argument('--record',     type=str, default=None,
                        metavar='FILE.gif',
                        help='Save a GIF of the robot walking, e.g. --record demo.gif')
    parser.add_argument('--gif-width',  type=int, default=960,
                        help='GIF frame width  (default: 960)')
    parser.add_argument('--gif-height', type=int, default=540,
                        help='GIF frame height (default: 540)')
    parser.add_argument('--gif-fps',    type=int, default=30,
                        help='GIF fps (default: 30)')
    args = parser.parse_args()

    foot_contacts, joint_pos = evaluate_agent(
        model_path    = args.model,
        norm_path     = args.norm,
        n_episodes    = args.episodes,
        terrain_level = args.terrain,
        render        = args.render,
        record_gif    = args.record,
        gif_width     = args.gif_width,
        gif_height    = args.gif_height,
        gif_fps       = args.gif_fps,
    )

    if args.gait:
        plot_gait_analysis(foot_contacts, joint_pos, save_path=args.gait_out)

# python test_sac.py --terrain 0 --episodes 5 --render --model sac_models/curriculum/stage1_flat/best_model/best_model --norm sac_models/curriculum/stage1_flat/vec_normalize.pkl 