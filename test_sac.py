import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import CONTROL_HZ, QuadrupedEnv



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


def _force_camera(eval_env):
    """Force TPP camera on the underlying QuadrupedEnv."""
    try:
        raw = _unwrap_raw(eval_env)
        if raw is not None and hasattr(raw, '_follow_camera') and raw.render_mode:
            raw._follow_camera()
    except Exception:
        pass


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
    raw_env = _unwrap_raw(eval_env)   # needed for offscreen capture

    print(f'   Episodes    : {n_episodes}')
    print(f'   Terrain     : {terrain_level}  (0=flat, 1=slope, 2=rough)')
    print(f'   Device      : {model.device}')
    if record_gif:
        print(f'   Recording   : {record_gif}  ({gif_width}x{gif_height} @ {gif_fps} fps)\n')
    else:
        print()

    all_foot_contacts: list = []
    all_joint_pos:     list = []
    episode_rewards:   list = []
    episode_distances: list = []
    frames:            list = []   # for GIF recording

    for ep in range(n_episodes):
        obs      = eval_env.reset()
        ep_reward = 0.0
        ep_steps  = 0

        if render:
            _force_camera(eval_env)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1

            if render:
                _force_camera(eval_env)

            # Capture frame for GIF (every other frame to keep file size down)
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

    eval_env.close()

    mean_r, std_r = np.mean(episode_rewards), np.std(episode_rewards)
    mean_d, std_d = np.mean(episode_distances), np.std(episode_distances)
    print(f'\n  Mean reward  : {mean_r:.2f} +/- {std_r:.2f}')
    print(f'  Mean distance: {mean_d:.3f} +/- {std_d:.3f} m')

    # ── Save GIF 
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
                        help='Open PyBullet GUI with TPP camera')
    parser.add_argument('--gait',       action='store_true',
                        help='Generate gait analysis plots')
    parser.add_argument('--gait-out',   type=str, default='sac_gait_analysis.png')
    # ── GIF recording 
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