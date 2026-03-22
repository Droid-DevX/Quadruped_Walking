
import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import CONTROL_HZ, QuadrupedEnv



# EVALUATE


def evaluate_agent(model_path: str = 'checkpoints/best_model',
                   norm_path: str  = 'checkpoints/vec_normalize.pkl',
                   n_episodes: int = 5,
                   terrain_level: int = 0,
                   render: bool = False):
    """Load a trained PPO model and run evaluation episodes."""
    eval_env = DummyVecEnv(
        [lambda: QuadrupedEnv(render=render, terrain_level=terrain_level)])

    if os.path.exists(norm_path):
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training    = False
        eval_env.norm_reward = False
        print(f'✅ Loaded model : {model_path}')
        print(f'   Norm stats  : {norm_path}')
    else:
        print(f'✅ Loaded model : {model_path}')
        print(f'⚠️  No vec_normalize.pkl found — results will be degraded')

    model = PPO.load(model_path, env=eval_env)
    print(f'   Episodes    : {n_episodes}')
    print(f'   Terrain     : {terrain_level}  (0=flat, 1=slope, 2=rough)\n')

    all_foot_contacts: list = []
    all_joint_pos:     list = []
    episode_rewards:   list = []
    episode_distances: list = []

    for ep in range(n_episodes):
        obs, done, ep_reward = eval_env.reset(), False, 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])

            # Obs indices for 44-dim observation (NUM_JOINTS=12)
            # Obs indices for 44-dim observation (NUM_JOINTS=12)
            #   [9:21]  joint positions  (12)
            #   [33:37] foot contacts    (4)
            all_foot_contacts.append(obs[0][33:37].copy())
            all_joint_pos.append(obs[0][9:21].copy())

        x_dist = info[0].get('x_position', 0.0)
        episode_rewards.append(ep_reward)
        episode_distances.append(x_dist)
        print(f'  Episode {ep + 1:>{len(str(n_episodes))}}/{n_episodes} '
              f'| Reward: {ep_reward:8.1f} '
              f'| Distance: {x_dist:.2f} m')

    eval_env.close()

    mean_r, std_r = np.mean(episode_rewards),   np.std(episode_rewards)
    mean_d        = np.mean(episode_distances)
    print(f'\n  Mean reward  : {mean_r:.2f} ± {std_r:.2f}')
    print(f'  Mean distance: {mean_d:.2f} m')

    return np.array(all_foot_contacts), np.array(all_joint_pos)



# GAIT ANALYSIS PLOTS

def plot_gait_analysis(foot_contacts: np.ndarray,
                       joint_pos: np.ndarray,
                       save_path: str = 'gait_analysis.png'):
    """
    Produce a 4-panel gait analysis figure:
      1. Foot contact gait diagram
      2. Contact frequency per leg
      3. Diagonal gait symmetry
      4. Hip joint trajectories

    Args:
        foot_contacts : shape (T, 4)
        joint_pos     : shape (T, 12)
        save_path     : where to save the PNG
    """
    n_steps = len(foot_contacts)
    time    = np.arange(n_steps) / float(CONTROL_HZ)

    COLORS    = ['#00ff88', '#ff6b6b', '#4dabf7', '#ffd43b']
    LEG_NAMES = ['Front-Left', 'Front-Right', 'Rear-Left', 'Rear-Right']

    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    fig.suptitle('🐾 Quadruped Gait Analysis', fontsize=16,
                 color='white', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Gait Diagram 
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#161b22')
    ax1.set_title('Foot Contact Pattern (Gait Diagram)', color='white', fontsize=11)
    for i in range(4):
        ax1.fill_between(time, i + foot_contacts[:, i] * 0.8, i,
                         color=COLORS[i], alpha=0.85, label=LEG_NAMES[i])
        ax1.axhline(i, color='#333', linewidth=0.5)
    ax1.set_yticks(np.arange(4) + 0.4)
    ax1.set_yticklabels(LEG_NAMES, color='#aaa', fontsize=9)
    ax1.set_xlabel('Time (s)', color='#aaa')
    ax1.tick_params(colors='#aaa')
    ax1.spines[:].set_color('#333')
    ax1.set_xlim(0, min(time[-1], 10))
    ax1.legend(loc='upper right', fontsize=8, facecolor='#161b22',
               labelcolor='white', edgecolor='#333')

    # ── 2. Contact Frequency 
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#161b22')
    ax2.set_title('Contact Frequency per Leg', color='white', fontsize=11)
    freqs = foot_contacts.mean(axis=0) * 100
    bars  = ax2.bar(LEG_NAMES, freqs, color=COLORS, alpha=0.85, edgecolor='#333')
    ax2.set_ylabel('% time in contact', color='#aaa')
    ax2.tick_params(axis='x', rotation=15, colors='#aaa')
    ax2.tick_params(axis='y', colors='#aaa')
    ax2.spines[:].set_color('#333')
    for bar, val in zip(bars, freqs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', color='white', fontsize=9)

    # ── 3. Gait Symmetry 
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#161b22')
    ax3.set_title('Gait Symmetry (Diagonal Pairs)', color='white', fontsize=11)
    sym = [np.mean(foot_contacts[:, 0] == foot_contacts[:, 3]) * 100,
           np.mean(foot_contacts[:, 1] == foot_contacts[:, 2]) * 100]
    bars2 = ax3.bar(['FL↔RR\n(Diagonal 1)', 'FR↔RL\n(Diagonal 2)'],
                    sym, color=['#00ff88', '#ff6b6b'], alpha=0.85, edgecolor='#333')
    ax3.axhline(50, color='#555', linestyle='--', linewidth=1, label='50% (random)')
    ax3.axhline(80, color='#ffd43b', linestyle='--', linewidth=1, label='80% (good trot)')
    ax3.set_ylabel('% synchronisation', color='#aaa')
    ax3.set_ylim(0, 100)
    ax3.tick_params(colors='#aaa')
    ax3.spines[:].set_color('#333')
    ax3.legend(fontsize=8, facecolor='#161b22', labelcolor='white', edgecolor='#333')
    for bar, val in zip(bars2, sym):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', color='white', fontsize=9)

    # ── 4. Joint Trajectories 
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#161b22')
    ax4.set_title('Hip Joint Trajectories × 4 Legs', color='white', fontsize=11)
    t_slice = min(300, len(joint_pos))
    # Hip joints for 12-joint robot (3 joints per leg): indices 0, 3, 6, 9
    for i, (j_idx, c) in enumerate(zip([0, 3, 6, 9], COLORS)):
        if j_idx < joint_pos.shape[1]:
            ax4.plot(time[:t_slice], joint_pos[:t_slice, j_idx],
                     color=c, alpha=0.85, linewidth=1.2,
                     label=f'{LEG_NAMES[i]} Hip')
    ax4.set_xlabel('Time (s)', color='#aaa')
    ax4.set_ylabel('Joint Position (normalised)', color='#aaa')
    ax4.tick_params(colors='#aaa')
    ax4.spines[:].set_color('#333')
    ax4.legend(loc='upper right', fontsize=8, facecolor='#161b22',
               labelcolor='white', edgecolor='#333')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f'📊 Saved: {save_path}')



# ENTRY POINT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test / evaluate a trained PPO model')
    parser.add_argument('--model', type=str, default='checkpoints/best_model',
                        help='Path to model zip (without .zip extension)')
    parser.add_argument('--norm', type=str, default='checkpoints/vec_normalize.pkl',
                        help='Path to VecNormalize stats pkl file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--terrain', type=int, default=0, choices=[0, 1, 2],
                        help='Terrain level: 0=flat, 1=slope, 2=rough (default: 0)')
    parser.add_argument('--render', action='store_true',
                        help='Open PyBullet GUI window')
    parser.add_argument('--gait', action='store_true',
                        help='Generate and save gait analysis plots')
    parser.add_argument('--gait-out', type=str, default='gait_analysis.png',
                        help='Output path for gait plot (default: gait_analysis.png)')
    args = parser.parse_args()

    foot_contacts, joint_pos = evaluate_agent(
        model_path=args.model,
        norm_path=args.norm,
        n_episodes=args.episodes,
        terrain_level=args.terrain,
        render=args.render,
    )

    if args.gait:
        plot_gait_analysis(foot_contacts, joint_pos, save_path=args.gait_out)
