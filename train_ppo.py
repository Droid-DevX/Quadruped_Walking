

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                 EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import QuadrupedEnv



# CALLBACKS


class CurriculumCallback(BaseCallback):
    """
    Auto-advances terrain difficulty when the agent sustains a mean episode
    reward above a threshold over a rolling window of episodes.

      Level 0 (flat)  -- mean_reward > 500 -> Level 1 (slope)
      Level 1 (slope) -- mean_reward > 400 -> Level 2 (rough heightfield)

    NOTE: reads info['episode']['r'] which is the raw (unnormalised) Monitor
    reward, so thresholds are in raw reward units (max ~2500/episode).
    """
    THRESHOLDS = {0: 500.0, 1: 400.0}

    def __init__(self, vec_env, advance_window=100, verbose=1):
        super().__init__(verbose)
      
        self.vec_env         = vec_env
        self.advance_window  = advance_window
        self.episode_rewards = []
        self._current_level  = 0

    def _get_raw_env(self):
        """Safely unwrap VecNormalize -> DummyVecEnv -> Monitor -> QuadrupedEnv."""
        
        return self.vec_env.envs[0].unwrapped

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                # info['episode']['r'] is raw Monitor reward (not VecNormalize scaled)
                self.episode_rewards.append(info['episode']['r'])

        if len(self.episode_rewards) >= self.advance_window:
            avg = np.mean(self.episode_rewards[-self.advance_window:])
            threshold = self.THRESHOLDS.get(self._current_level, np.inf)
            if avg >= threshold and self._current_level < 2:
                self._current_level += 1
                self._get_raw_env().set_terrain_level(self._current_level)
                if self.verbose:
                    print(f'\n[CURRICULUM] avg_reward {avg:.1f} >= {threshold:.1f}'
                          f' -> advancing to terrain level {self._current_level}'
                          f' (0=flat, 1=slope, 2=rough)\n')
        return True


class RewardLogCallback(BaseCallback):
    """Logs per-episode x-distance and terrain level to TensorBoard."""

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.logger.record('curriculum/terrain_level',
                                   info.get('terrain_level', 0))
                self.logger.record('rollout/ep_x_distance',
                                   info.get('x_position', 0))
        return True


class SaveNormCallback(EvalCallback):
    """
    EvalCallback that:
      1. Syncs eval_env obs normalisation stats from training_env before each
         evaluation -- without this, eval obs are on a different scale than
         training obs, making best_model selection unreliable.
      2. Saves vec_normalize.pkl only when best_mean_reward genuinely improves,
         keeping the model and norm stats always in sync.
    """
    def __init__(self, *args, vec_normalize_path='checkpoints/vec_normalize.pkl',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._norm_path = vec_normalize_path
        self._last_best = -np.inf  
    def _on_step(self) -> bool:
        if hasattr(self.training_env, 'obs_rms') and hasattr(self.eval_env, 'obs_rms'):
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms

        result = super()._on_step()

        if self.best_mean_reward > self._last_best:
            self._last_best = self.best_mean_reward
            if hasattr(self.training_env, 'save'):
                self.training_env.save(self._norm_path)
                if self.verbose >= 1:
                    print(f'  [norm] Saved normalisation stats -> {self._norm_path}')

        return result



# TRAINING


def train(total_timesteps: int = 5_000_000,
          eval_freq: int = 25_000,
          checkpoint_freq: int = 25_000):
    os.makedirs('logs/quadruped_ppo', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  [device] Using: {device}')

    # -- environments 
    def make_env():
        env = QuadrupedEnv(render=False, terrain_level=0, target_velocity=0.5)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env])
    # norm_reward=False so EvalCallback sees true reward for best-model selection
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # -- PPO model 
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log='logs/quadruped_ppo',
        device=device,
        verbose=1,
    )

    # -- callbacks 
    eval_callback = SaveNormCallback(
        eval_env,
        vec_normalize_path='checkpoints/vec_normalize.pkl',
        best_model_save_path='checkpoints/',
        log_path='logs/quadruped_ppo',
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = [
        CurriculumCallback(vec_env, advance_window=100, verbose=1),  
        RewardLogCallback(),
        CheckpointCallback(save_freq=checkpoint_freq, save_path='checkpoints/',
                           name_prefix='quadruped_ppo', verbose=1),
        eval_callback,
    ]

    # -- train 
    print('Starting training...')
    print(f'   Total timesteps : {total_timesteps:,}')
    print(f'   Eval every      : {eval_freq:,} steps')
    print(f'   Checkpoint every: {checkpoint_freq:,} steps')
    print('   tensorboard --logdir logs/quadruped_ppo  ->  http://localhost:6006')
    print('   Best model -> checkpoints/best_model.zip\n')

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name='ppo_curriculum',
    )

    model.save('checkpoints/quadruped_final')
    vec_env.save('checkpoints/vec_normalize.pkl')

    print('\nTraining complete!')
    print('   Best model  -> checkpoints/best_model.zip')
    print('   Final model -> checkpoints/quadruped_final.zip')
    print('   Norm stats  -> checkpoints/vec_normalize.pkl')



# ENTRY POINT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train quadruped with PPO')
    parser.add_argument('--timesteps', type=int, default=2_500_000)
    parser.add_argument('--eval-freq', type=int, default=100_000)
    parser.add_argument('--checkpoint-freq', type=int, default=100_000)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
    )