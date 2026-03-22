import argparse
import os

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                 EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from environment import QuadrupedEnv



# FOLDER STRUCTURE


DIRS = {
    'best':   'sac_models/best',
    'ckpt':   'sac_models/checkpoints',
    'final':  'sac_models/final',
    'logs':   'sac_models/logs/tensorboard',
}



# CALLBACKS


class CurriculumCallback(BaseCallback):
    
    THRESHOLDS = {0: 800.0, 1: 600.0}

    def __init__(self, vec_env, advance_window=100, verbose=1):
        super().__init__(verbose)
        self.vec_env         = vec_env
        self.advance_window  = advance_window
        self.episode_rewards = []
        self._current_level  = 0

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

        if len(self.episode_rewards) >= self.advance_window:
            avg       = np.mean(self.episode_rewards[-self.advance_window:])
            threshold = self.THRESHOLDS.get(self._current_level, np.inf)
            if avg >= threshold and self._current_level < 2:
                self._current_level += 1
                n = len(self.vec_env.envs)
                for i in range(n):
                    self.vec_env.envs[i].unwrapped.set_terrain_level(self._current_level)
                if self.verbose:
                    print(f'\n[CURRICULUM] avg_reward={avg:.1f} >= {threshold:.1f}'
                          f' -- advancing all {n} envs to terrain level'
                          f' {self._current_level} (0=flat, 1=slope, 2=rough)\n')
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
   
    def __init__(self, *args,
                 norm_save_path=f'{DIRS["best"]}/vec_normalize.pkl',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._norm_save_path = norm_save_path
        self._last_best      = -np.inf

    def _on_step(self) -> bool:
        if hasattr(self.training_env, 'obs_rms') and hasattr(self.eval_env, 'obs_rms'):
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms

        result = super()._on_step()

        if self.best_mean_reward > self._last_best:
            self._last_best = self.best_mean_reward
            if hasattr(self.training_env, 'save'):
                self.training_env.save(self._norm_save_path)
                if self.verbose >= 1:
                    print(f'  [norm] best_mean_reward={self.best_mean_reward:.2f}'
                          f' -- saved norm stats -> {self._norm_save_path}')
        return result



# ENVIRONMENT FACTORY


def make_single_env(rank: int = 0, seed: int = 0):
    """Returns a callable that builds one Monitor-wrapped QuadrupedEnv."""
    def _init():
        env = QuadrupedEnv(
            render=False,
            terrain_level=0,
            
            # alive_bonus and energy_penalty are set in environment.py defaults.
            target_velocity=0.5,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def build_vec_env(n_envs: int, norm_obs=True, norm_reward=True):
  
    fns = [make_single_env(rank=i, seed=42) for i in range(n_envs)]

    if n_envs == 1:
        vec = DummyVecEnv(fns)
    else:
        vec = SubprocVecEnv(fns, start_method='spawn')

   
    vec = VecNormalize(vec, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=5.0)
    return vec



# TRAINING


def train(total_timesteps: int = 3_000_000,
          n_envs: int = 1,
          eval_freq: int = 20_000,
          checkpoint_freq: int = 50_000):
    """
    Build environments, SAC model, callbacks and run training.
    """
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n  [device] {device}  |  [n_envs] {n_envs}\n')

    # -- environments 
    vec_env  = build_vec_env(n_envs, norm_obs=True, norm_reward=True)

    # Eval env: single process, norm_reward=False (show true rewards)
    eval_env = build_vec_env(1, norm_obs=True, norm_reward=False)

    # -- SAC model 
    model = SAC(
        policy='MlpPolicy',
        env=vec_env,

        # Core SAC hyperparameters
        learning_rate=3e-4,
        buffer_size=1_000_000,
     
        learning_starts=20_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, 'step'),
        gradient_steps=-1,   # as many  steps as env steps collected

        # Entropy tuning
        ent_coef='auto',
        target_entropy='auto',

        # Network
        policy_kwargs=dict(
            net_arch=[256, 256],
          
            log_std_init=-2.0,
        ),

        tensorboard_log=DIRS['logs'],
        device=device,
        verbose=1,
    )

    # -- callbacks 
    eval_callback = SaveNormCallback(
        eval_env,
        norm_save_path=f'{DIRS["best"]}/vec_normalize.pkl',
        best_model_save_path=DIRS['best'],
        log_path=DIRS['logs'],
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=DIRS['ckpt'],
        name_prefix='sac_quad',
        save_replay_buffer=False,
        verbose=1,
    )

    callbacks = [
        CurriculumCallback(vec_env, advance_window=100, verbose=1),
        RewardLogCallback(),
        checkpoint_cb,
        eval_callback,
    ]

    # -- train 
    print('Starting SAC training...')
    print(f'   Total timesteps : {total_timesteps:,}')
    print(f'   Parallel envs   : {n_envs}')
    print(f'   Eval every      : {eval_freq:,} steps')
    print(f'   Checkpoint every: {checkpoint_freq:,} steps')
    print(f'   Best model      : {DIRS["best"]}/best_model.zip')
    print(f'   Checkpoints     : {DIRS["ckpt"]}/')
    print(f'   TensorBoard     : tensorboard --logdir {DIRS["logs"]}\n')

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name='sac_curriculum',
        log_interval=10,
    )

    # -- save final 
    model.save(f'{DIRS["final"]}/quadruped_sac_final')
    vec_env.save(f'{DIRS["final"]}/vec_normalize_final.pkl')

    print('\nTraining complete!')
    print(f'   Best model   -> {DIRS["best"]}/best_model.zip')
    print(f'   Final model  -> {DIRS["final"]}/quadruped_sac_final.zip')
    print(f'   Norm stats   -> {DIRS["final"]}/vec_normalize_final.pkl')



# ENTRY POINT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train quadruped with SAC')
    parser.add_argument('--timesteps',       type=int, default=3_000_000,
                        help='Total training timesteps (default: 3,000,000)')
    parser.add_argument('--n-envs',          type=int, default=1,
                        help='Parallel envs via SubprocVecEnv (default: 1). '
                             'Recommended: 4 on multi-core machines.')
 
    parser.add_argument('--eval-freq',       type=int, default=20_000,
                        help='Eval callback frequency in steps (default: 20,000)')
    parser.add_argument('--checkpoint-freq', type=int, default=50_000,
                        help='Checkpoint save frequency in steps (default: 50,000)')
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
    )