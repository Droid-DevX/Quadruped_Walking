"""
train_curriculum.py  —  Curriculum Learning for Quadruped SAC (FROM SCRATCH)
=============================================================================
Trains entirely from scratch across 3 stages, 2M steps each.

Curriculum:
  Stage 1  →  flat      (terrain_id=0)  — 2M steps, fresh SAC
  Stage 2  →  slope     (terrain_id=1)  — 2M steps, warm-start from stage 1
  Stage 3  →  obstacles (terrain_id=2)  — 2M steps, warm-start from stage 2
                        obstacle difficulty increases progressively:
                        0-33%  steps → difficulty 0.0  (gentle)
                        33-66% steps → difficulty 0.0 → 0.5
                        66-100% steps → difficulty 0.5 → 1.0 (full)

# Reward fixes/tuning applied:
#   • Added universal -50.0 terminal penalty upon falling (prevents suicide policy)
#   • Reverted stillness_penalty to 0.5 to prevent agent from killing itself to escape negative returns
#   • forward_reward multiplied by clipped(vx / target_vx) factor

Usage:
  python train_curriculum.py                        # full 3-stage curriculum from scratch
  python train_curriculum.py --start_stage 2        # resume from stage 2 (needs stage 1 output)
  python train_curriculum.py --start_stage 3        # resume from stage 3 (needs stage 2 output)
  python train_curriculum.py --stage1_steps 1000000 # custom step counts
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

from env import QuadrupedEnv


# PROGRESSIVE DIFFICULTY CALLBACK




class ProgressiveDifficultyCallback(BaseCallback):
    """
    Gradually increases difficulty on all envs as training progresses.

    Schedule:
      0%–33%   → difficulty = 0.0  (gentle hills, small sparse obstacles)
      33%–100% → difficulty ramps linearly 0.0 → 1.0
      100%     → difficulty = 1.0  (full steep hills, dense tall obstacles)
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
            if hasattr(inner, 'difficulty'):
                inner.difficulty = diff

        if (self.num_timesteps - self._last_log) >= self.log_freq:
            print(f'  [ProgressiveDifficulty] '
                  f'steps={self.num_timesteps:>8,} | '
                  f'difficulty={diff:.2f}')
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


# HELPERS


STAGE_DIRS = {
    1: 'sac_models/curriculum/stage1_flat',
    2: 'sac_models/curriculum/stage2_slope',
    3: 'sac_models/curriculum/stage3_rough',
}

# terrain_id per stage
STAGE_TERRAIN = {1: 0, 2: 1, 3: 2}
STAGE_NAMES   = {1: 'flat', 2: 'slope', 3: 'obstacles'}


def make_env(terrain_id: int, seed: int = 0, difficulty: float = 0.0):
    """Factory returning a monitored QuadrupedEnv for the given terrain_id."""
    def _init():
        env = QuadrupedEnv(terrain_id=terrain_id, difficulty=difficulty)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(terrain_id: int, n_envs: int = 1, seed: int = 0,
                  norm_path: str = None, difficulty: float = 0.0):
    """Build a vectorised, normalised environment for the given terrain_id."""
    vec = DummyVecEnv([make_env(terrain_id, seed + i, difficulty)
                       for i in range(n_envs)])
    vec = VecMonitor(vec)
    if norm_path and os.path.exists(norm_path):
        print(f'  Loading VecNormalize stats from: {norm_path}')
        vec = VecNormalize.load(norm_path, vec)
        vec.training = True
        # Safety patch for curriculum: if previous stage had constant features 
        # (variance ~ 0), reset their variance to 1 to prevent division by zero
        # explosions when the new stage introduces dynamic values for them.
        import numpy as np
        if hasattr(vec, 'obs_rms') and vec.obs_rms is not None:
            zero_vars = vec.obs_rms.var < 1e-5
            if np.any(zero_vars):
                print(f"  [VecNormalize] Patching {np.sum(zero_vars)} features with zero variance to var=1.0")
                vec.obs_rms.var[zero_vars] = 1.0
                vec.obs_rms.mean[zero_vars] = 0.0
        vec.norm_reward = True
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec


def create_fresh_model(vec_env, device: str):
    """
    Initialise a brand-new SAC with the same hyperparameters used in fine-tuning,
    but with a higher learning rate suitable for training from scratch.
    """
    print('\n  [Stage 1] Creating fresh SAC model (training from scratch)')
    model = SAC(
        'MlpPolicy',
        vec_env,
        learning_rate        = 3e-4,   
        buffer_size          = 1_000_000,
        learning_starts      = 10_000,
        batch_size           = 256,
        tau                  = 0.005,
        gamma                = 0.99,
        train_freq           = 1,
        gradient_steps       = -1,      # auto = 1 grad step per env step collected
        ent_coef             = 'auto',
        target_update_interval = 1,
        verbose              = 1,
        device               = device,
        policy_kwargs        = dict(net_arch=[256, 256]),
    )
    return model


def load_model_for_stage(stage: int, vec_env, device: str, prev_stage_override: int = None):
    """
    Stage 1 → fresh model (no base model needed).
    Stage 2+ → load best (or final) model from the previous stage.
    """
    if stage == 1:
        return create_fresh_model(vec_env, device)

    actual_prev_stage = prev_stage_override if prev_stage_override is not None else stage - 1

    prev_dir   = STAGE_DIRS[actual_prev_stage]
    model_path = os.path.join(prev_dir, 'best_model', 'best_model')
    if not os.path.exists(model_path + '.zip'):
        model_path = os.path.join(prev_dir, f'stage{actual_prev_stage}_final')
        if not os.path.exists(model_path + '.zip'):
            raise FileNotFoundError(
                f'Cannot find stage {actual_prev_stage} model in {prev_dir}.\n'
                f'Run stage {actual_prev_stage} first, or use --start_stage {actual_prev_stage}.')
        print(f'\n  [Stage {stage}] best_model not found, loading final: {model_path}.zip')
    else:
        print(f'\n  [Stage {stage}] Loading stage-{actual_prev_stage} best model: {model_path}.zip')

    model = SAC.load(model_path, env=vec_env, device=device)
    model.learning_rate = 1e-4   # lower lr for fine-tuning
    return model



# STAGE TRAINING


def train_stage(stage:        int,
                total_steps:  int,
                device:       str,
                n_envs:       int = 1,
                prev_stage_override: int = None):

    terrain_level = STAGE_TERRAIN[stage]
    terrain_name  = STAGE_NAMES[stage]
    out_dir       = STAGE_DIRS[stage]

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'eval'), exist_ok=True)

    stage_name = f'Stage{stage}({terrain_name})'
    print(f'\n{"="*60}')
    print(f'  {stage_name}  |  terrain_id={terrain_level}  |  steps={total_steps:,}')
    if terrain_level == 1:
        print(f'  Terrain 1 (Slope): 3-segment procedural slope (downhill->plateau->uphill)')
    elif terrain_level == 2:
        print(f'  Terrain 2 (Obstacles): 13-ray lidar, random box/cylinder obstacles + goal rewards')
        print(f'  Progressive difficulty: 0.0 -> 1.0')
    print(f'{"="*60}')

    # resolve norm path: stage 1 has no prior norm, stages 2+ inherit from prev
    if stage == 1:
        prev_norm = None
    else:
        actual_prev_stage = prev_stage_override if prev_stage_override is not None else stage - 1
        prev_dir  = STAGE_DIRS[actual_prev_stage]
        prev_norm = os.path.join(prev_dir, 'vec_normalize.pkl')
        if not os.path.exists(prev_norm):
            print(f'  Warning: no norm found at {prev_norm}, starting fresh norm.')
            prev_norm = None

    train_env = build_vec_env(terrain_level, n_envs=n_envs,
                              norm_path=prev_norm, difficulty=0.0)
    eval_env  = build_vec_env(terrain_level, n_envs=1,
                              norm_path=prev_norm,
                              difficulty=0.5 if terrain_level > 0 else 0.0)  # terrain_level == terrain_id here
    eval_env.training    = False
    eval_env.norm_reward = False

    model = load_model_for_stage(stage, train_env, device, prev_stage_override)

    checkpoint_cb = CheckpointCallback(
        save_freq        = max(50_000 // n_envs, 1),
        save_path        = os.path.join(out_dir, 'checkpoints'),
        name_prefix      = f'stage{stage}',
        save_vecnormalize= True,
        verbose          = 1)

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

    # Stage 1 resets timestep counter; fine-tune stages continue counting
    reset_timesteps = (stage == 1)

    t0 = time.time()
    model.learn(
        total_timesteps     = total_steps,
        callback            = callbacks,
        reset_num_timesteps = reset_timesteps,
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



# MAIN


def main():
    parser = argparse.ArgumentParser(
        description='Curriculum SAC training from scratch (flat → slope → rough)')
    parser.add_argument('--start_stage',  type=int, default=1,
                        help='Stage to start from (1=flat, 2=slope, 3=rough). '
                             'Stages before this must already be trained.')
    parser.add_argument('--skip_stage2',  action='store_true',
                        help='Skip stage 2 and start stage 3 using stage 1 best model')
    parser.add_argument('--stage1_steps', type=int, default=1_000_000,
                        help='Steps for Stage 1 (flat terrain, from scratch)')
    parser.add_argument('--stage2_steps', type=int, default=2_000_000,
                        help='Steps for Stage 2 (slope terrain, fine-tune)')
    parser.add_argument('--stage3_steps', type=int, default=3_000_000,
                        help='Steps for Stage 3 (rough terrain, fine-tune + progressive difficulty)')
    parser.add_argument('--device',       default='cuda',
                        help='Torch device: cuda or cpu')
    parser.add_argument('--n_envs',       type=int, default=1,
                        help='Number of parallel training environments')
    args = parser.parse_args()

    stage_steps = {
        1: args.stage1_steps,
        2: args.stage2_steps,
        3: args.stage3_steps,
    }

    print(f'\n{"="*60}')
    print(f'  Curriculum SAC — FROM SCRATCH')
    print(f'{"="*60}')
    print(f'  Stage 1 (flat)      : {args.stage1_steps:>10,} steps  [fresh model, lr=3e-4]')
    print(f'  Stage 2 (slope)     : {args.stage2_steps:>10,} steps  [fine-tune,   lr=1e-4]')
    print(f'  Stage 3 (obstacles) : {args.stage3_steps:>10,} steps  [fine-tune,   lr=1e-4, progressive difficulty]')
    print(f'  Start stage     : {args.start_stage}')
    print(f'  Device          : {args.device}')
    print(f'  n_envs          : {args.n_envs}')
    print(f'{"="*60}')

    for stage in [1, 2, 3]:
        if stage == 2 and args.skip_stage2:
            print(f'\n  Skipping stage 2 ({STAGE_NAMES[2]}) via --skip_stage2')
            continue

        if stage < args.start_stage:
            print(f'\n  Skipping stage {stage} ({STAGE_NAMES[stage]})')
            continue

        prev_stage_override = 1 if stage == 3 and args.skip_stage2 else None

        train_stage(
            stage               = stage,
            total_steps         = stage_steps[stage],
            device              = args.device,
            n_envs              = args.n_envs,
            prev_stage_override = prev_stage_override)

    print('\n  ✓  Curriculum training complete!')
    print('\n  Test with:')
    print('    python test_sac.py --terrain 2 --episodes 5 \\')
    print('      --model sac_models/curriculum/stage3_rough/best_model/best_model \\')
    print('      --norm  sac_models/curriculum/stage3_rough/vec_normalize.pkl')


if __name__ == '__main__':
    main()


# python train_curriculum.py --n_envs 4 --device cuda
