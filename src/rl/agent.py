"""Phase 7: Deep RL Agents — PPO (primary) + SAC (comparison).

Wraps Stable-Baselines3 algorithms with:
  - Custom policy network sizes for 4GB VRAM
  - Training with callbacks (logging, early stopping)
  - Model saving/loading
  - Evaluation and comparison utilities
"""

import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from finrl.agents.stablebaselines3.models import DRLAgent
    _FINRL_AVAILABLE = True
except ImportError:
    _FINRL_AVAILABLE = False

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.metrics import sharpe_ratio, max_drawdown

logger = get_logger('rl_agent')
if not _FINRL_AVAILABLE:
    logger.warning('FinRL not available; using SB3 directly for TD3/A2C/DDPG (functionally identical)')


# ---------------------------------------------------------------------------
# Custom callback for logging portfolio metrics
# ---------------------------------------------------------------------------

class PortfolioMetricsCallback(BaseCallback):
    """Log portfolio-specific metrics during training."""

    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        self.metrics_history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            metrics = evaluate_agent(self.model, self.eval_env, n_episodes=3)
            self.metrics_history.append(metrics)

            if metrics['mean_sharpe'] > self.best_sharpe:
                self.best_sharpe = metrics['mean_sharpe']

            if self.verbose > 0:
                logger.info(
                    f'Step {self.n_calls}: '
                    f'return={metrics["mean_return"]:.2%}, '
                    f'sharpe={metrics["mean_sharpe"]:.2f}, '
                    f'max_dd={metrics["mean_max_dd"]:.2%}'
                )
        return True


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

def create_ppo_agent(env, device='auto', **kwargs):
    """Create PPO agent with config-driven hyperparameters.

    Args:
        env: Gymnasium environment (or VecEnv)
        device: 'auto', 'cpu', or 'cuda'
        **kwargs: Override any PPO parameter

    Returns:
        PPO model
    """
    cfg = get_config('rl')

    # Default policy network: small for 4GB VRAM
    policy_kwargs = kwargs.pop('policy_kwargs', {
        'net_arch': dict(pi=[128, 64], vf=[128, 64]),
    })

    params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': cfg.get('lr', 0.0003),
        'n_steps': cfg.get('n_steps', 2048),
        'batch_size': cfg.get('batch_size', 64),
        'n_epochs': cfg.get('n_epochs', 10),
        'gamma': cfg.get('gamma', 0.99),
        'clip_range': cfg.get('clip_range', 0.2),
        'policy_kwargs': policy_kwargs,
        'device': device,
        'verbose': 0,
    }
    params.update(kwargs)

    model = PPO(**params)
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f'PPO agent created: {n_params:,} policy parameters, '
                f'device={model.device}')
    return model


def create_sac_agent(env, device='auto', **kwargs):
    """Create SAC agent with config-driven hyperparameters.

    Args:
        env: Gymnasium environment (or VecEnv)
        device: 'auto', 'cpu', or 'cuda'
        **kwargs: Override any SAC parameter

    Returns:
        SAC model
    """
    cfg_rl = get_config('rl')
    cfg_sac = cfg_rl.get('sac', {})

    policy_kwargs = kwargs.pop('policy_kwargs', {
        'net_arch': dict(pi=[128, 64], qf=[128, 64]),
    })

    params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': cfg_sac.get('lr', 0.0003),
        'buffer_size': cfg_sac.get('buffer_size', 100000),
        'batch_size': cfg_sac.get('batch_size', 256),
        'tau': cfg_sac.get('tau', 0.005),
        'gamma': cfg_rl.get('gamma', 0.99),
        'ent_coef': cfg_sac.get('ent_coef', 'auto'),
        'policy_kwargs': policy_kwargs,
        'device': device,
        'verbose': 0,
    }
    params.update(kwargs)

    model = SAC(**params)
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f'SAC agent created: {n_params:,} policy parameters, '
                f'device={model.device}')
    return model


def create_td3_agent(env, device='auto', **kwargs):
    """Create TD3 agent (Twin Delayed DDPG — off-policy, continuous actions).

    TD3 uses delayed policy updates and target policy smoothing to reduce
    overestimation bias. More stable than vanilla DDPG.
    """
    cfg = get_config('rl').get('td3', {})
    policy_kwargs = kwargs.pop('policy_kwargs', {'net_arch': [128, 64]})
    params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': cfg.get('lr', 0.0003),
        'buffer_size': cfg.get('buffer_size', 100000),
        'batch_size': cfg.get('batch_size', 256),
        'tau': cfg.get('tau', 0.005),
        'policy_delay': cfg.get('policy_delay', 2),
        'target_policy_noise': cfg.get('target_policy_noise', 0.2),
        'policy_kwargs': policy_kwargs,
        'device': device,
        'verbose': 0,
    }
    params.update(kwargs)
    model = TD3(**params)
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f'TD3 agent created: {n_params:,} parameters, device={model.device}')
    return model


def create_a2c_agent(env, device='auto', **kwargs):
    """Create A2C agent (Advantage Actor-Critic — on-policy, faster convergence).

    A2C uses shorter rollouts (n_steps=5) vs PPO's 2048, making it faster
    per update but noisier. Good for quick exploration.
    """
    cfg = get_config('rl').get('a2c', {})
    policy_kwargs = kwargs.pop('policy_kwargs', {'net_arch': dict(pi=[128, 64], vf=[128, 64])})
    params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': cfg.get('lr', 0.0007),
        'n_steps': cfg.get('n_steps', 5),
        'gamma': get_config('rl').get('gamma', 0.99),
        'ent_coef': cfg.get('ent_coef', 0.01),
        'policy_kwargs': policy_kwargs,
        'device': device,
        'verbose': 0,
    }
    params.update(kwargs)
    model = A2C(**params)
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f'A2C agent created: {n_params:,} parameters, device={model.device}')
    return model


def create_ddpg_agent(env, device='auto', **kwargs):
    """Create DDPG agent (Deep Deterministic Policy Gradient — off-policy).

    Deterministic policy gradient without entropy regularization.
    More sensitive to hyperparameters than SAC but simpler objective.
    """
    cfg = get_config('rl').get('ddpg', {})
    policy_kwargs = kwargs.pop('policy_kwargs', {'net_arch': [128, 64]})
    params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': cfg.get('lr', 0.001),
        'buffer_size': cfg.get('buffer_size', 100000),
        'batch_size': cfg.get('batch_size', 256),
        'tau': cfg.get('tau', 0.005),
        'gamma': get_config('rl').get('gamma', 0.99),
        'policy_kwargs': policy_kwargs,
        'device': device,
        'verbose': 0,
    }
    params.update(kwargs)
    model = DDPG(**params)
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f'DDPG agent created: {n_params:,} parameters, device={model.device}')
    return model


class EnsembleAgent:
    """Averages predicted portfolio weights from multiple trained RL models.

    Reduces single-model variance by combining predictions. Acts as a
    meta-policy: each constituent model votes on the portfolio allocation.

    Args:
        models: List of trained SB3 models (PPO, SAC, TD3, A2C, DDPG).
        weights: Per-model voting weights (default: equal weight).
    """

    def __init__(self, models: list, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        logger.info(f'EnsembleAgent: {len(models)} models, weights={[round(w, 3) for w in self.weights]}')

    def predict(self, obs, deterministic=True):
        actions = [m.predict(obs, deterministic=deterministic)[0] for m in self.models]
        averaged = np.average(np.stack(actions, axis=0), axis=0, weights=self.weights)
        if averaged.sum() > 0:
            averaged = averaged / averaged.sum()
        return averaged, None

    def __repr__(self):
        return f'EnsembleAgent(n_models={len(self.models)})'


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(model, total_timesteps=None, eval_env=None,
                eval_freq=5000, save_path=None, callbacks=None):
    """Train an RL agent.

    Args:
        model: SB3 model (PPO or SAC)
        total_timesteps: Training steps (default from config)
        eval_env: Optional evaluation environment
        eval_freq: Evaluation frequency in steps
        save_path: Path to save best model
        callbacks: Additional callbacks

    Returns:
        model: Trained model
        metrics: Training metrics history
    """
    cfg = get_config('rl')
    if total_timesteps is None:
        total_timesteps = cfg.get('total_timesteps', 500000)

    callback_list = []

    # Portfolio metrics callback
    if eval_env is not None:
        portfolio_cb = PortfolioMetricsCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            verbose=1,
        )
        callback_list.append(portfolio_cb)

    # SB3 eval callback for best model saving
    if eval_env is not None and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(save_path),
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )
        callback_list.append(eval_cb)

    if callbacks:
        callback_list.extend(callbacks)

    combined_cb = CallbackList(callback_list) if callback_list else None

    logger.info(f'Training {type(model).__name__} for {total_timesteps:,} steps')
    model.learn(total_timesteps=total_timesteps, callback=combined_cb)
    logger.info('Training complete')

    # Save final model
    if save_path is not None:
        model.save(save_path)
        logger.info(f'Model saved to {save_path}')

    metrics = portfolio_cb.metrics_history if eval_env is not None else []
    return model, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(model, env, n_episodes=5, deterministic=True):
    """Evaluate agent performance over multiple episodes.

    Args:
        model: Trained SB3 model
        env: Evaluation environment
        n_episodes: Number of episodes to average
        deterministic: Use deterministic policy

    Returns:
        dict with mean metrics across episodes
    """
    all_returns = []
    all_sharpes = []
    all_max_dds = []
    all_steps = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_returns = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if 'portfolio_return' in info:
                episode_returns.append(info['portfolio_return'])

        summary = env.get_portfolio_summary()
        all_returns.append(summary['total_return'])
        all_sharpes.append(summary['sharpe'])
        all_max_dds.append(summary['max_drawdown'])
        all_steps.append(summary['n_steps'])

    return {
        'mean_return': np.mean(all_returns),
        'std_return': np.std(all_returns),
        'mean_sharpe': np.mean(all_sharpes),
        'mean_max_dd': np.mean(all_max_dds),
        'mean_steps': np.mean(all_steps),
        'n_episodes': n_episodes,
    }


def compare_agents(ppo_model, sac_model, env, n_episodes=10,
                   td3_model=None, a2c_model=None, ddpg_model=None,
                   ensemble_model=None, finrl_model=None):
    """Compare all available RL agents by Sharpe ratio.

    Args:
        ppo_model: Trained PPO model (required)
        sac_model: Trained SAC model (required)
        env: Evaluation environment
        n_episodes: Episodes per agent
        td3_model, a2c_model, ddpg_model, ensemble_model: Optional additional agents
        finrl_model: Optional FinRL baseline model for thesis comparison

    Returns:
        dict with per-algorithm metrics and overall winner
    """
    candidates = {'PPO': ppo_model, 'SAC': sac_model}
    if td3_model is not None:
        candidates['TD3'] = td3_model
    if a2c_model is not None:
        candidates['A2C'] = a2c_model
    if ddpg_model is not None:
        candidates['DDPG'] = ddpg_model
    if ensemble_model is not None:
        candidates['Ensemble'] = ensemble_model
    if finrl_model is not None:
        candidates['FinRL'] = finrl_model

    results = {}
    for name, model in candidates.items():
        metrics = evaluate_agent(model, env, n_episodes)
        results[name.lower()] = metrics
        logger.info(f'{name}: return={metrics["mean_return"]:.2%}, sharpe={metrics["mean_sharpe"]:.2f}')

    winner = max(results, key=lambda k: results[k]['mean_sharpe'])
    results['winner'] = winner.upper()
    return results


# ---------------------------------------------------------------------------
# FinRL Baseline
# ---------------------------------------------------------------------------

def create_finrl_agent(env, algorithm='PPO', device='auto', **kwargs):
    """Create a FinRL DRLAgent-wrapped model for thesis baseline comparison.

    FinRL's DRLAgent is a thin wrapper over SB3. This trains a vanilla PPO/SAC
    without our custom T-GAT graph features — used as the external baseline.
    Falls back to plain SB3 if FinRL is unavailable.

    Args:
        env: PortfolioEnv instance
        algorithm: 'PPO', 'SAC', 'TD3', 'A2C', or 'DDPG'
        device: 'auto', 'cpu', or 'cuda'

    Returns:
        SB3 model (usable with evaluate_agent / compare_agents)
    """
    cfg = get_config('rl')

    if _FINRL_AVAILABLE:
        algo_map = {
            'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'A2C': A2C, 'DDPG': DDPG,
        }
        AlgoCls = algo_map.get(algorithm.upper(), PPO)
        # DRLAgent.get_model builds the SB3 model; we pass our env directly
        model = DRLAgent.get_model(
            algorithm.lower(),
            env,
            model_kwargs={
                'learning_rate': cfg.get('lr', 3e-4),
                'batch_size': cfg.get('batch_size', 64),
                'gamma': cfg.get('gamma', 0.99),
                'device': device,
                'verbose': 0,
            },
        )
        logger.info(f'FinRL DRLAgent ({algorithm}) created via FinRL wrapper')
    else:
        # SB3 direct fallback — functionally identical
        algo_map = {
            'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'A2C': A2C, 'DDPG': DDPG,
        }
        AlgoCls = algo_map.get(algorithm.upper(), PPO)
        model = AlgoCls(
            'MlpPolicy', env,
            learning_rate=cfg.get('lr', 3e-4),
            batch_size=cfg.get('batch_size', 64),
            gamma=cfg.get('gamma', 0.99),
            device=device,
            verbose=0,
            **kwargs,
        )
        logger.info(f'FinRL baseline ({algorithm}) via SB3 fallback (finrl not installed)')

    return model


def run_finrl_baseline(env, total_timesteps=None, algorithm='PPO', device='auto'):
    """Train and evaluate FinRL baseline. Returns metrics for thesis comparison table.

    This is the 12th baseline in the comparison table:
      Our Ensemble > PPO > SAC > ... > FinRL > Equal-Weight

    Args:
        env: PortfolioEnv for training
        total_timesteps: Training steps (default from config)
        algorithm: Which algorithm to use as FinRL baseline (default PPO)
        device: compute device

    Returns:
        dict: {sharpe, return, max_drawdown, algorithm, finrl_available}
    """
    cfg = get_config('rl')
    if total_timesteps is None:
        total_timesteps = cfg.get('total_timesteps', 500000)

    model = create_finrl_agent(env, algorithm=algorithm, device=device)

    logger.info(f'Training FinRL baseline ({algorithm}) for {total_timesteps:,} steps...')
    if _FINRL_AVAILABLE:
        model = DRLAgent.train_model(model, tb_log_name='finrl_baseline',
                                     total_timesteps=total_timesteps)
    else:
        model.learn(total_timesteps=total_timesteps)

    metrics = evaluate_agent(model, env, n_episodes=5)
    logger.info(
        f'FinRL baseline result: return={metrics["mean_return"]:.2%}, '
        f'sharpe={metrics["mean_sharpe"]:.2f}, '
        f'max_dd={metrics["mean_max_dd"]:.2%}'
    )

    return {
        'sharpe': metrics['mean_sharpe'],
        'return': metrics['mean_return'],
        'max_drawdown': metrics['mean_max_dd'],
        'algorithm': algorithm,
        'finrl_available': _FINRL_AVAILABLE,
        'model': model,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_agent(model, path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info(f'Agent saved to {path}')


def load_agent(path, env=None, algorithm='PPO'):
    """Load model from disk.

    Args:
        path: Path to saved model (.zip)
        env: Optional environment for continued training
        algorithm: 'PPO' or 'SAC'

    Returns:
        Loaded model
    """
    algo_map = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'A2C': A2C, 'DDPG': DDPG}
    cls = algo_map.get(algorithm.upper(), PPO)
    model = cls.load(path, env=env)
    logger.info(f'{algorithm} agent loaded from {path}')
    return model
