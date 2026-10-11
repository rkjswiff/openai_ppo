"""Expose Gymnasium's MuJoCo-v4 tasks through the old Gym API expected by PPO1.

This is a PORT, not a faithful simulation of 2017 Gym MuJoCo-v1 dynamics.
Use backend=legacy with real Hopper-v1 etc. to improve historical fidelity.
"""
import gym
import numpy as np
from types import SimpleNamespace


def convert_space(space):
    import gymnasium as gymnasium
    if not isinstance(space, gymnasium.spaces.Box):
        raise TypeError('MuJoCo PPO1 needs continuous Box spaces: %r' % space)
    return gym.spaces.Box(
        low=np.asarray(space.low, dtype=np.float32),
        high=np.asarray(space.high, dtype=np.float32),
        dtype=np.float32,
    )


class OldGymMuJoCoAdapter(gym.Env):
    """Gymnasium reset/step (two/five values) -> Baselines Gym (one/four values)."""
    def __init__(self, paper_env_id, seed):
        from experiments.experiment import MODERN_ENVS
        import gymnasium as gymnasium
        new_id = MODERN_ENVS[paper_env_id]
        self.inner = gymnasium.make(new_id)
        self.observation_space = convert_space(self.inner.observation_space)
        self.action_space = convert_space(self.inner.action_space)
        self.spec = SimpleNamespace(id=paper_env_id)
        self._next_seed = int(seed)
        self.action_space.seed(seed)
        self.inner.action_space.seed(seed)

    def seed(self, seed=None):
        self._next_seed = seed
        self.action_space.seed(seed)
        self.inner.action_space.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        seed = kwargs.pop('seed', self._next_seed)
        self._next_seed = None
        obs, _info = self.inner.reset(seed=seed, **kwargs)
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        info = dict(info)
        if truncated and not terminated:
            info['TimeLimit.truncated'] = True
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated or truncated), info

    def close(self):
        self.inner.close()


def create_env(paper_env, seed, backend):
    if backend == 'modern':
        return OldGymMuJoCoAdapter(paper_env, seed)
    if backend == 'legacy':
        env = gym.make(paper_env)
        env.seed(seed)
        return env
    raise ValueError('Unknown backend: '+backend)
