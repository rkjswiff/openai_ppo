"""Run ONE PPO1 training job and store actual rewards and experiment metadata."""
import argparse
import csv
import json
from pathlib import Path
import platform
from importlib.metadata import version, PackageNotFoundError
import time

from experiments.experiment import BY_NAME, ENVIRONMENTS, PAPER_STEPS


def read_episode_rewards(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('No episode data: '+str(path))
    with path.open() as stream:
        rows = [line for line in stream if not line.startswith('#')]
    return [float(row['r']) for row in csv.DictReader(rows)]


def run(args):
    if args.env not in ENVIRONMENTS:
        raise ValueError('Environment not in paper Table 1: '+args.env)
    obj = BY_NAME[args.objective]
    output = Path(args.out) / args.backend / args.env / args.objective / ('seed_'+str(args.seed))
    output.mkdir(parents=True, exist_ok=True)
    completed = output / 'result.json'
    if completed.exists() and not args.overwrite:
        raise FileExistsError('Completed run exists: %s (use --overwrite to rerun)' % completed)

    # Disable eager BEFORE importing any Baselines PPO1 TensorFlow modules.
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from baselines.common import set_global_seeds
    from baselines.common import tf_util as U
    from baselines import logger
    from baselines.bench import Monitor
    from baselines.ppo1 import mlp_policy, pposgd_simple
    from experiments.env_adapter import create_env
    import numpy as np
    import gym

    tf.reset_default_graph()
    set_global_seeds(args.seed)
    logger.configure(dir=str(output), format_strs=['stdout', 'csv', 'log'])
    cfg = dict(
        paper_environment=args.env,
        actual_environment=args.env if args.backend == 'legacy' else args.env.replace('-v1', '-v4'),
        backend=args.backend, objective=args.objective, seed=args.seed,
        timesteps=int(args.steps), horizon=int(args.horizon),
        epochs=int(args.epochs), minibatch_size=int(args.minibatch_size),
        learning_rate=float(args.learning_rate), gamma=0.99, gae_lambda=0.95,
        entropy_coefficient=0.0, adam_epsilon=1e-5,
        hidden_layers=[64, 64], nonlinearity='tanh',
        policy_value_networks='separate', learning_rate_schedule='constant',
        clip_epsilon=obj.parameter if obj.variant == 'clip' else None,
        fixed_kl_beta=obj.parameter if obj.variant == 'fixed_kl' else None,
        adaptive_kl_target=obj.parameter if obj.variant == 'adaptive_kl' else None,
        initial_beta=1.0 if obj.variant == 'adaptive_kl' else None,
    )
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    installed = {}
    for package in ('tensorflow-cpu', 'tensorflow', 'gym', 'gymnasium', 'mujoco', 'numpy'):
        try: installed[package] = version(package)
        except PackageNotFoundError: pass
    provenance = {
        'source': 'modified code from user-uploaded OpenAI Baselines archive',
        'source_paper': 'https://arxiv.org/pdf/1707.06347',
        'python': platform.python_version(), 'platform': platform.platform(),
        'installed_packages': installed, 'modern_port': args.backend == 'modern',
    }
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    env = create_env(args.env, args.seed, args.backend)
    env = Monitor(env, str(output / 'episodes'), allow_early_resets=True)
    beginning = time.time()
    try:
        with U.make_session(num_cpu=1, make_default=False).as_default():
            def policy_fn(name, ob_space, ac_space):
                return mlp_policy.MlpPolicy(
                    name=name, ob_space=ob_space, ac_space=ac_space,
                    hid_size=64, num_hid_layers=2)

            pposgd_simple.learn(
                env, policy_fn,
                max_timesteps=int(args.steps),
                timesteps_per_actorbatch=args.horizon,
                clip_param=(obj.parameter if obj.variant == 'clip' else 0.2),
                entcoeff=0.0,
                optim_epochs=args.epochs,
                optim_stepsize=args.learning_rate,
                optim_batchsize=args.minibatch_size,
                gamma=0.99, lam=0.95,
                schedule='constant',
                objective=obj.variant,
                kl_beta=(obj.parameter if obj.variant == 'fixed_kl' else 1.0),
                target_kl=(obj.parameter if obj.variant == 'adaptive_kl' else 0.01),
            )
    finally:
        env.close()

    all_rewards = read_episode_rewards(output / 'episodes.monitor.csv')
    actual_steps = 0
    with (output / 'progress.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('TimestepsSoFar'):
                actual_steps = int(float(row['TimestepsSoFar']))
    if not all_rewards:
        raise RuntimeError('Training finished but no complete episodes were logged')
    last = all_rewards[-100:]
    result = {
        **cfg,
        'episodes_completed': len(all_rewards),
        'actual_sampled_timesteps': actual_steps,
        'last100_episode_count': len(last),
        'last100_episode_reward': float(np.mean(last)),
        'last100_episode_std': float(np.std(last)),
        'elapsed_seconds': round(time.time()-beginning, 2),
        'paper_reference_score': obj.paper_score,
        'warning': 'Only %d episodes completed; need at least 100 for paper final-100 metric' % len(last) if len(last)<100 else None,
    }
    completed.write_text(json.dumps(result, indent=2)+'\n')
    print('\nSAVED:', completed, '\nFinal-100 train episode reward:', result['last100_episode_reward'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', default='Hopper-v1', choices=ENVIRONMENTS)
    p.add_argument('--objective', default='clip_0.2', choices=list(BY_NAME))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=PAPER_STEPS)
    p.add_argument('--horizon', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--minibatch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--backend', choices=['modern','legacy'], default='modern')
    p.add_argument('--out', default='results')
    p.add_argument('--overwrite', action='store_true')
    a=p.parse_args()
    if a.steps<1 or a.horizon<1 or a.epochs<1 or a.minibatch_size<1:
        p.error('timesteps, horizon, epochs, minibatch must be positive')
    run(a)


if __name__=='__main__':
    main()
