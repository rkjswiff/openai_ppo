"""Check dependencies and a real MuJoCo step before committing to expensive runs."""
import argparse
import importlib
import platform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['modern','legacy'],default='modern')
    a=p.parse_args()
    print('Python:',platform.python_version(), '| System:',platform.platform())
    for name in ['numpy','gym','tensorflow','mujoco'] + (['gymnasium'] if a.backend=='modern' else []):
        module=importlib.import_module(name)
        print(name, getattr(module,'__version__','(version not exposed)'))
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from experiments.env_adapter import create_env
    from experiments.experiment import ENVIRONMENTS
    for env_name in ENVIRONMENTS:
        env=create_env(env_name,seed=0,backend=a.backend)
        try:
            obs=env.reset()
            obs1,r,done,info=env.step(env.action_space.sample())
            print('PASS:',env_name, 'obs',obs.shape,'reward',round(r,3),'done',done)
        finally:
            env.close()
    print('Environment preflight passed. Next: python -m experiments.batch --mode smoke')


if __name__=='__main__': main()
