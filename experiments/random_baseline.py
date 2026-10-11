"""Measure random-action rewards for each environment's normalization baseline."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.experiment import ENVIRONMENTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['modern','legacy'],default='modern')
    p.add_argument('--envs',nargs='+',choices=ENVIRONMENTS,default=list(ENVIRONMENTS))
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--out',default='results')
    a=p.parse_args()
    if a.episodes<1: p.error('episodes must be positive')
    from experiments.env_adapter import create_env
    path=Path(a.out)/a.backend/'random_baselines.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    existing={}
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f): existing[row['environment']]=row
    for env_id in a.envs:
        env=create_env(env_id,a.seed,a.backend)
        returns=[]
        try:
            for ep in range(a.episodes):
                obs=env.reset()
                total=0.0
                done=False
                while not done:
                    obs,reward,done,_=env.step(env.action_space.sample())
                    total+=float(reward)
                returns.append(total)
        finally:
            env.close()
        existing[env_id]={'environment':env_id,'mean_reward':float(np.mean(returns)),
                          'std_reward':float(np.std(returns)), 'episodes':a.episodes,'seed':a.seed}
        print(env_id, 'random reward:',round(float(np.mean(returns)),3),flush=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['environment','mean_reward','std_reward','episodes','seed'])
        w.writeheader();w.writerows(existing.values())
    print('Saved:',path)


if __name__=='__main__': main()
