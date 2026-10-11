"""Enumerate (and optionally execute) every job for the original Table 1."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from experiments.experiment import ENVIRONMENTS, OBJECTIVES, PAPER_STEPS, SEEDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke','clip','full'],default='smoke')
    p.add_argument('--backend', choices=['modern','legacy'],default='modern')
    p.add_argument('--steps',type=int,default=None)
    p.add_argument('--out',default='results')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--limit',type=int,default=None, help='Run the first N jobs only')
    p.add_argument('--continue-on-error',action='store_true')
    a=p.parse_args()
    if a.mode=='smoke':
        envs=['Hopper-v1']; objectives=[o for o in OBJECTIVES if o.name in ['unclipped','clip_0.1','clip_0.2','clip_0.3']];seeds=[0]
    elif a.mode=='clip':
        envs=['Hopper-v1']; objectives=[o for o in OBJECTIVES if o.name in ['unclipped','clip_0.1','clip_0.2','clip_0.3']];seeds=list(SEEDS)
    else:
        envs=list(ENVIRONMENTS); objectives=list(OBJECTIVES);seeds=list(SEEDS)
    jobs=[(env,obj.name,seed) for env in envs for obj in objectives for seed in seeds]
    if a.limit is not None: jobs=jobs[:a.limit]
    steps=a.steps if a.steps is not None else (8192 if a.mode=='smoke' else PAPER_STEPS)
    failures=[]
    print('Jobs: %d | steps/run: %s | mode: %s | backend: %s'%(len(jobs), steps,a.mode,a.backend),flush=True)
    for i,(env,obj,seed) in enumerate(jobs,1):
        result=Path(a.out)/a.backend/env/obj/('seed_'+str(seed))/'result.json'
        if result.exists():
            try:
                cfg=json.loads(result.read_text())
                if cfg['timesteps']==steps and cfg['backend']==a.backend:
                    print('[%d/%d] SKIP completed %s'%(i,len(jobs),result),flush=True)
                    continue
            except (ValueError,KeyError): pass
            raise RuntimeError('Existing result settings differ from this run: '+str(result))
        cmd=[sys.executable,'-m','experiments.train','--env',env,
             '--objective',obj,'--seed',str(seed),'--steps',str(steps),
             '--backend',a.backend,'--out',a.out]
        print('[%d/%d] %s'%(i,len(jobs),' '.join(cmd)),flush=True)
        if not a.dry_run:
            r=subprocess.run(cmd,check=False)
            if r.returncode!=0:
                failures.append((env,obj,seed,r.returncode))
                if not a.continue_on_error:
                    raise SystemExit('Training failed, stop. Fix the problem and re-run; completed runs will be skipped.')
    if failures: raise SystemExit('Failed jobs: '+repr(failures))
    print('Batch finished. Next: python -m experiments.summarize --out %s --backend %s'%(a.out,a.backend))


if __name__=='__main__': main()
