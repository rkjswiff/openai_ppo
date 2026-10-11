"""Calculate Table 1 scores from actual training episodes (never invent results)."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from experiments.experiment import ENVIRONMENTS, OBJECTIVES, PAPER_STEPS, SEEDS


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def analyze(out, backend='modern', require_full=True, make_plots=True):
    root=Path(out)/backend
    rows=[]
    for path in root.glob('*/*/seed_*/result.json'):
        x=json.loads(path.read_text())
        x['source_file']=str(path)
        rows.append(x)
    if not rows:
        raise RuntimeError('No completed result.json files found in '+str(root))
    records={(r['paper_environment'],r['objective'],int(r['seed'])):r for r in rows}
    if len(records)!=len(rows):
        raise ValueError('Duplicate environment/objective/seed results')
    expected={(env,o.name,seed) for env in ENVIRONMENTS for o in OBJECTIVES for seed in SEEDS}
    missing=sorted(expected-set(records))
    extras=set(records)-expected
    insufficient=[k for k,r in records.items() if r['last100_episode_count']<100]
    wrong_steps=[k for k,r in records.items() if r['timesteps'] != PAPER_STEPS]
    complete=not missing and not extras and not insufficient and not wrong_steps
    if require_full and not complete:
        raise ValueError('Not a full paper protocol: %d/%d runs, %d incomplete last-100, '
                         '%d wrong training budgets. See --allow-partial.'%(len(records),len(expected),len(insufficient),len(wrong_steps)))
    if not (root/'random_baselines.csv').exists():
        raise FileNotFoundError('Random baselines not found. Run: python -m experiments.random_baseline --backend '+backend)
    baselines={x['environment']:float(x['mean_reward']) for x in read_csv(root/'random_baselines.csv')}
    best={}
    for env in {r['paper_environment'] for r in rows}:
        env_scores=[float(r['last100_episode_reward']) for r in rows if r['paper_environment']==env]
        if env not in baselines:
            raise ValueError('Random baseline missing for '+env)
        best[env]=max(env_scores)
        if best[env]<=baselines[env]:
            if require_full:
                raise ValueError('No trained agent beats the random policy in '+env+'; cannot normalize')
            print('WARNING: no trained agent beats random in %s; only raw rewards can be reported yet' % env)
            best[env]=None
    data=[]
    for r in rows:
        env=r['paper_environment']
        normalized=(float(r['last100_episode_reward'])-baselines[env])/(best[env]-baselines[env]) if best[env] is not None else float('nan')
        data.append({'environment':env,'objective':r['objective'],'seed':r['seed'],
                     'last100_reward':r['last100_episode_reward'],
                     'random_reward':baselines[env], 'best_observed_reward':best[env],
                     'normalized_score':normalized,'episodes':r['episodes_completed'],
                     'steps':r['timesteps'],'backend':backend})
    report=root/'report';report.mkdir(parents=True,exist_ok=True)
    with (report/'run_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    averages=defaultdict(list)
    for d in data:
        if math.isfinite(d['normalized_score']):
            averages[d['objective']].append(d['normalized_score'])
    comparison=[]
    for o in OBJECTIVES:
        scores=averages.get(o.name,[])
        comparison.append({'objective':o.name,'paper_score':o.paper_score,
                           'reproduced_score':round(sum(scores)/len(scores),5) if scores else '',
                           'n_runs':len(scores),'full_protocol':complete})
    with (report/'table1_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    meta={'full_protocol':complete,'backend':backend,'record_count':len(records),
          'expected_record_count':len(expected),'missing_jobs':len(missing),
          'missing_job_examples':missing[:20], 'insufficient_episode_jobs':insufficient,
          'wrong_training_budget_jobs':wrong_steps,
          'scoring':'(final_100_reward - random_reward) / (best_observed_final_100_reward - random_reward)',
          'normalization_note':'Best computed across the loaded runs in each environment. In a partial run, these scores are NOT comparable with original paper scores. NaN means no training variant beats random yet.',
          'modern_backend_note':'Gymnasium MuJoCo-v4 is not historical MuJoCo-v1; exact published numerical reproduction not promised.'}
    (report/'report_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('\nTABLE 1 COMPARISON ('+('FULL PROTOCOL' if complete else 'PARTIAL / NOT PAPER-COMPARABLE')+')')
    print('%-23s %13s %16s %8s'%('OBJECTIVE','PAPER','OUR NORMALIZED','RUNS'))
    for c in comparison:
        print('%-23s %13.2f %16s %8d'%(c['objective'],c['paper_score'],str(c['reproduced_score']),c['n_runs']))
    print('Saved:',report)
    if make_plots:
        plot_learning_curves(root, report)
    return comparison, meta


def plot_learning_curves(root, report):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib not installed; skipped plots')
        return
    by_env=defaultdict(list)
    for progress in root.glob('*/*/seed_*/progress.csv'):
        try:
            records=read_csv(progress)
            steps=[float(r['TimestepsSoFar']) for r in records if r.get('TimestepsSoFar') and r.get('EpRewMean')]
            rewards=[float(r['EpRewMean']) for r in records if r.get('TimestepsSoFar') and r.get('EpRewMean')]
            if steps: by_env[progress.parts[-4]].append((progress.parts[-3],steps,rewards))
        except (ValueError,KeyError): continue
    for env,curves in by_env.items():
        fig,ax=plt.subplots(figsize=(10,6))
        for obj,steps,rewards in curves:
            ax.plot(steps,rewards,alpha=.55,label=obj)
        ax.set_title(env+' — mean return of last 100 episodes (Baselines monitor)')
        ax.set_xlabel('Environment timesteps');ax.set_ylabel('Training reward (rolling 100)')
        ax.grid(alpha=.2)
        if len(curves)<=15: ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(report/(env.replace('-','_')+'_learning_curve.png'),dpi=150)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='results')
    p.add_argument('--backend',choices=['modern','legacy'],default='modern')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    analyze(a.out,a.backend,require_full=not a.allow_partial,make_plots=not a.no_plots)


if __name__=='__main__': main()
