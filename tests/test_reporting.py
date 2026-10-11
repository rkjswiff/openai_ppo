import csv
import json
from pathlib import Path
import pytest
from experiments.summarize import analyze


def test_partial_scoring_and_reject_full(tmp_path):
    root=tmp_path/'modern'
    baseline=root/'random_baselines.csv';baseline.parent.mkdir(parents=True)
    baseline.write_text('environment,mean_reward,std_reward,episodes,seed\nHopper-v1,10,0,100,12345\n')
    for objective,score in [('clip_0.2',110),('unclipped',20)]:
        folder=root/'Hopper-v1'/objective/'seed_0';folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps({'paper_environment':'Hopper-v1','objective':objective,'seed':0,
            'last100_episode_count':100,'last100_episode_reward':score,'episodes_completed':200,
            'timesteps':1_000_000}))
    with pytest.raises(ValueError,match='Not a full paper protocol'):
        analyze(tmp_path,'modern',require_full=True,make_plots=False)
    rows,meta=analyze(tmp_path,'modern',require_full=False,make_plots=False)
    assert not meta['full_protocol']
    x={row['objective']:row for row in rows}
    assert x['clip_0.2']['reproduced_score']==1.0
    assert x['unclipped']['reproduced_score']==0.1
