"""Math tests run with pytest, without requiring training or TensorFlow."""
import numpy as np
from experiments.experiment import OBJECTIVES,ENVIRONMENTS,SEEDS


def objective(ratio,advantage,mode,eps=.2,beta=1.0,kl=.1):
    surr=ratio*advantage
    if mode=='unclipped': return surr
    if mode=='clip': return min(surr,np.clip(ratio,1-eps,1+eps)*advantage)
    return surr-beta*kl


def test_paper_objective_count():
    assert len(OBJECTIVES)==11
    assert len(ENVIRONMENTS)*len(OBJECTIVES)*len(SEEDS)==231


def test_positive_advantage_clips_upper_side():
    assert objective(1.5,4,'clip')==4.8
    assert objective(.5,4,'clip')==2.0


def test_negative_advantage_clips_lower_side():
    assert objective(.5,-4,'clip')==-3.2
    assert objective(1.5,-4,'clip')==-6.0


def test_kl_penalty():
    assert np.isclose(objective(1.5,4,'fixed_kl',beta=2,kl=.1),5.8)
