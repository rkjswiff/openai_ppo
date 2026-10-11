"""Single-worker replacement for the few MPI collectives used by Baselines PPO1.

The paper's Table 3 has one actor (T=2048), so MPI is unnecessary here.
This module intentionally doesn't pretend to support multiprocess training.
"""
import numpy as np

class _Comm:
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def allgather(self, value): return [value]
    def allreduce(self, value, op=None): return np.copy(value)
    def Allreduce(self, source, destination, op=None):
        np.copyto(destination, source)
    def Bcast(self, value, root=0): return None

class _MPI:
    COMM_WORLD = _Comm()
    SUM = 'sum'

MPI = _MPI()
