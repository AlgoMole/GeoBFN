import sys
import core.data.origin_rotation_icp as oricp
import core.data.qm9_gen as qm9_gen
from core.data.prefetch import PrefetchLoader
from absl import logging
import torch
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pickle
from time import time


def average(x: Counter):
    return sum([k * v for k, v in x.items()]) / sum(x.values())


if __name__ == "__main__":
    cntr = defaultdict(Counter)
    time_cntr = defaultdict(Counter)
    for idx in range(1000000):
        num_nodes = np.random.randint(5, 200)
        _start_time = time()
        _, A_rotated, _, iter_count = oricp.icp(
            np.random.randn(num_nodes, 3) * 1.38,
            np.random.randn(num_nodes, 3),
            max_iterations=50,
        )
        _end_time = time()
        cntr[num_nodes][iter_count] += 1
        time_cntr[num_nodes][iter_count] += _end_time - _start_time

    # save cntr
    with open("/home/gongjj/project/tmp/sim_cntr.pkl", "wb") as f:
        pickle.dump(cntr, f)
    with open("/home/gongjj/project/tmp/sim_time_cntr.pkl", "wb") as f:
        pickle.dump(time_cntr, f)
