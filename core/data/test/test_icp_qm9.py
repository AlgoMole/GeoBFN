import sys

sys.path.append("../..")
import core.data.origin_rotation_icp as oricp
import core.data.qm9_gen as qm9_gen
from core.data.prefetch import PrefetchLoader
from absl import logging
import torch
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pickle
from time import time

if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    cfg = {"datadir": "/home/gongjj/project/data/qm9", "batch_size": 1}
    ds = PrefetchLoader(
        qm9_gen.QM9Gen(
            cfg["datadir"],
            cfg["batch_size"],
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
        ),
        device=torch.device("cpu"),
    )
    cntr = defaultdict(Counter)
    time_cntr = defaultdict(Counter)
    for idx, d in enumerate(ds):
        _start_time = time()
        _, A_rotated, _, iter_count = oricp.icp(
            d.pos.numpy(), d.zpos.numpy(), max_iterations=50
        )
        _end_time = time()
        cntr[d.pos.shape[0]][iter_count] += 1
        time_cntr[d.pos.shape[0]][iter_count] += _end_time - _start_time
        # if idx > 10000:
        #     break
    with open("/home/gongjj/project/tmp/qm9_cntr.pkl", "wb") as f:
        pickle.dump(cntr, f)
    with open("/home/gongjj/project/tmp/qm9_time_cntr.pkl", "wb") as f:
        pickle.dump(time_cntr, f)
