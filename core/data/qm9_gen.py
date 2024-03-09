# from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate
import torch
from functools import partial
import numpy as np
from core.data.qm9 import QM9
from core.data.prefetch import PrefetchLoader
import core.utils.ctxmgr as ctxmgr
from absl import logging
import time


def remove_mean(pos, dim=0):
    mean = torch.mean(pos, dim=dim, keepdim=True)
    pos = pos - mean
    return pos


def _make_global_adjacency_matrix(n_nodes):
    device = "cpu"
    row = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, -1, 1)
        .repeat(1, 1, n_nodes)
        .to(device=device)
    )
    col = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, 1, -1)
        .repeat(1, n_nodes, 1)
        .to(device=device)
    )
    full_adj = torch.concat([row, col], dim=0)
    diag_bool = torch.eye(n_nodes, dtype=torch.bool).to(device=device)
    return full_adj, diag_bool


class QM9Gen(DataLoader):
    num_atom_types = 5

    def __init__(
        self, datadir, batch_size, n_node_histogram, device="cpu", **kwargs
    ) -> None:
        print(f"datadir is: {datadir}")
        self.datadir = datadir
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.device = device
        self.max_n_nodes = len(n_node_histogram) + 10
        self.full_adj, self.diag_bool = _make_global_adjacency_matrix(self.max_n_nodes)
        ds = QM9(
            root=datadir,
            transform=self.transform,
            split=kwargs.get("split", "train"),
        )
        self.ds = ds
        if kwargs.get("debug", False):
            ds = ds[:129]

        super().__init__(ds, batch_size=batch_size, shuffle=kwargs.get("shuffle", True))

    def transform(self, data):
        data.pos = remove_mean(data.pos, dim=0).to(
            self.device
        )  # [N, 3] zero center of mass

        data.zx = torch.randn_like(data.x).to(self.device)
        data.zcharges = torch.randn_like(data.charges).to(self.device)
        data.zpos = remove_mean(torch.randn_like(data.pos), dim=0).to(
            self.device
        )  # [N, 3] zero center of mass
        data.edge_index = self.make_adjacency_matrix(data.x.shape[0]).to(self.device)
        data.edge_attr = None
        # data.z = None
        return data

    def make_adjacency_matrix(self, n_nodes):
        full_adj = self.full_adj[:, :n_nodes, :n_nodes].reshape(2, -1)
        diag_bool = self.diag_bool[:n_nodes, :n_nodes].reshape(-1)
        return full_adj[:, ~diag_bool]

    @classmethod
    def initiate_evaluation_dataloader(cls, data_num, n_node_histogram, batch_size=4):
        """
        Initiate a dataloader for evaluation, which will generate data from prior distribution with n_node_histogram
        """
        max_n_nodes = len(n_node_histogram) + 10
        n_node_histogram = np.array(n_node_histogram / np.sum(n_node_histogram))
        full_adj, diag_bool = _make_global_adjacency_matrix(max_n_nodes)
        make_adjacency_matrix = lambda x: full_adj[:, :x, :x].reshape(2, -1)[
            :, ~(diag_bool[:x, :x].reshape(-1))
        ]

        def _evaluate_transform(data):
            # sample n_nodes from n_node_histogram
            n_nodes = np.random.choice(n_node_histogram.shape[0], p=n_node_histogram)
            data.zx = torch.randn(n_nodes, cls.num_atom_types)
            data.zcharges = torch.randn(n_nodes, 1)
            data.zpos = remove_mean(torch.randn(n_nodes, 3))
            data.edge_index = make_adjacency_matrix(n_nodes)
            data.num_nodes = n_nodes
            return data

        data_list = [Data() for _ in range(data_num)]
        data_list = list(map(_evaluate_transform, data_list))
        ds = InMemoryDataset(transform=_evaluate_transform)
        ds.data, ds.slices = ds.collate(data_list)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    cfg = {"datadir": "/sharefs/gongjj/project/data/qm9_debug", "batch_size": 64}
    ds = QM9Gen(
        cfg["datadir"],
        cfg["batch_size"],
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
        pin_memory_device="cuda:0",
    )

    # print(len(ds.ds))
    # for _i, _d in enumerate(ds.ds):
    #     if _i > 10:
    #         break
    #     print(_d, _d.idx, _d.x, _d.pos.dtype, _d.edge_index.dtype)
    #     print(_d.pos)
    # with ctxmgr.timing("enumerating data"):
    #     for _i, _d in enumerate(ds):
    #         time.sleep(0.03)
    #         pass
    # with ctxmgr.timing("enumerating data loop2"):
    #     for _i, _d in enumerate(ds):
    #         time.sleep(0.03)
    #         pass

    for _i, _d in enumerate(ds):
        if _i > 2:
            break
        # separate(_d)
        print("batch", _d, _d.x.device, _d.pos.device, _d.edge_index.device)
        # print("x", _d.x)
        # print("_slice_dict", _d._slice_dict)
        # print("_inc_dict", _d._inc_dict)
    # # print("ptr", _d.ptr)
    # # print(_d.name)

    # dl = QM9Gen.initiate_evaluation_dataloader(
    #     data_num=10, n_node_histogram=np.array([0, 0, 0, 1, 1, 1, 1, 1])
    # )
    # for b in dl:
    #     print(b)
    # for b in dl:
    #     print(b)
