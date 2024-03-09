from core.data.geom_dataset import Geom
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate
import torch
from functools import partial
import numpy as np


import sys

sys.path.append("/sharefs/yxsong/project/bfn_rollback/")

from core.data.geom_dataset import Geom
from core.data.prefetch import PrefetchLoader
import core.utils.ctxmgr as ctxmgr
from absl import logging
import time

# from


""" TODO:
    1. DONE! In GeomGen.transform() that will be passed into the Geom() dataset, 
        append z_x and z_pos which are randomly sampled.
"""


def _make_global_adjacency_matrix(n_nodes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def remove_mean(pos, dim=0):
    mean = torch.mean(pos, dim=dim, keepdim=True)
    pos = pos - mean
    return pos


class GeomGen(DataLoader):
    num_atom_types = 16

    def __init__(
        self, data_dir, batch_size, n_node_histogram, device="cpu", **kwargs
    ) -> None:
        print(f"data_dir is: {data_dir}")
        self.datadir = data_dir
        self.batch_size = batch_size
        self.device = device
        self.kwargs = kwargs
        shuffle = kwargs.get("shuffle", True)
        max_kept_conformers = kwargs.get("max_kept_conformers", None)
        remove_h = kwargs.get("remove_h", False)
        max_mol_len = kwargs.get("max_mol_len", None)
        if max_mol_len is not None:
            assert max_mol_len > 0, "max_mol_len must be a positive integer"
        self.max_n_nodes = len(n_node_histogram) + 10
        self.full_adj, self.diag_bool = _make_global_adjacency_matrix(self.max_n_nodes)
        """ 
        NOTE: If you have already called Geom() with some max_mol_len and want to have some new max_mol_len later on,
                1. call Geom() with the new max_mol_len,
                2. follow the warning to delete the processed_dir,
                3. and call Geom() again and wait for the dataset to be reprocessed.
              More on https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
               and check out torch_geometric.data.dataset._process()
        """

        ds = Geom(
            root=data_dir,
            max_kept_conformers=max_kept_conformers,
            remove_h=remove_h,
            max_mol_len=max_mol_len,
            transform=self.transform,
        )

        self.ds = ds

        if kwargs.get("debug", False):
            ds = ds[:129]

        super().__init__(ds, batch_size=batch_size, shuffle=shuffle)

    def transform(self, data):
        data.charges = data.charge.to(self.device).float()
        data.x = data.x.to(self.device).float()
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
        data.idx = data.mol_id.to(self.device)

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

    """
    def max_mol_len_filter(self, data) -> bool: # return a boolean: output true to keep the datapoint
        # return data
        # return (not max_mol_len) or (not data.x.shape[0] > max_mol_len) # false iff max_mol_len is set to some postive integer threshold and data has more atoms than this threshold
        # return not (self.max_mol_len and data.x.shape[0] > self.max_mol_len) # false iff max_mol_len is set to some postive integer threshold and data has more atoms than this threshold
        # return data.x.shape[0] <= self.max_mol_len
    """


if __name__ == "__main__":
    cfg = dict(data_dir="/sharefs/yxsong/project/data/geom", batch_size=4)
    kwargs = dict(
        shuffle=False,
        remove_h=False,
        max_mol_len=1000,
        max_kept_conformers=30,
        debug=True,
        device="cuda",
    )
    geom_loader = GeomGen(*cfg.values(), **kwargs)
    for i, batch in enumerate(geom_loader):
        print("batch: ", batch)
        print("batch_idx", batch.idx)
        print("batch_mol_id: ", batch.mol_id)
        # data = batch[0]
        # print("mol_id: ", data.mol_id, data.mol_id.shape)
        # print("x: ", data.x, data.x.shape)
        # print("pos: ", data.pos)
        # print("geom_id")
        # print("y: ", data.y)
        print(batch._slice_dict)
        # print("charge: ", data.charges)
        break
