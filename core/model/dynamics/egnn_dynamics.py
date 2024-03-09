import torch
import torch.nn as nn
from core.module.egnn_new import EGNN
from core.model.dynamics.dynamics_base import DynamicsBase, CNFbase, DiffusionBase
import numpy as np


class EGNN_dynamics_CNF(CNFbase):
    def __init__(
        self,
        in_node_nf,
        hidden_nf=64,
        device="cuda",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
    ):
        super(EGNN_dynamics_CNF, self).__init__()
        self.egnn = EGNN(
            in_node_nf=in_node_nf + int(condition_time),  # +1 for time
            hidden_nf=hidden_nf,
            out_node_nf=in_node_nf + int(condition_time),
            in_edge_nf=0,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            # normalize=True,
            tanh=tanh,
        )
        self.in_node_nf = in_node_nf

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, time, h_t, pos_t, edge_index, edge_attr=None, segment_ids=None):
        """
        Args:
            time: should be a scalar tensor or the shape of [node_num x batch_size, 1]
            h_state: [node_num x batch_size, in_node_nf]
            coord_state: [node_num x batch_size, 3]
            edge_index: [2, edge_num]
            edge_attr: [edge_num, 1] / None
        """
        if self.condition_time:
            if np.prod(time.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h_t[:, 0:1]).fill_(time.item())
            else:
                h_time = time
            h = torch.cat([h_t, h_time], dim=1)

        h_final, coord_final = self.egnn(h, pos_t, edge_index, edge_attr)
        vel = coord_final - pos_t
        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = self.zero_center_of_mass(vel, segment_ids)

        return vel, h_final


class EGNN_dynamics_diffusion(DiffusionBase):
    def __init__(
        self,
        in_node_nf,
        hidden_nf=64,
        device="cuda",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
    ):
        super(EGNN_dynamics_diffusion, self).__init__()
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=in_node_nf,
            in_edge_nf=0,  # no edge features by default
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            normalize=True,
            tanh=tanh,
        )
        self.in_node_nf = in_node_nf

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, time, h_t, pos_t, edge_index, edge_attr=None, segment_ids=None):
        """
        Args:
            time: should be a scalar tensor or the shape of [node_num x batch_size, 1]
            h_state: [node_num x batch_size, in_node_nf]
            coord_state: [node_num x batch_size, 3]
            edge_index: [2, edge_num]
            edge_attr: [edge_num, 1] / None
        """
        if self.condition_time:
            if np.prod(time.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h_t[:, 0:1]).fill_(time.item())
            else:
                h_time = time
            h = torch.cat([h_t, h_time], dim=1)

        h_final, coord_final = self.egnn(h, pos_t, edge_index, edge_attr)
        vel = coord_final - pos_t
        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = self.zero_center_of_mass(vel, segment_ids)

        return vel, h_final
