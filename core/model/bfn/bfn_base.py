import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torchdiffeq import odeint
from torch_scatter import scatter_mean
import torch.distributions as dist

from core.module.egnn_new import EGNN
import numpy as np
from absl import logging


def corrupt_t_pred(self, mu, t, gamma):
    # if t < self.t_min:
    #   return torch.zeros_like(mu)
    # else:
    # eps_pred = self.model()
    t = torch.clamp(t, min=self.t_min)
    # t = torch.ones((mu.size(0),1)).cuda() * t
    eps_pred = self.model(mu, t)
    x_pred = mu / gamma - torch.sqrt((1 - gamma) / gamma) * eps_pred
    return x_pred


class bfnBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(bfnBase, self).__init__(*args, **kwargs)

    # def zero_center_of_mass(self, x_pos, segment_ids):
    def zero_center_of_mass(self, x_pos, segment_ids):
        size = x_pos.size()
        assert len(size) == 2  # TODO check this
        seg_means = scatter_mean(x_pos, segment_ids, dim=0)
        mean_for_each_segment = seg_means.index_select(0, segment_ids)
        x = x_pos - mean_for_each_segment

        return x

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_r

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # print("msx",mu,sigma,x)
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        # print(sigma.min(),sigma.max())
        # print(mu.min(),mu.max())

        f_ = 0.5 * (1 + torch.erf((x - mu) / ((sigma) * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)
        # if not torch.all(f_.isfinite()):
        #     print("f_", f_.min(), f_.max())
        #     print("mu", mu.min(), mu.max())
        #     print("sigma", sigma.min(), sigma.max())
        #     print("x", x.min(), x.max())
        #     print("flag_upper", flag_upper.min(), flag_upper.max())
        #     print("flag_lower", flag_lower.min(), flag_lower.max())
        #     raise ValueError("f_ is not finite")
        return f_

    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        """
        x: [N, K]
        """
        beta = beta1 * (t**2)  # (B,)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        return theta

    def ctime4continuous_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x):
        # TODO not debuged yet
        weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        return weight * (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K):
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def dtime_discrete_loss(self):
        pass

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class bfn4MolEGNN(bfnBase):
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
        sigma1_coord=0.02,
        sigma1_charges=0.02,
        beta1=3.0,
        bins=9,
        sample_steps=100,
        t_min=0.0001,
        no_diff_coord=False,
        charge_discretised_loss=False,
        charge_clamp=False,
    ):
        super(bfn4MolEGNN, self).__init__()

        out_node_nf = 2  # for the coordinates
        # print("bfn",seperate_charge_net)

        self.egnn = EGNN(
            in_node_nf=in_node_nf + int(condition_time),  # +1 for time
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,  # need to predict the mean and variance of the charges for discretised data
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
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)
        self.sample_steps = sample_steps
        self.t_min = t_min
        self.no_diff_coord = no_diff_coord
        k_c, self.k_l, self.k_r = self.get_k_params(bins)
        self.sigma1_charges = torch.tensor(sigma1_charges, dtype=torch.float32)
        self.charge_discretised_loss = charge_discretised_loss
        self.bins = torch.tensor(bins, dtype=torch.float32)
        self.charge_clamp = charge_clamp
        start, end = -2, 2
        self.width = (end - start) / self.in_node_nf
        self.centers = torch.linspace(
            start, end, self.in_node_nf, device="cuda:0"
        )  # [feature_num]
        # print(in_node_nf + int(condition_time),in_node_nf + int(condition_time)+1)
        self.K_c = torch.tensor(k_c).to(self.device)

    def gaussian_basis(self, x):
        """x: [batch_size, ...]"""
        # x = torch.unsqueeze(x, dim=-1)  # [batch_size, ..., 1]
        out = (x - self.centers) / self.width
        ret = torch.exp(-0.5 * out**2)

        return F.normalize(ret, dim=-1, p=1) * 2 - 1  # [batch_size, ..., feature_num]

    def interdependency_modeling(
        self,
        time,
        mu_charge_t,
        mu_pos_t,
        gamma_coord,
        gamma_charge,
        edge_index,
        edge_attr=None,
        segment_ids=None,
        inference=False,
    ):
        """
        Args:
            time: should be a scalar tensor or the shape of [node_num x batch_size, 1]
            h_state: [node_num x batch_size, in_node_nf]
            coord_state: [node_num x batch_size, 3]
            edge_index: [2, edge_num]
            edge_attr: [edge_num, 1] / None
        """

        h_in = self.gaussian_basis(mu_charge_t)
        if self.condition_time:
            if np.prod(time.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(mu_charge_t[:, 0:1]).fill_(time.item())
            else:
                h_time = time
            h = torch.cat([h_in, h_time], dim=1)
        else:
            h = h_in

        # print("mu_pos_t_in", mu_pos_t_in.shape, "h", h.shape, "h_in", h_in.shape)
        h_final, coord_final = self.egnn(h, mu_pos_t, edge_index, edge_attr)
        # here we want the last two dimensions of h_final is mu_eps and ln_sigma_eps
        # h_final = [atom_types, charges_mu,charge_sigma, t]
        # if not torch.all(torch.isfinite(h_final)) or not torch.all(
        #     coord_final.isfinite()
        # ):
        #     print("h_time", h_time.min(), h_time.max())
        #     print("h_in", h_in.min(), h_in.max())
        #     print("mu_charge_t", mu_charge_t.min(), mu_charge_t.max())
        #     print("mu_pos_t", mu_pos_t.min(), mu_pos_t.max())
        #     print("h_final", h_final.min(), h_final.max())
        #     print("coord_final", coord_final.min(), coord_final.max())
        #     raise ValueError("h_final is not finite or coord_final is not finite")

        if self.no_diff_coord:
            eps_coord_pred = coord_final
        else:
            eps_coord_pred = coord_final - mu_pos_t

        # DEBUG coord clamp
        eps_coord_pred = self.zero_center_of_mass(eps_coord_pred, segment_ids)

        mu_charge_eps = h_final[:, -2:-1]  # [n_nodes,1]
        sigma_charge_eps = h_final[:, -1:]  # [n_nodes,1]

        eps_coord_pred = torch.clamp(eps_coord_pred, min=-10, max=10)
        mu_charge_eps = torch.clamp(mu_charge_eps, min=-10, max=10)
        sigma_charge_eps = torch.clamp(sigma_charge_eps, min=-10, max=10)
        # if not torch.all(mu_charge_eps.isfinite()) and not torch.all(sigma_charge_eps.isfinite()):
        #     print("mu_charge_eps", mu_charge_eps.min(), mu_charge_eps.max())
        #     print("sigma_charge_eps", sigma_charge_eps.min(), sigma_charge_eps.max())
        #     print("pre clamp mu_charge_eps", h_final[:, -2:-1].min(), h_final[:, -2:-1].max())
        #     print("pre clamp sigma_charge_eps", h_final[:, -1:].min(), h_final[:, -1:].max())
        #     raise ValueError("mu_charge_eps or sigma_charge_eps is not finite")

        coord_pred = (
            mu_pos_t / gamma_coord
            - torch.sqrt((1 - gamma_coord) / gamma_coord) * eps_coord_pred
        )
        # DEBUG coord clamp
        # coord_pred = self.zero_center_of_mass(
        #     torch.clamp(coord_pred, min=-15.0, max=15.0), segment_ids
        # )

        if self.charge_discretised_loss:
            # print(
            #     "mu_charge_t",
            #     mu_charge_t.min().detach().cpu().numpy(),
            #     mu_charge_t.max().detach().cpu().numpy(),
            # )
            # we do not predict the log sigma.
            sigma_charge_eps = torch.exp(sigma_charge_eps)
            mu_charge_x = (
                mu_charge_t / gamma_charge
                - torch.sqrt((1 - gamma_charge) / gamma_charge) * mu_charge_eps
            )
            sigma_charge_x = (
                torch.sqrt((1 - gamma_charge) / gamma_charge) * sigma_charge_eps
            )
            # if not torch.all(sigma_charge_x.isfinite()) or not torch.all(
            #     mu_charge_x.isfinite()
            # ):
            #     print("mu_charge_x", mu_charge_x.min(), mu_charge_x.max())
            #     print("sigma_charge_x", sigma_charge_x.min(), sigma_charge_x.max())
            #     print("mu_charge_t", mu_charge_t.min(), mu_charge_t.max())
            #     print("gamma_charge", gamma_charge.min(), gamma_charge.max())
            #     print("mu_charge_eps", mu_charge_eps.min(), mu_charge_eps.max())
            #     print(
            #         "sigma_charge_eps", sigma_charge_eps.min(), sigma_charge_eps.max()
            #     )
            #     raise ValueError("sigma_charge_x is not finite")

            if self.charge_clamp:
                mu_charge_x = torch.clamp(mu_charge_x, min=-2, max=2)
                sigma_charge_x = torch.clamp(sigma_charge_x, min=1e-6, max=4)

            k_r = torch.tensor(self.k_r).to(self.device).unsqueeze(-1).unsqueeze(0)
            k_l = torch.tensor(self.k_l).to(self.device).unsqueeze(-1).unsqueeze(0)
            p_o = self.discretised_cdf(
                mu_charge_x, sigma_charge_x, k_r
            ) - self.discretised_cdf(mu_charge_x, sigma_charge_x, k_l)
            k_hat = p_o
            # if not torch.all(k_hat.isfinite()):
            #     print("k_hat", k_hat.min(), k_hat.max())
            #     print("mu_charge_x", mu_charge_x.min(), mu_charge_x.max())
            #     print("sigma_charge_x", sigma_charge_x.min(), sigma_charge_x.max())
            #     print("k_r", k_r.min(), k_r.max())
            #     print("k_l", k_l.min(), k_l.max())
            #     print("p_o", p_o.min(), p_o.max())
            #     raise ValueError("k_hat is not finite")
        else:
            """
            charge is taken as the continous variable.
            the sigma is just not trained and fixed. And the previous mu is considered as the eps
            """
            k_hat = (
                mu_charge_t / gamma_charge
                - torch.sqrt((1 - gamma_charge) / gamma_charge) * mu_charge_eps
            )
            if self.charge_clamp:
                k_hat = torch.clamp(k_hat, min=-1, max=1)

        return coord_pred, k_hat

    def loss_one_step(
        self,
        t,
        x,
        pos,
        edge_index,
        edge_attr=None,
        segment_ids=None,
    ):
        charges = x[:, -1:]
        mask = t > self.t_min
        t = torch.clamp(t, min=self.t_min)
        mu_charge, gamma_charge = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_charges, x=charges
        )

        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_coord, x=pos
        )
        mu_charge = torch.clamp(mu_charge, min=-10, max=10)
        mu_coord = torch.clamp(mu_coord, min=-10, max=10)
        mu_coord = torch.where(mask, mu_coord, torch.zeros_like(mu_coord))
        mu_charge = torch.where(mask, mu_charge, torch.zeros_like(mu_charge))
        mu_coord = self.zero_center_of_mass(mu_coord, segment_ids=segment_ids)
        coord_pred, k_hat = self.interdependency_modeling(
            t,
            mu_charge_t=mu_charge,
            mu_pos_t=mu_coord,
            gamma_coord=gamma_coord,
            gamma_charge=gamma_charge,
            edge_index=edge_index,
            edge_attr=edge_attr,
            segment_ids=segment_ids,
        )
        posloss = self.ctime4continuous_loss(
            t=t, sigma1=self.sigma1_coord, x_pred=coord_pred, x=pos
        )

        if self.charge_discretised_loss:
            k_c = self.K_c.unsqueeze(-1).unsqueeze(0)
            k_hat = (k_hat * k_c).sum(dim=1)
            charge_loss = self.ctime4discreteised_loss(
                t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges
            )
        else:
            charge_loss = self.ctime4continuous_loss(
                t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges
            )

        return (
            posloss,
            charge_loss,
            (mu_coord, mu_charge, coord_pred, k_hat, gamma_coord, gamma_charge),
        )

    def forward(
        self, n_nodes, edge_index, sample_steps=None, edge_attr=None, segment_ids=None
    ):
        """
        The function implements a sampling procedure for BFN
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1, note here we use a single t
            theta_t: [node_num x batch_size, atom_type]
            mu_t: [node_num x batch_size, 3]
            edge_index: [2, edge_num]
            edge_attr: [edge_num, 1] / None
        """
        mu_pos_t = torch.zeros((n_nodes, 3)).to(self.device)  # [N, 4] coordinates prior
        mu_charge_t = torch.zeros((n_nodes, 1)).to(self.device)

        ro_coord = torch.tensor(1, dtype=torch.float32).to(self.device)
        ro_charge = torch.tensor(1, dtype=torch.float32).to(self.device)

        if sample_steps is None:
            sample_steps = self.sample_steps
        theta_traj = []
        for i in range(1, sample_steps + 1):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            t = torch.clamp(t, min=self.t_min)
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)
            gamma_charge = 1 - torch.pow(self.sigma1_charges, 2 * t)
            mu_charge_t = torch.clamp(mu_charge_t, min=-10, max=10)
            mu_pos_t = torch.clamp(mu_pos_t, min=-10, max=10)
            mu_pos_t = self.zero_center_of_mass(mu_pos_t, segment_ids=segment_ids)
            coord_pred, k_hat = self.interdependency_modeling(
                time=t,
                mu_charge_t=mu_charge_t,
                mu_pos_t=mu_pos_t,
                gamma_coord=gamma_coord,
                gamma_charge=gamma_charge,
                edge_index=edge_index,
                edge_attr=edge_attr,
                segment_ids=segment_ids,
                inference=True,
            )

            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
            )
            y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(
                1 / alpha_coord
            )
            y_coord = self.zero_center_of_mass(
                torch.clamp(y_coord, min=-10, max=10), segment_ids
            )
            mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (
                ro_coord + alpha_coord
            )
            ro_coord = ro_coord + alpha_coord

            if not self.charge_discretised_loss:
                alpha_charge = torch.pow(self.sigma1_charges, -2 * i / sample_steps) * (
                    1 - torch.pow(self.sigma1_charges, 2 / sample_steps)
                )
                y_charge = k_hat + torch.randn_like(k_hat) * torch.sqrt(
                    1 / alpha_charge
                )
                mu_charge_t = (ro_charge * mu_charge_t + alpha_charge * y_charge) / (
                    ro_charge + alpha_charge
                )
                ro_charge = ro_charge + alpha_charge
                theta_traj.append((coord_pred, k_hat))
            else:
                k_c = self.K_c.unsqueeze(-1).unsqueeze(0)
                e_k_hat = (k_hat * k_c).sum(dim=1, keepdim=True)
                e_k_c = self.K_c[(e_k_hat - k_c).abs().argmin(dim=1)]

                theta_traj.append((coord_pred, e_k_c))

                alpha_charge = torch.pow(self.sigma1_charges, -2 * i / sample_steps) * (
                    1 - torch.pow(self.sigma1_charges, 2 / sample_steps)
                )
                # print("k_hat",k_hat,k_hat.shape,k_hat.min(),k_hat.max())

                y_charge = e_k_c + torch.randn_like(e_k_c) * torch.sqrt(
                    1 / alpha_charge
                )
                mu_charge_t = (ro_charge * mu_charge_t + alpha_charge * y_charge) / (
                    ro_charge + alpha_charge
                )
                ro_charge = ro_charge + alpha_charge
        mu_charge_t = torch.clamp(mu_charge_t, min=-10, max=10)
        mu_pos_t = torch.clamp(mu_pos_t, min=-10, max=10)
        mu_pos_final, k_hat_final = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1)).to(self.device),
            mu_charge_t=mu_charge_t,
            mu_pos_t=mu_pos_t,
            gamma_coord=1 - self.sigma1_coord**2,
            gamma_charge=1 - self.sigma1_charges**2,
            edge_index=edge_index,
            edge_attr=edge_attr,
            segment_ids=segment_ids,
        )
        if self.charge_discretised_loss:
            k_c = self.K_c.unsqueeze(-1).unsqueeze(0)
            e_k_hat = (k_hat_final * k_c).sum(dim=1, keepdim=True)
            e_k_c = self.K_c[(e_k_hat - k_c).abs().argmin(dim=1)]
            theta_traj.append((mu_pos_final, e_k_c))
        else:
            theta_traj.append((mu_pos_final, k_hat_final))

        return theta_traj
