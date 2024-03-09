import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torchdiffeq import odeint
from torch_scatter import scatter_mean


class DynamicsBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(DynamicsBase, self).__init__(*args, **kwargs)

    # def zero_center_of_mass(self, x_pos, segment_ids):
    def zero_center_of_mass(self, x_pos, segment_ids):
        size = x_pos.size()
        assert len(size) == 2  # TODO check this
        seg_means = scatter_mean(x_pos, segment_ids, dim=0)
        mean_for_each_segment = seg_means.index_select(0, segment_ids)
        x = x_pos - mean_for_each_segment

        return x

    def forward(
        self, time, h_t, pos_t, edge_index, edge_attr, segment_ids, *args, **kwargs
    ):
        # output vector field
        raise NotImplementedError

    def integrate(
        self, h_0, pos_0, t_seqs, method, edge_index, segment_ids, *args, **kwargs
    ):
        # sampling method [ODE, Discrete_Diff, SDE]
        raise NotImplementedError

    # def quantize(self, h, pos, edge_index, edge_attr, *args, **kwargs):
    #     # quantize the latent space
    #     h = F.one_hot(torch.argmax(h, dim=-1), num_classes=h.shape[-1])

    #     return pos, h


class CNFbase(DynamicsBase):
    # A base class for continuous normalizing flows
    def __init__(self, *args, **kwargs):
        super(CNFbase, self).__init__(*args, **kwargs)

    def integrate(
        self,
        h_0,
        pos_0,
        edge_index,
        edge_attr,
        segment_ids,
        t_seqs,
        method,
        p_x,
        p_h,
        *args,
        **kwargs
    ):
        # TODO review this method
        # implement decoding with this integrate function
        """
        Return a list of (h_t, pos_t) tensors, each of shape [n_nodes, h_dim] and [n_nodes, 3]
        """
        pos_dim = pos_0.shape[1]
        h_dim = h_0.shape[1]
        z_0 = torch.cat([h_0, pos_0], dim=1)

        def func(t, z):
            _pos_t = z[:, h_dim:]
            _h_t = z[:, :h_dim]
            _dpos_t, _dh_t = self.forward(
                t, _h_t, _pos_t, edge_index, segment_ids=segment_ids, *args, **kwargs
            )
            #add a factor, for path such as VP. 
            _dh_t = p_h.M_para(t) * _dh_t
            _dpos_t = p_x.M_para(t) * _dpos_t
            return torch.cat([_dh_t, _dpos_t], dim=1)

        chain = odeint(
            func,
            z_0,
            t_seqs,
            method=method,
            rtol=kwargs.get("rtol", 1e-7),
            atol=kwargs.get("atol", 1e-9),
        )
        chain = [torch.split(ch, [h_dim, pos_dim], dim=-1) for ch in chain]
        # TODO: check dequantize here.
        return chain


class DiffusionBase(DynamicsBase):
    # A base class for diffusion models
    def __init__(self, *args, **kwargs):
        super(DiffusionBase, self).__init__(*args, **kwargs)
        self.diffusion_timesteps = kwargs.get("diffusion_timesteps", 1000)
        self.scheduler = kwargs.get("scheduler", None)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        sigma2_t_given_s = -torch.expm1(
            torch.softplus(gamma_s) - torch.softplus(gamma_t)
        )
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sample_pzs_given_zt(
        self, s, t, pos_t, h_t, edge_index, segment_ids, edge_attr=None, *args, **kwargs
    ):
        """
        A function used for sampling p(z_s|z_t), e.g., s could be t-1. following the Eq. 21, 22, 29 in the variational diffusion models.
        """
        gamma_s = self.scheduler(s)
        gamma_t = self.scheduler(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)

        sigma_s = torch.sqrt(torch.sigmoid(gamma_s))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))

        eps_xt, eps_ht = self.forward(
            s, h_t, pos_t, edge_index, edge_attr, *args, **kwargs
        )

        mu_xs = (
            pos_t / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_xt
        )

        mu_hs = (
            h_t / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_ht
        )

        sigma = sigma_s * sigma_t_given_s / sigma_t

        reparamter_epsx = torch.randn_like(mu_xs)
        reparamter_epsh = torch.randn_like(mu_hs)

        x_s = mu_xs + sigma * reparamter_epsx
        h_s = mu_hs + sigma * reparamter_epsh

        x_s = self.zero_center_of_mass(x_s, segment_ids)

        return x_s, h_s

    def integrate(
        self, h_0, pos_0, edge_index, edge_attr, segment_ids, *args, **kwargs
    ):
        # TODO: check the
        # implement decoding with this integrate function, which is used to sample the
        """
        The function is used to sample from the diffusion models.
        """
        chain = []
        z_x, z_h = pos_0, h_0

        for s in reversed(range(0, self.diffusion_timesteps)):
            s_array = torch.full((h_0.shape[0], 1), fill_value=s, device=h_0.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            # self,s,t,h_t,pos_t, edge_index, edge_attr, segment_ids, *args, **kwargs
            z_x, z_h = self.sample_pzs_given_zt(
                s_array, t_array, z_x, z_h, edge_index, segment_ids, *args, **kwargs
            )
            # z_x, z_h = self.quantize(z_h, z_x, *args, **kwargs)

            chain.append(torch.concat([z_x, z_h], dim=-1))

        return chain


# class DiffusionBase:
#     def sa,
