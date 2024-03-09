import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from core.losses.loss_utils import sample_zero_centered_gaussian, OT_path, VP_path


class DynamicsLossBase(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, t, dx_t, dh_t, z_x, z_h, x, h, segment_ids, *args, **kwargs):
        """
        #TODO implement this function in the child class
        segment_ids: [n_nodes], for example a tensor like [0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2] indicates that there are 3 molecules in the batch, with 3, 4, 9 atoms respectively
        dx_t: [n_nodes, 3], zero-centered x_dot
        """

        raise NotImplementedError


class Discrete_Diffusion_loss(DynamicsLossBase):
    # Note the objective for discrete diffusion is （SNR(s) - SNR(t)）||x - x(t)||^2, usually s=t-1, refer to Eq. 13 in the VDM https://arxiv.org/pdf/2107.00630.pdf
    """
    Based on VDM, https://arxiv.org/pdf/2107.00630.pdf and EDM
    """

    def __init__(self, scheduler, reduction="mean", timesteps=1000):
        super().__init__(reduction=reduction)
        # self.l2 =
        self.scheduler = scheduler
        self.timesteps = timesteps

    def zeroth_term_loss(
        self, t, dx_t, dh_t, z_x, z_h, x, h, segment_ids, *args, **kwargs
    ):
        raise NotImplementedError

    def forward(self, t, dx_t, dh_t, z_x, z_h, x, h, segment_ids, *args, **kwargs):
        """implement a diffusion loss function for molecule generation."""
        """this is for sampled t, here we just use the t-1 to t as the interval. """
        t_int = torch.round(t * self.timesteps).float()
        s_int = (
            t_int - 1
        )  # s is the previous time step, here we consider the interval as 1.
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        gamma_t = self.scheduler(t_int)  # -log SNR(t) = -log(1-sigma^2/sigma^2)
        gamma_s = self.scheduler(s_int)  # -log SNR(s)

        SNR_weight = torch.exp(-gamma_s) - torch.exp(-gamma_t)  # SNR(t)/SNR(s)

        loss = (
            SNR_weight
            * [
                torch.sum((dx_t - z_x) ** 2, dim=-1)
                + torch.sum((dh_t - z_h) ** 2, dim=-1)
            ]
            * t_is_not_zero
        )

        loss = scatter_mean(loss, segment_ids, dim=0)

        if self.reduction == "mean":
            loss = loss.mean()

        return loss

        # sample x_t
        #
        # sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        # alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        # eps_x = sample_zero_centered_gaussian(x.shape, device=x.device,segment_ids=segment_ids)
        # eps_h = torch.rand_like(h)

        # t_int = torch.round(t * self.timesteps).float()
        # if t_int != 0:
        #     gamma_t = kwargs["gamma_t"] #-log SNR(t) = -log(1-sigma^2/sigma^2)
        #     gamma_s = kwargs["gamma_s"] #-log SNR(s)
        #     sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        #     alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        #     SNR_weight = SNR_weight = torch.exp(-gamma_s) - torch.exp(-gamma_t)
        #     # alpha_t = kwargs["alpha_t"]
        #     # sigma_t = kwargs["sigma_t"]

        # eps_x = (x_t - alpha_t * x) / (sigma_t+1e-8)
        # eps_h = (h_t - alpha_t * h) / (sigma_t+1e-8)

        # compute the loss


class Continous_Diffusion_loss(DynamicsLossBase):
    """
    diffusion model with learned diffusion/ some continous version of SDE. To be added.
    """

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)
        # self.l2 =

    def forward(self, t, dx_t, dh_t, z_x, z_h, x, h, segment_ids, *args, **kwargs):
        return super().forward(
            t, dx_t, dh_t, z_x, z_h, x, h, segment_ids, *args, **kwargs
        )


class FM_loss(DynamicsLossBase):
    """
    Flow matching loss for learning the dynamics of molecules./ note this accept a hybrid training of the dynamics.
    """

    def __init__(
        self,
        probability_path_x=OT_path(),
        probability_path_h=VP_path(),
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.p_x = probability_path_x  # function to get the vector field of the probability path on x
        self.p_h = probability_path_h  # function to get the vector field of the probability path on h

        # self.l2 =

    def forward(self, t, dx_t, dh_t, z_x, z_h, x, h, segment_ids, *args, **kwargs):
        target_x_field = self.p_x.target_field(z_x, x, t)
        target_h_field = self.p_h.target_field(z_h, h, t)
        loss = torch.sum((dx_t - target_x_field) ** 2, dim=-1) + torch.sum(
            (dh_t - target_h_field) ** 2, dim=-1
        )
        loss = scatter_mean(loss, segment_ids, dim=0)

        # if self.reduction == "mean":
        #     loss = loss.mean()

        return loss
