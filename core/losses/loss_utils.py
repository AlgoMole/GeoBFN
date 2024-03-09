"""
This file implement the noise schedule for the diffusion model/ SDE model. Also include some helper functions for the flow matching model.
"""
import torch
import numpy as np
from torch_scatter import scatter_mean, scatter_add

# Note the objective for discrete diffusion is （SNR(s) - SNR(t)）||x - x(t)||^2, usually s=t-1, refer to Eq. 13 in the VDM https://arxiv.org/pdf/2107.00630.pdf
# Here we implement two schedule, which is the polynomial schedule and the cosine schedule. The polynomial schedule is the same as the one in the VDM paper.

#Take this in mind that q(x_t|x_0) = N(x_t;sqrt(alpht_cumprod)*x_0,1-alphat_cumprod), refer to Eq. 2 in DDPM
"""
Utils used for all the models
"""


def sample_zero_centered_gaussian(size, device, segment_ids):
    assert len(size) == 2  # TODO check this
    x = torch.randn(size, device=device)
    seg_means = scatter_mean(x, segment_ids, dim=0)
    mean_for_each_segment = seg_means.index_select(0, segment_ids)
    x = x - mean_for_each_segment
    return x


"""
Loss utils for the diffsuion model
"""


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    # alphas2 is the cumprod of alpha_t (alphas_step)
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    return the alpha^2 for each timestep [0, timesteps]
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    f_t = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    # alpha_t^{-} = f(t)/f(0), where f(t) = cos^2((t/T + s)/(1+s) * pi/2)
    alphat_cumprod = f_t / f_t[0]
    betas = 1 - (alphat_cumprod[1:] / alphat_cumprod[:-1])
    # beta_t =1 -  alphat_cumprod/alpha(t-1)_cumprod
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    # recover, note alphas_cumprod is the same as alphat_cumprod, but with some clip operation.
    # refer to Eq.18 in VDM
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print("alphas2", alphas2)
        # Note alpha_t = sqrt(alphas2), SNR(t) = 1 - sigma^2 / sigma^2 ;
        # q(z_t|x_0) = N(z_t;sqrt(alphas2)*x_0,1-alphas2)

        # Gamma = -log(SNR(t)), which is monotonically increasing.
        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print("gamma", -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


# TODO: add learnable noise schedule/ continous diffusion model.

"""
Loss utils for the flow matching model. 
"""


class Probability_path_base:
    def __init__(self, *args, **kwargs):
        super(Probability_path_base, self).__init__(*args, **kwargs)

    def sample_x_t(self, z, x, t):
        """
        sample from the distribution
        """
        raise NotImplementedError

    def target_field(self, z, x, t):
        """
        return a vector field on that transport plan
        """
        raise NotImplementedError
    
    def M_para(self,t):

        return 1.0 



class OT_path(Probability_path_base):
    # Eq 23 in Flow matching paper https://openreview.net/pdf?id=PqvMRDCJT9t
    # TODO: check the direction of the vector field
    """
    return a vector field on that transport plan
    """

    def __init__(self, *args, **kwargs):
        super(OT_path, self).__init__(*args, **kwargs)

    def sample_x_t(self, z, x, t):
        #t -> 0, then x_0 is nosie / t-> 1, then x_1 is data 
        x_t = (1.0 - t) * z + t * x
        return x_t

    def target_field(self, z, x, t):
        return x - z
    


class VP_path(Probability_path_base):
    # TODO: check the target field in the data
    def __init__(self, *args, **kwargs):
        super(VP_path, self).__init__(*args, **kwargs)

    def sample_x_t(self, z, x, t):
        # t in zeros and ones 
        beta_min = 0.1
        beta_max = 20
        # t = 1 - t # Reverse time, x0 for sample, x1 for noise
        log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
        # log_mean_coeff.to(x.device)
        mean = (1 - torch.exp(log_mean_coeff)) * x
        std = torch.sqrt(torch.exp(2.0 * log_mean_coeff))
        # t-->0, log_mean_coeff-->0, mean-->0, std-->1 x_0 GAUSSIAN
        # t-->1, log_mean_coeff-->-0.5*beta_min, mean-->1*x, std-->sqrt(1-exp(-beta_min) x_1 DATA

        return mean + std * z

    def target_field(self, z, x, t):
        x_t = self.sample_x_t(x, z, t)
        vector = torch.exp(-self.T(1-t)) * x_t - torch.exp(-0.5 * self.T(1-t)) * x
        #vector = (1 - torch.exp(-0.5 * T(t))) * x - (1 - torch.exp(-T(t))) * x_t
        # vector =  - 

        return -vector

    def T(self,t):
        # 0   0, 1 beta_max
        beta_min = 0.1
        beta_max = 20
        return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t
    
    def T_hat(self,t):
        # 0 beta_min, 1 beta_max
        beta_min = 0.1
        beta_max = 20
        return (beta_max - beta_min) * t + beta_min

    def M_para(self,t):
        M_para =  0.5 * self.T_hat(1- t) / (1 - torch.exp(-self.T(1-t)) + 1e-5)  # add epsilon to stable it
        return M_para

        
# def VP_path(z,x,t):
#     #Eq. 19 in Flow matching paper https://openreview.net/pdf?id=PqvMRDCJT9t
#     """
#     return a vector field on that transport plan
#     """

#     def T(t):
#         # 0   0, 1 beta_max
#         beta_min = 0.1
#         beta_max = 20
#         return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t


#     def T_hat(t):
#         # 0 beta_min, 1 beta_max
#         beta_min = 0.1
#         beta_max = 20
#         return (beta_max - beta_min) * t + beta_min

#     def VP_path(x, z, t):
#         # t in zeros and ones
#         # if noi
#         beta_min = 0.1
#         beta_max = 20
#         # u = 1 - t
#         # t = 1 - t # Reverse time, x0 for sample, x1 for noise
#         log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
#         # log_mean_coeff.to(x.device)
#         mean = torch.exp(log_mean_coeff[:, None]) * x
#         std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

#         # t-->1, log_mean_coeff-->0, mean-->0, std-->1 GAUSSIAN
#         # t-->0, log_mean_coeff-->-0.5*beta_min, mean-->1*x, std-->sqrt(1-exp(-beta_min) DATA

#         return mean + std[:, None] * z

#     # M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)  # add epsilon to stable it
#     # M_para = M_para[:, None]
#     x_t = VP_path(x, z, t)
#     vector = (
#         torch.exp(-T(t))[:, None] * x_t
#         - torch.exp(-0.5 * T(t))[:, None] * x
#     )


#     return -vector
