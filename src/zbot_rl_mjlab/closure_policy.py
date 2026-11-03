"""Bounded actions and control-phase-only PPO for initial closure."""
import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import BetaDistribution


class ClosureBetaDistribution(BetaDistribution):
    """Unimodal bounded policy; deterministic output is its executed mean.

    Concentrations >=2 avoid endpoint-heavy exploration. Unlike clipped
    Gaussians, every sampled action is already in the actuator action range.
    """
    def __init__(self,output_dim,initial_concentration=6.):
        super().__init__(output_dim,action_range=(-1.,1.))
        if not 2.<initial_concentration<=1000.:
            raise ValueError('initial_concentration must be in (2,1000]')
        self.initial_concentration=initial_concentration
        self.statistics_only=False

    @staticmethod
    def concentrations(output):
        return (torch.nn.functional.softplus(output)+2.).clamp_max(1000.)

    def update(self,mlp_output):
        self._alpha,self._beta=self.concentrations(mlp_output).unbind(dim=-2)
        self._distribution=torch.distributions.Beta(self._alpha,self._beta)

    def deterministic_output(self,mlp_output):
        alpha,beta=self.concentrations(mlp_output).unbind(dim=-2)
        return 2.*alpha/(alpha+beta)-1.

    def sample(self):
        # RSL calls actor(stochastic_output=True) during PPO updates only to
        # refresh likelihood/entropy. Its returned draw is unused there.
        return self.mean if self.statistics_only else super().sample()

    @property
    def entropy(self):
        # Native Beta entropy is in [0,1] coordinates; actions live in [-1,1].
        return super().entropy+self.output_dim*math.log(2.)

    def init_mlp_weights(self,mlp):
        head=[m for m in mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(head.weight)
        raw=math.log(math.expm1(self.initial_concentration-2.))
        torch.nn.init.constant_(head.bias,raw)

    def as_deterministic_output_module(self):
        return _ClosureBetaMean()


class _ClosureBetaMean(torch.nn.Module):
    def forward(self,output):
        alpha,beta=(torch.nn.functional.softplus(output)+2.).clamp_max(1000.).unbind(dim=-2)
        return 2.*alpha/(alpha+beta)-1.


def control_phase_advantages(raw,active):
    """Zero actor credit outside CURL, normalize only controllable samples."""
    mask=active.to(raw.dtype)
    count=mask.sum().clamp_min(1.)
    center=(raw*mask).sum()/count
    variance=((raw-center).square()*mask).sum()/count
    scale=mask.numel()/count
    return torch.where(active,(raw-center)/(variance.sqrt()+1e-8)*scale,0.)


class ClosurePPO(PPO):
    def compute_returns(self,obs):
        super().compute_returns(obs)
        # First 41 channels stay unchanged; index24 is the CURL indicator.
        self.actor_active=(self.storage.observations['actor'][...,24:25]>.5)
        self.storage.advantages=control_phase_advantages(
            self.storage.returns-self.storage.values,self.actor_active)

    def update(self):
        inactive=not bool(self.actor_active.any())
        parameters=list(self.actor.parameters())
        if inactive:
            # Zero gradients alone do not stop Adam's existing momentum.
            for parameter in parameters:parameter.requires_grad_(False)
        self.actor.distribution.statistics_only=True
        try:
            result=super().update()
        finally:
            self.actor.distribution.statistics_only=False
            if inactive:
                for parameter in parameters:parameter.requires_grad_(True)
        result['actor_active_fraction']=float(self.actor_active.float().mean())
        return result
