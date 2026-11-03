"""GPU distance Jacobian for the virtual serial chain after one weld opens.

The five intact welds make the six free-root modules a serial chain. Anchor
its inlet root; common rigid motion does not affect distance. Every hinge then
contributes axis x (source_endpoint - hinge_anchor) to the other endpoint.
This is a kinematic approximation to the compliant weld assembly, not a
collision-aware or force-feasible motion planner.
"""
import torch
from mjlab.utils.lab_api.math import quat_apply
from .model import TIP_NAMES


class DistanceJacobianController:
    def __init__(self,env,modules):
        self.env=env;self.modules=modules
        self.tip_ids=[m.find_bodies(TIP_NAMES[i])[0][0] for i,m in enumerate(modules)]
        self.command=torch.zeros(env.num_envs,6,device=env.device)
        self.predicted_speed=torch.zeros(env.num_envs,device=env.device)

    def jacobian(self,s):
        g=s.geometry();i=s.cfg.target_connector
        endpoint=self.modules[i].data.site_pos_w[:,s.out_sites[i]]
        anchors=torch.stack([m.data.body_link_pos_w[:,b] for m,b in zip(self.modules,self.tip_ids)],1)
        rotations=torch.stack([m.data.body_link_quat_w[:,b] for m,b in zip(self.modules,self.tip_ids)],1)
        z=torch.zeros_like(anchors);z[...,2]=1.
        axes=quat_apply(rotations,z)
        # g.delta is destination - source; the destination inlet is anchored.
        J=-torch.cross(axes,endpoint[:,None]-anchors,dim=-1).transpose(1,2)
        normal=quat_apply(g['q0'],z[:,0])
        direction=torch.where((g['distance']>1e-3)[:,None],
                              g['delta']/g['distance'][:,None].clamp_min(1e-8),normal)
        return torch.einsum('ni,nij->nj',direction,J),g['distance']

    def update(self,s,enabled):
        cfg=self.env.cfg
        row,distance=self.jacobian(s)
        speed=(cfg.separation_distance+cfg.separation_goal_buffer-distance).clamp_min(0.)*cfg.separation_distance_gain
        speed=speed.clamp(max=cfg.separation_cartesian_speed)
        desired=row*speed[:,None]/(row.square().sum(-1,keepdim=True)+cfg.separation_jacobian_damping**2)
        scale=(desired.abs().amax(-1,keepdim=True)/cfg.separation_max_target_speed).clamp_min(1.)
        desired=desired/scale
        step=cfg.separation_joint_acceleration*self.env.physics_dt
        self.command.add_((desired-self.command).clamp(-step,step))
        self.command.masked_fill_(~enabled[:,None],0.)
        self.predicted_speed.copy_((row*self.command).sum(-1))
        return self.command

    def reset(self,ids):
        self.command[ids]=0.;self.predicted_speed[ids]=0.
