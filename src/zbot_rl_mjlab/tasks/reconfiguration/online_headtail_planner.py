"""GPU feedback planner for initial closure, without a time-indexed reference."""
import torch
from mjlab.utils.lab_api.math import quat_apply,quat_conjugate,quat_mul
from .state_machine import rotvec


class OnlineHeadtailPlanner:
    def __init__(self,env):
        self.env=env;m=env.sim.mj_model;dev=env.device
        self.joints=torch.tensor([m.joint(f'module_{i}/joint{i+1}').id for i in range(6)],device=dev)
        self.dofs=torch.tensor([int(m.jnt_dofadr[m.joint(f'module_{i}/joint{i+1}').id]) for i in range(6)],device=dev)
        self.distance_ids=torch.tensor([int(m.sensor_adr[m.sensor(f'headtail_distance_{i}').id]) for i in range(4)],device=dev)
        self.point_ids=torch.tensor([[int(m.sensor_adr[m.sensor(f'headtail_fromto_{i}').id])+j for j in range(6)] for i in range(4)],device=dev)
        # Fixed inverse-kinematics endpoint, not a recorded trajectory.
        self.goal=torch.tensor([-.93612577,2.03228742,-1.79854868,1.28187906,-1.58940248,2.44844829],device=dev)
        self.speed=.32;self.clearance=.0001;self.activation=.05;self.gain=2.
        self._head_mask=torch.tensor([[0,0,0,0,0,0],[0,0,0,0,0,0],
                                      [1,0,0,0,0,0],[1,0,0,0,0,0]],device=dev,dtype=torch.bool)
        self._tail_mask=torch.tensor([[1,1,1,1,1,0],[1,1,1,1,1,1],
                                      [1,1,1,1,1,0],[1,1,1,1,1,1]],device=dev,dtype=torch.bool)

    def calculate(self,s,q,residual,clock,previous):
        d=self.env.sim.data;pose=s.capture_geometry();axis=d.xaxis[:,self.joints];anchor=d.xanchor[:,self.joints]
        source=d.site_xpos[:,s.capture_port_ids[5:6]].squeeze(1)
        linear=torch.cross(axis,source[:,None,:]-anchor,dim=-1).transpose(1,2)
        relative=quat_mul(quat_conjugate(pose['q1']),pose['q0'])
        orientation=-quat_apply(pose['q1'],rotvec(relative))
        J=torch.cat((linear,axis.transpose(1,2)*.1),dim=1)
        goal=torch.cat((pose['delta']*5.,orientation*.5),dim=-1)
        J=J*5.
        eye=torch.eye(6,device=q.device,dtype=q.dtype)[None]
        vpose,_=torch.linalg.solve_ex(J.transpose(1,2)@J+.03*eye,(J.transpose(1,2)@goal[...,None]),check_errors=False)
        vpose=vpose.squeeze(-1)
        blend=((.18-pose['gap'])/.12).clamp(0.,1.)[:,None]
        v=((1.-blend)*(1.4*(self.goal-q)).clamp(-self.speed,self.speed)+blend*vpose.clamp(-self.speed,self.speed))
        v=(v*(1.+.5*clock[:,None])+residual).clamp(-self.speed,self.speed)
        load=s.utilization[:,:5].amax(-1)*self.env.cfg.break_force
        load_scale=((65.-load)/30.).clamp(0.,1.)
        v=v*load_scale[:,None]
        v=previous+(v-previous).clamp(-.075*self.env.physics_dt,.075*self.env.physics_dt)
        distances=d.sensordata[:,self.distance_ids];points=d.sensordata[:,self.point_ids]
        p0,p1=points[...,:3],points[...,3:]
        normal=p1-p0;norm=normal.norm(dim=-1,keepdim=True)
        normal=normal/norm.clamp_min(1e-8)
        ax=axis[:,None,:,:];anc=anchor[:,None,:,:]
        j0=torch.cross(ax,p0[:,:,None,:]-anc,dim=-1)*self._head_mask[None,:,:,None]
        j1=torch.cross(ax,p1[:,:,None,:]-anc,dim=-1)*self._tail_mask[None,:,:,None]
        C=((j1-j0)*normal[:,:,None,:]).sum(-1)
        active=(distances<self.activation)&(norm.squeeze(-1)>1e-8)
        closing=-(C*d.qvel[:,self.dofs][:,None,:]).sum(-1)
        margin=self.clearance+.005*((pose['gap']-.001)/.02).clamp(0.,1.)[:,None]+.15*closing.clamp_min(0.)
        limit=self.gain*(distances-margin)
        # Cyclic half-space projections keep policy residuals behind the same
        # collision barrier. A failed projection commands zero, not unsafe speed.
        for _ in range(8):
            for i in range(4):
                row=C[:,i];slack=(row*v).sum(-1)+limit[:,i]
                correction=(-slack).clamp_min(0.)*active[:,i]/row.square().sum(-1).clamp_min(1e-8)
                v=(v+correction[:,None]*row).clamp(-self.speed,self.speed)
        feasible=(((C*v[:,None,:]).sum(-1)+limit>=-1e-5)|~active).all(-1)
        valid=torch.isfinite(v).all(-1)&torch.isfinite(distances).all(-1)
        return torch.where((feasible&valid)[:,None],v,0.)
