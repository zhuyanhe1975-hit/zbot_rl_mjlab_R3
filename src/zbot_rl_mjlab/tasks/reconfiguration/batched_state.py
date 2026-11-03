"""Per-world connector bookkeeping and collision monitoring for mixed PPO."""
import math
import torch
from mjlab.utils.lab_api.math import quat_mul,quat_conjugate

from .disconnect.state import ConnectorState
from .mixed_closed_dynamic import add_mixed_connectors


def add_batched_connectors(spec):
    add_mixed_connectors(spec)
    # Explicit pairs bypass static flange exclusions. A model-scoped GPU
    # filter removes their solver flag only while the corresponding weld is on.
    for i in range(5):
        spec.add_pair(name=f'open_face_{i}',geomname1=f'module_{i}/b{i+1}_geom',
                      geomname2=f'module_{i+1}/a{i+2}_geom',condim=3,margin=0.,gap=0.,
                      friction=(.6,.6,.1,.1,.1))


class BatchedState(ConnectorState):
    def __init__(self,env):
        super().__init__(env);dev=env.device;n=env.num_envs;m=env.sim.mj_model
        self.index=torch.zeros(n,dtype=torch.long,device=dev)
        self.order=torch.tensor([0,1,3,4],device=dev)
        self.variant_ids=torch.tensor([m.equality(f'connector_{i}_rz270').id for i in range(5)],device=dev)
        self.pair_ids=torch.tensor([m.pair(f'open_face_{i}').id for i in range(5)],device=dev)
        names=[m.body(int(b)).name for b in m.geom_bodyid]
        self.geom_module=torch.tensor([int(name.split('/')[0][7:]) if name.startswith('module_') else -1 for name in names],device=dev)
        self.contact_violation=torch.zeros(n,device=dev,dtype=torch.bool)
        self.has_sample=torch.zeros_like(self.contact_violation)
        self.max_module_depth=torch.zeros(n,device=dev)
        self.max_ground_depth=torch.zeros(n,device=dev)
        self.max_headtail_depth=torch.zeros(n,device=dev)
        self.headtail_penetration=torch.zeros(n,device=dev,dtype=torch.bool)
        self.max_other_load=torch.zeros(n,device=dev)
        self.loads_at_break=torch.zeros(n,6,device=dev)

    def target_indices(self):
        return self.order[self.index.clamp_max(3)]

    @property
    def new_eq(self):
        return self.variant_ids[self.target_indices()]

    def constraint_ids(self):
        ids=super().constraint_ids().clone()
        on=self.env.sim.data.eq_active[:,self.variant_ids].bool()
        ids[:,:5]=torch.where(on,self.variant_ids[None],ids[:,:5])
        return ids

    def reset(self,ids):
        ids=slice(None) if ids is None else ids
        self.index[ids]=0
        super().reset(ids)
        if isinstance(ids,slice):self.env.sim.data.eq_active[:,self.variant_ids]=False
        else:self.env.sim.data.eq_active[ids[:,None],self.variant_ids[None]]=False
        for v in (self.contact_violation,self.has_sample,self.max_module_depth,self.max_ground_depth,self.max_headtail_depth,self.headtail_penetration,self.max_other_load,self.loads_at_break):v[ids]=0

    def geometry(self,index=None):
        target=self.target_indices() if index is None else torch.full_like(self.index,index)
        other=(target+1)%6
        def gather(name,sites,indices):
            values=torch.stack([getattr(m.data,name)[:,site] for m,site in zip(self.modules,sites)],1)
            return values[self.worlds,indices]
        p0=gather('site_pos_w',self.out_sites,target);p1=gather('site_pos_w',self.in_sites,other)
        q0=gather('site_quat_w',self.out_sites,target);q1=gather('site_quat_w',self.in_sites,other)
        relative=quat_mul(quat_conjugate(q0),q1);relative=torch.where(relative[:,:1]<0,-relative,relative)
        dv=gather('site_lin_vel_w',self.in_sites,other)-gather('site_lin_vel_w',self.out_sites,target)
        dw=gather('site_ang_vel_w',self.in_sites,other)-gather('site_ang_vel_w',self.out_sites,target)
        delta=p1-p0
        return {'delta':delta,'distance':delta.norm(dim=-1),'q0':q0,'q1':q1,'relative_quat':relative,
                'angle':2*torch.atan2(relative[:,1:].norm(dim=-1),relative[:,0].abs()),'dv':dv,'dw':dw}

    def other_margin(self):
        other=self.other_mask()
        margin=1.-torch.where(other,self.utilization,0.).amax(-1)
        return torch.where((self.connected()|~other).all(-1),margin,-1.)

    def update_separation(self,next_active):
        self.max_target_gap.copy_(torch.maximum(self.max_target_gap,self.geometry()['distance']))
        self.safety_margin_violation|=((self.utilization>1.-self.cfg.minimum_other_margin)&self.other_mask()).any(-1)

    def update(self):
        if self.last_substep==self.env._sim_step_counter:return
        target=self.target_indices();before=self.connected()[self.worlds,target].clone()
        super().update()
        broke=before&~self.connected()[self.worlds,target]
        self.loads_at_break.copy_(torch.where(broke[:,None],self.utilization*self.cfg.break_force,self.loads_at_break))
        self.max_other_load.copy_(torch.maximum(self.max_other_load,torch.where(self.other_mask(),self.utilization,0.).amax(-1)*self.cfg.break_force))
        d=self.env.sim.data;c=d.contact;n=self.env.num_envs
        present=(torch.arange(c.dist.numel(),device=self.env.device)<d.nacon[0])&(c.worldid>=0)&(c.worldid<n)
        valid=present&((c.type.int()&1)!=0)
        ids=c.worldid.long().clamp(0,n-1)
        geom=c.geom.long().clamp(0,len(self.geom_module)-1);mods=self.geom_module[geom]
        robot=valid&(mods[:,0]>=0)&(mods[:,1]>=0)&(mods[:,0]!=mods[:,1])
        ground=valid&((mods[:,0]>=0)^(mods[:,1]>=0))
        depth=(-c.dist).clamp_min(0.)
        if getattr(self.env.cfg,'full_headtail_penetration_guard',False):
            is_headtail=(((mods[:,0]==0)&(mods[:,1]==5))|((mods[:,0]==5)&(mods[:,1]==0)))
            # The dedicated head-tail rule checks alignment, including contacts
            # whose solver bit is removed by the phase filter.
            robot &= ~is_headtail
        self.max_module_depth.scatter_reduce_(0,ids,torch.where(robot,depth,0.),reduce='amax',include_self=True)
        self.max_ground_depth.scatter_reduce_(0,ids,torch.where(ground,depth,0.),reduce='amax',include_self=True)
        self.contact_violation|=(self.max_module_depth>.002)|(self.max_ground_depth>.003)
        if getattr(self.env.cfg,'full_headtail_penetration_guard',False):
            # Inspect even filtered contacts (type=0); otherwise welded/phase
            # filtering can conceal the very head-tail overlap we must reject.
            source=self.capture_port_ids[5:6];destination=self.port_ids[0:1,1]
            delta=d.site_xpos[:,source]-d.site_xpos[:,destination]
            pos_ok=delta.square().sum(dim=(-1,-2))<self.env.cfg.full_headtail_alignment_position_tolerance**2
            a=d.site_xmat[:,source];b=d.site_xmat[:,destination]
            cosine=((a*b).sum(dim=(-1,-2,-3))-1.)*.5
            angle_ok=cosine>math.cos(self.env.cfg.full_headtail_alignment_angle_tolerance)
            headtail=present&is_headtail&~(pos_ok&angle_ok)[ids]&(c.dist<=0.)
            self.max_headtail_depth.scatter_reduce_(0,ids,torch.where(headtail,depth,0.),reduce='amax',include_self=True)
            hits=torch.zeros_like(self.headtail_penetration,dtype=torch.int32)
            hits.scatter_reduce_(0,ids,headtail.int(),reduce='amax',include_self=True)
            self.headtail_penetration |= hits.bool()
            self.contact_violation |= self.headtail_penetration
        self.has_sample.fill_(True)
