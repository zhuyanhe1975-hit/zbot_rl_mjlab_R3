"""Per-world physics connector state for the full GPU reconstruction."""
import torch
from .batched_state import BatchedState
from .batched_collision import install_model_contact_filter
from .full_batched_fsm import FullBatchedFSM,P
from mjlab.utils.lab_api.math import quat_mul,quat_conjugate,quat_from_matrix

class FullBatchedState(BatchedState):
    def __init__(self,env):
        super().__init__(env)
        self.fsm=FullBatchedFSM(env.num_envs,env.device,dt=env.step_dt,refined=getattr(env.cfg,'full_refined_reference',False))
        if getattr(env.cfg,'full_online_headtail',False):
            self.fsm.phase_timeouts[P['CAPTURE_HEADTAIL']]=15.
        model=env.sim.mj_model
        self.port_ids=torch.tensor([[model.site(f'module_{i}/port_{side}').id for side in ('out','in')] for i in range(6)],device=env.device)
        self.capture_sites=[m.find_sites('port_out_rz270')[0][0] for m in self.modules[:5]]+[self.out_sites[5]]
        tail='module_5/port_out_refined90' if getattr(env.cfg,'full_refined_reference',False) else 'module_5/port_out'
        self.capture_port_ids=torch.tensor([model.site(f'module_{i}/port_out_rz270').id for i in range(5)]+[model.site(tail).id],device=env.device)
        self.site_body=torch.as_tensor(model.site_bodyid.copy(),device=env.device)
        self.port_roots=torch.tensor([m.data.indexing.root_body_id for m in self.modules],device=env.device)
        self.pair_ids=torch.tensor([env.sim.mj_model.pair(f'open_face_{i}').id for i in range(6)],device=env.device)
        self.phase_contacts=torch.zeros(env.num_envs,6,device=env.device,dtype=torch.bool)
        self.measurement_graph=None;self.measurement_ready=False
        install_model_contact_filter(env,phase_enabled=self.phase_contacts,include_headtail=True)
    def update(self):
        if not getattr(self.env.cfg,'full_cuda_graphs',False):
            return super().update()
        if self.last_substep==self.env._sim_step_counter:return
        if not self.measurement_ready:
            super().update();self.measurement_ready=True
        elif self.measurement_graph is None:
            self.last_substep=None
            self.measurement_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.measurement_graph):
                BatchedState.update(self)
            self.measurement_graph.replay();self.last_substep=self.env._sim_step_counter
        else:
            self.measurement_graph.replay();self.last_substep=self.env._sim_step_counter
    def outgoing_port_frames(self):
        data=self.env.sim.data;ids=self.port_ids[:,0]
        return data.site_xpos[:,ids],quat_from_matrix(data.site_xmat[:,ids])
    def update_separation(self,next_active):
        # Per-physics-step bookkeeping only needs the selected site distance;
        # avoid recomputing every module's full site pose and velocity.
        data=self.env.sim.data;target=self.target_indices();other=(target+1)%6
        a=data.site_xpos[self.worlds,self.port_ids[target,0]]
        b=data.site_xpos[self.worlds,self.port_ids[other,1]]
        self.max_target_gap.copy_(torch.maximum(self.max_target_gap,(b-a).norm(dim=-1)))
        self.safety_margin_violation|=((self.utilization>1.-self.cfg.minimum_other_margin)&self.other_mask()).any(-1)
    def target_indices(self):
        if not hasattr(self,'fsm'):
            return torch.full((self.env.num_envs,),5,device=self.env.device,dtype=torch.long)
        return self.fsm.target

    def terminal_ready(self,q,v,angle_tolerance):
        roots=torch.stack([m.data.root_link_pos_w for m in self.modules],1)
        axis=roots[:,-1]-roots[:,0]
        axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        offsets=roots-roots[:,0:1]
        perpendicular=offsets-(offsets*axis[:,None,:]).sum(-1,keepdim=True)*axis[:,None,:]
        return ((q.abs().amax(-1)<angle_tolerance)&(v.abs().amax(-1)<.05)&
                (self.geometry(index=5)['distance']>.05)&(perpendicular.norm(dim=-1).amax(-1)<.005))
    def reset(self,ids):
        ids=slice(None) if ids is None else ids
        super().reset(ids)
        self.env.sim.data.eq_active[ids,self.eq_ids[5]]=False
        self.fsm.reset(ids);self.phase_contacts[ids]=False
    def set_contact_phase(self):
        """Enable the opening face during rotation and final head-tail unfolding."""
        phase=self.fsm.phase;target=self.fsm.target
        self.phase_contacts.zero_()
        enabled=(phase==P['ROTATE'])|((target==5)&((phase==P['UNFOLD'])|(phase==P['FINAL_STRAIGHT_HOLD'])))
        if self.fsm.refined:enabled|=(phase==P['SEPARATE'])
        self.phase_contacts[self.worlds,target]=enabled

    def _selected_port_pair(self,capture=False,index=None):
        data=self.env.sim.data
        target=self.target_indices() if index is None else torch.full_like(self.index,index)
        other=(target+1)%6
        source=self.capture_port_ids[target] if capture else self.port_ids[target,0]
        destination=self.port_ids[other,1]
        sites=torch.stack((source,destination),-1);bodies=self.site_body[sites]
        pos=data.site_xpos[self.worlds[:,None],sites]
        quat=quat_from_matrix(data.site_xmat[self.worlds[:,None],sites])
        cvel=data.cvel[self.worlds[:,None],bodies]
        roots=torch.stack((target,other),-1)
        com=data.subtree_com[self.worlds[:,None],self.port_roots[roots]]
        velocity=cvel[...,3:6]-torch.cross(cvel[...,:3],com-pos,dim=-1)
        return pos,quat,velocity,cvel[...,:3]
    def geometry(self,index=None):
        pos,quat,velocity,omega=self._selected_port_pair(index=index)
        p0,p1=pos[:,0],pos[:,1];q0,q1=quat[:,0],quat[:,1]
        relative=quat_mul(quat_conjugate(q0),q1);relative=torch.where(relative[:,:1]<0,-relative,relative)
        delta=p1-p0
        return {'delta':delta,'distance':delta.norm(dim=-1),'q0':q0,'q1':q1,'relative_quat':relative,
                'angle':2*torch.atan2(relative[:,1:].norm(dim=-1),relative[:,0].abs()),
                'dv':velocity[:,1]-velocity[:,0],'dw':omega[:,1]-omega[:,0]}
    def capture_geometry(self):
        """Use the actual equality sites, including the rotated registration offset."""
        pos,quat,velocity,omega=self._selected_port_pair(capture=True)
        p0,p1=pos[:,0],pos[:,1];q0,q1=quat[:,0],quat[:,1]
        relative=quat_mul(quat_conjugate(q1),q0)
        return {'delta':p1-p0,'q0':q0,'q1':q1,'gap':(p1-p0).norm(dim=-1),
                'angle':2*torch.atan2(relative[:,1:].norm(dim=-1),relative[:,0].abs()),
                'linear_speed':(velocity[:,1]-velocity[:,0]).norm(dim=-1),
                'angular_speed':(omega[:,1]-omega[:,0]).norm(dim=-1),
                'dv':velocity[:,1]-velocity[:,0],'dw':omega[:,1]-omega[:,0]}
