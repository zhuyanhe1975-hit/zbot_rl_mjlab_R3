"""Batched GPU constraint-Jacobian loading with measured-wrench feedback."""
import numpy as np
import torch

from .disconnect.state import port_wrench


def peel(w, radius):
    return w[..., 2].clamp_min(0.) + w[..., 3:5].norm(dim=-1) / radius


def projected_increments(A, w, torque, target, radius, iterations=20, goal_limit=84., other_limit=56.):
    """Search tensile/bending support directions; retain only safe predictions.

    This replaces CPU SLSQP with bounded GPU projections. Local predictions
    remain approximate; the physical 64 N latch is independently enforced.
    """
    b = len(w); dev=w.device; dtype=w.dtype
    angles=torch.arange(16,device=dev,dtype=dtype)*(2*torch.pi/16)
    axis=torch.stack((angles.cos(),angles.sin()),-1).repeat_interleave(2,0)
    tension=torch.arange(32,device=dev,dtype=dtype).remainder(2)
    worlds=torch.arange(b,device=dev)
    At=A[worlds,target];wt=w[worlds,target]
    row=tension[None,:,None]*At[:,None,2]+torch.einsum('ca,baj->bcj',axis,At[:,3:5])/radius
    value=tension[None]*wt[:,None,2].clamp_min(0.)+torch.einsum('ca,ba->bc',axis,wt[:,3:5])/radius
    current=peel(w,radius);goal=(current[worlds,target]+3.).clamp(12.,goal_limit)
    low=torch.maximum(torch.full_like(torque,-.20),-14.-torque)[:,None]
    high=torch.minimum(torch.full_like(torque,.20),14.-torque)[:,None]
    delta=(row*(goal[:,None]-value)[:,:,None]/row.square().sum(-1,keepdim=True).clamp_min(1e-12)).clamp(low,high)
    other=(torch.arange(6,device=dev)[None]!=target[:,None])
    for _ in range(iterations):
        estimated=w[:,None]+torch.einsum('bfwa,bca->bcfw',A,delta)
        loads=peel(estimated,radius)
        bend=estimated[...,3:5]/estimated[...,3:5].norm(dim=-1,keepdim=True).clamp_min(1e-8)
        gradient=(estimated[...,2]>0.)[:,:,:,None]*A[:,None,:,2]+torch.einsum('bcfi,bfij->bcfj',bend,A[:,:,3:5])/radius
        violation=(loads-other_limit).clamp_min(0.)*other[:,None]
        correction=(violation[:,:,:,None]*gradient/gradient.square().sum(-1,keepdim=True).clamp_min(1e-10)).sum(2)
        shortfall=(goal[:,None]-value-(row*delta).sum(-1)).clamp_min(0.)
        correction-=shortfall[:,:,None]*row/row.square().sum(-1,keepdim=True).clamp_min(1e-10)
        count=(violation>0.).sum(-1)+(shortfall>0.)
        delta=(delta-correction/count.clamp_min(1)[:,:,None]).clamp(low,high)
    estimated=w[:,None]+torch.einsum('bfwa,bca->bcfw',A,delta)
    loads=peel(estimated,radius)
    safe_end=torch.where(other[:,None],loads,0.).amax(-1)<=other_limit+1e-4
    safe_start=torch.where(other,current,0.).amax(-1)<=other_limit
    left=torch.zeros_like(value);right=torch.ones_like(value)
    response=torch.einsum('bfwa,bca->bcfw',A,delta)
    for _ in range(10):
        middle=(left+right)*.5
        trial=peel(w[:,None]+response*middle[:,:,None,None],radius)
        safe_middle=torch.where(other[:,None],trial,0.).amax(-1)<=other_limit
        left=torch.where(safe_middle,middle,left);right=torch.where(safe_middle,right,middle)
    scale=torch.where(~safe_end&safe_start[:,None],left,1.)
    delta=delta*scale[:,:,None]
    estimated=w[:,None]+torch.einsum('bfwa,bca->bcfw',A,delta)
    loads=peel(estimated,radius)
    safe=(torch.where(other[:,None],loads,0.).amax(-1)<=other_limit+1e-4)&torch.isfinite(delta).all(-1)
    full=safe&((value+(row*delta).sum(-1))>=goal[:,None]-1e-3)
    objective=delta.square().sum(-1)+.002*(torque[:,None]+delta).square().sum(-1)+.00002*torch.where(other[:,None],loads.square(),0.).sum(-1)
    best_full=torch.where(full,objective,torch.inf).argmin(-1)
    target_load=loads.gather(2,target[:,None,None].expand(-1,32,1)).squeeze(-1)
    best_partial=torch.where(safe,target_load,-torch.inf).argmax(-1)
    choice=torch.where(full.any(-1),best_full,best_partial)
    selected=delta[worlds,choice]
    improves=target_load[worlds,choice]>current[worlds,target]+1e-4
    usable=full.any(-1)|(safe.any(-1)&improves)
    retreat=(-torque).clamp(-.20,.20)
    retreat_load=peel(w+torch.einsum('bfwa,ba->bfw',A,retreat),radius)
    old_peak=torch.where(other,current,0.).amax(-1)
    better=torch.where(other,retreat_load,0.).amax(-1)<old_peak
    fallback=torch.where(better[:,None],retreat,0.)
    result=torch.where(usable[:,None],selected,fallback)
    return result, usable, peel(w+torch.einsum('bfwa,ba->bfw',A,result),radius)


class BatchedLoading:
    def __init__(self,env,async_linalg=False):
        self.env=env;self.async_linalg=async_linalg;m=env.sim.mj_model;dev=env.device
        self.rows=torch.as_tensor(np.repeat(np.arange(m.nv),m.M_rownnz),device=dev,dtype=torch.long)
        self.cols=torch.as_tensor(m.M_colind.copy(),device=dev,dtype=torch.long)
        self.B=torch.zeros(m.nv,6,device=dev,dtype=torch.float64)
        for i in range(6):self.B[m.jnt_dofadr[m.joint(f'module_{i}/joint{i+1}').id],i]=1.
        self.torque=torch.zeros(env.num_envs,6,device=dev)
        self.predicted_loads=torch.zeros_like(self.torque)
        self.feasible=torch.zeros(env.num_envs,device=dev,dtype=torch.bool)
        self.report={'method':'gpu_measured_constraint_J_M_R_projected_support_search','other_prediction_limit_N':float(getattr(env.cfg,'full_loading_other_limit',56.)),'chunk_size':512}

    def reset(self,ids):
        if isinstance(ids,torch.Tensor) and ids.dtype==torch.bool:
            self.torque.masked_fill_(ids[:,None],0.);self.predicted_loads.masked_fill_(ids[:,None],0.);self.feasible.masked_fill_(ids,False)
        else:
            self.torque[ids]=0.;self.predicted_loads[ids]=0.;self.feasible[ids]=False

    def response(self,s,ids):
        e=self.env;d=e.sim.data;efc=d.efc;nv=e.sim.mj_model.nv;dev=e.device
        # Slice only the current active row bound, with one host scalar per
        # batch chunk, rather than a CPU forward/optimization per environment.
        k=max(1,int(d.nefc[ids].max().item()))
        J=efc.J[ids,:k,:nv].double();b=len(ids)
        M=torch.zeros(b,nv,nv,device=dev,dtype=torch.float64)
        M[:,self.rows,self.cols]=d.M[ids].double()
        M=M+M.transpose(1,2)-torch.diag_embed(M.diagonal(dim1=-2,dim2=-1))
        active=(torch.arange(k,device=dev)[None]<d.nefc[ids,None])&((efc.type[ids,:k]==0)|(efc.state[ids,:k]==1))
        Ja=J*active[:,:,None]
        rhs=torch.cat((Ja.transpose(1,2),self.B.expand(b,-1,-1)),-1)
        if self.async_linalg:
            inverse_rhs,info_m=torch.linalg.solve_ex(M,rhs,check_errors=False)
        else:
            inverse_rhs=torch.linalg.solve(M,rhs);info_m=None
        Minv_J=inverse_rhs[:,:,:k];Minv_B=inverse_rhs[:,:,k:]
        R=torch.where(active,efc.D[ids,:k].double().clamp_min(1e-15).reciprocal(),1.)
        H=Ja@Minv_J+torch.diag_embed(R)
        if self.async_linalg:
            sensitivity,info_h=torch.linalg.solve_ex(H,Ja@Minv_B,check_errors=False)
            sensitivity=-sensitivity
        else:
            sensitivity=-torch.linalg.solve(H,Ja@Minv_B);info_h=None
        eq=s.constraint_ids()[ids]
        row=torch.arange(k,device=dev)[None,None]
        mask=(row<d.nefc[ids,None,None])&(efc.type[ids,None,:k]==0)&(efc.id[ids,None,:k]==eq[:,:,None])
        first=torch.where(mask,row,k).amin(-1)
        present=(mask.sum(-1)==6).all(-1)
        if self.async_linalg:present&=(info_m==0)&(info_h==0)
        indices=(first[:,:,None]+torch.arange(6,device=dev)[None,None]).clamp_max(k-1)
        worlds=torch.arange(b,device=dev)
        local=J[worlds[:,None,None,None],indices[:,:,:,None],s.root_dofs[None,:,None,:]]
        G=local.transpose(-1,-2)@sensitivity[worlds[:,None,None],indices]
        force=efc.force[ids,:k].double()[worlds[:,None,None],indices]
        baseline=(local*force[:,:,:,None]).sum(2)
        roots=torch.stack([m.data.root_link_pos_w[ids] for m in s.modules],1).double()
        rootq=torch.stack([m.data.root_link_quat_w[ids] for m in s.modules],1).double()
        ports=torch.stack([m.data.site_pos_w[ids,i] for m,i in zip(s.modules,s.out_sites)],1).double()
        portq=torch.stack([m.data.site_quat_w[ids,i] for m,i in zip(s.modules,s.out_sites)],1).double()
        f,t=port_wrench(baseline.reshape(-1,6),rootq.reshape(-1,4),roots.reshape(-1,3),ports.reshape(-1,3),portq.reshape(-1,4))
        baseline=torch.cat((f,t),-1).reshape(b,6,6)
        def repeat(x):return x[:,:,None].expand(-1,-1,6,-1).reshape(-1,x.shape[-1])
        f,t=port_wrench(G.transpose(-1,-2).reshape(-1,6),repeat(rootq),repeat(roots),repeat(ports),repeat(portq))
        A=torch.cat((f,t),-1).reshape(b,6,6,6).transpose(-1,-2)
        return A,baseline,present

    def update(self,s,loading):
        ids=torch.where(loading)[0]
        for chunk in ids.split(512):
            if not len(chunk):continue
            A,baseline,present=self.response(s,chunk)
            measured=torch.cat((s.force[chunk],s.moment[chunk]),-1).double()
            w=torch.where(s.has_sample[chunk,None,None],measured,baseline)
            valid=present&torch.isfinite(A).all(dim=(1,2,3))&torch.isfinite(w).all(dim=(1,2))
            delta,feasible,predicted=projected_increments(torch.nan_to_num(A).float(),torch.nan_to_num(w).float(),
                self.torque[chunk],s.target_indices()[chunk],self.env.cfg.connector_radius,
                goal_limit=1.05*self.env.cfg.break_force,
                other_limit=float(getattr(self.env.cfg,'full_loading_other_limit',56.)))
            self.torque[chunk]+=torch.where(valid[:,None],delta,0.)
            self.predicted_loads[chunk]=predicted;self.feasible[chunk]=feasible&valid
        return self.torque
