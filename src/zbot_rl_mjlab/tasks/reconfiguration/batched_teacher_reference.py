"""Read CPU teacher states for independent asynchronous GPU environments."""
from pathlib import Path
import numpy as np
import torch
from .full_cpu_student import PHASES


class BatchedTeacherReference:
    """Match each world's phase, target, stage and phase time to a teacher sample.

    Nearest-neighbour lookup is batched and remains on the requested device;
    worlds whose exact phase/target/stage combination is absent are invalid.
    """
    def __init__(self, path, device):
        with np.load(Path(path), allow_pickle=False) as data:
            if tuple(data['phase_names']) != PHASES:
                raise ValueError('teacher phase vocabulary mismatch')
            phase=data['phase'].astype(np.int64);target=data['target'].astype(np.int64)
            stage=data['stage'].astype(np.int64);times=data['phase_time'].astype(np.float64)
            if np.any((phase<0)|(phase>=len(PHASES))|(target<0)|(target>=6)|(stage<0)|(stage>=5)):
                raise ValueError('teacher indices out of range')
            if not np.isfinite(times).all() or np.any(times<0) or np.any(times>=1000):
                raise ValueError('invalid teacher phase time')
            key=phase*30+target*5+stage
            order=np.lexsort((times,key))
            self.key=torch.as_tensor(key[order],device=device,dtype=torch.long)
            self.time=torch.as_tensor(times[order],device=device,dtype=torch.float64)
            # Distinct key ranges permit one global searchsorted on GPU.
            self.index=self.key.to(torch.float64)*1000.+self.time
            self.q=torch.as_tensor(data['q'][order],device=device,dtype=torch.float32)
            self.qvel=torch.as_tensor(data['qvel'][order],device=device,dtype=torch.float32)
            self.qref=torch.as_tensor(data['qref'][order],device=device,dtype=torch.float32)
            self.has_control='ctrl' in data
            self.has_connected='connected' in data
            t=times[order];k=key[order];qref=data['qref'][order]
            self.control=torch.as_tensor(data['ctrl'][order] if self.has_control else np.zeros_like(qref),device=device,dtype=torch.float32)
            # The collector can sample after a passive break but before the
            # FSM changes phase. At that point ctrl is already a position
            # target, not a loading torque. Extend the last valid torque ramp
            # across those samples rather than replaying a different mode.
            load_control=np.array(data['ctrl'][order] if self.has_control else np.zeros_like(qref),dtype=np.float32,copy=True)
            tail_time=np.zeros(len(PHASES)*30,dtype=np.float32)
            tail_value=np.zeros((len(PHASES)*30,6),dtype=np.float32)
            tail_slope=np.zeros_like(tail_value)
            tail_valid=np.zeros(len(PHASES)*30,dtype=bool)
            if self.has_control and self.has_connected:
                connected=data['connected'][order].astype(bool)
                sorted_target=target[order];sorted_phase=phase[order]
                is_loading=np.isin(sorted_phase,[PHASES.index('LOAD'),PHASES.index('LOAD_HEADTAIL')])
                for group in np.unique(k[is_loading]):
                    ids=np.flatnonzero((k==group)&is_loading)
                    valid=ids[connected[ids,sorted_target[ids]]]
                    if len(valid)<2:continue
                    last,prev=valid[-1],valid[-2];dt=t[last]-t[prev]
                    if dt<=1e-8:continue
                    tail_time[group]=t[last];tail_value[group]=load_control[last]
                    tail_slope[group]=(load_control[last]-load_control[prev])/dt;tail_valid[group]=True
                    invalid=ids[(ids>last)&~connected[ids,sorted_target[ids]]]
                    load_control[invalid]=np.clip(load_control[last]+(t[invalid]-t[last])[:,None]*(load_control[last]-load_control[prev])/dt,-14.,14.)
            self.load_control=torch.as_tensor(load_control,device=device,dtype=torch.float32)
            self.tail_time=torch.as_tensor(tail_time,device=device)
            self.tail_value=torch.as_tensor(tail_value,device=device)
            self.tail_slope=torch.as_tensor(tail_slope,device=device)
            self.tail_valid=torch.as_tensor(tail_valid,device=device)
            slope=np.zeros_like(qref,dtype=np.float32)
            valid=(k[1:]==k[:-1]) & ((t[1:]-t[:-1])>1e-8)
            slope[1:][valid]=(qref[1:][valid]-qref[:-1][valid])/(t[1:][valid]-t[:-1][valid])[:,None]
            self.qref_velocity=torch.as_tensor(slope,device=device,dtype=torch.float32)
            self.count=len(order)
            if not self.count:raise ValueError('empty teacher reference')

    def sample(self,phase,target,stage,phase_time):
        if not (phase.shape==target.shape==stage.shape==phase_time.shape):
            raise ValueError('teacher query shapes must match')
        valid_query=(phase>=0)&(phase<len(PHASES))&(target>=0)&(target<6)&(stage>=0)&(stage<5)&torch.isfinite(phase_time)&(phase_time>=0)&(phase_time<1000)
        key=phase.long()*30+target.long()*5+stage.long()
        query=key.to(torch.float64)*1000.+torch.nan_to_num(phase_time.to(torch.float64),nan=0.).clamp(0.,999.999)
        right=torch.searchsorted(self.index,query).clamp(0,self.count-1)
        left=(right-1).clamp(0,self.count-1)
        dl=torch.where(self.key[left]==key,(self.time[left]-phase_time).abs(),torch.inf)
        dr=torch.where(self.key[right]==key,(self.time[right]-phase_time).abs(),torch.inf)
        chosen=torch.where(dl<=dr,left,right)
        valid=valid_query&((self.key[chosen]==key))
        def select(x):return torch.where(valid[...,None],x[chosen],torch.zeros_like(x[chosen]))
        pair=valid_query&(self.key[left]==key)&(self.key[right]==key)&(self.time[right]>self.time[left])
        dt=(self.time[right]-self.time[left]).clamp_min(1e-8)
        blend=((phase_time-self.time[left])/dt).clamp(0.,1.).to(self.qref.dtype)
        interp=torch.lerp(self.qref[left],self.qref[right],blend[...,None])
        slope=(self.qref[right]-self.qref[left])/dt.to(self.qref.dtype)[...,None]
        qref_interp=torch.where(pair[...,None],interp,select(self.qref))
        qref_slope=torch.where(pair[...,None],slope,torch.zeros_like(slope))
        index=key.clamp(0,len(PHASES)*30-1)
        extension=(phase_time-self.tail_time[index]).clamp(0.,.2).to(self.tail_value.dtype)
        extended=(self.tail_value[index]+extension[...,None]*self.tail_slope[index]).clamp(-14.,14.)
        extend=valid&self.tail_valid[index]&(phase_time>self.tail_time[index])
        load_extended=torch.where(extend[...,None],extended,select(self.load_control))
        return {'q':select(self.q),'qvel':select(self.qvel),'qref':select(self.qref),'qref_velocity':select(self.qref_velocity),'qref_interp':qref_interp,'qref_slope':qref_slope,'ctrl':select(self.control),'load_ctrl':select(self.load_control),'load_ctrl_extended':load_extended,'valid':valid}
