"""Interactive 90-degree reconfiguration: 1 forward, 2 reverse, Enter reset."""
import argparse
import inspect
import json
import math
import queue
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

import scripts.view_headtail_search as visual
import zbot_rl_mjlab.tasks.reconfiguration.full_cpu as full
import zbot_rl_mjlab.tasks.reconfiguration.reference_trajectory as trajectory

START = Path('outputs/reconfiguration/headtail_reverse_2026_10_29/start.npz')


class ReverseReconfiguration(full.FullCpuReconfiguration):
    """Use the same physical controller, reconnecting the original sites."""
    def geometry(self,face=None,new_weld=False):
        i=self.target if face is None else face
        if not new_weld or i==5:
            return super().geometry(face,new_weld)
        saved=self.new
        self.new=saved.copy();self.new[i]=self.old[i]
        try:return super().geometry(face,new_weld)
        finally:self.new=saved

    def approach_feedback_update(self):
        if self.target==5:return super().approach_feedback_update()
        saved=self.new
        self.new=saved.copy();self.new[self.target]=self.old[self.target]
        try:return super().approach_feedback_update()
        finally:self.new=saved


# Reuse the verified state machine while changing only the reconnect equality
# and final registration expected by the reverse task.
advance_source=textwrap.dedent(inspect.getsource(full.FullCpuReconfiguration.advance))
advance_source=advance_source.replace('self.new[self.target]', 'self.old[self.target]')
advance_source=advance_source.replace('[True,True,False,True,True]', '[False,False,False,False,False]')
advance_ns=dict(full.__dict__)
exec(compile(advance_source,__file__,'exec'),advance_ns)
ReverseReconfiguration.advance=advance_ns['advance']

# The reverse planner aims at the original zero-offset registration.
plan_source=textwrap.dedent(inspect.getsource(trajectory.AxialPlanner.plan))
plan_source=plan_source.replace('Rgoal=rz(-k.target_delta[cut])[:3,:3]', 'Rgoal=np.eye(3)')
plan_ns=dict(trajectory.__dict__)
exec(compile(plan_source,__file__,'exec'),plan_ns)


def config():
    cfg=visual.make_config(SimpleNamespace(speed_deg_s=105.*1.3,accel_deg_s2=980.*1.3**2))
    cfg.curl_speed_multiplier*=1.3
    return cfg


def create(direction):
    cfg=config();full.ORDER=(0,1,3,4) if direction=='forward' else (4,3,1,0)
    advance_ns['ORDER']=full.ORDER
    f=(full.FullCpuReconfiguration if direction=='forward' else ReverseReconfiguration)(cfg)
    try:
        branch=json.loads((visual.out/'verification_90_3_1245.json').read_text())['branch_q']
        visual.prepare(f,90,branch)
        if direction=='reverse':
            with np.load(START) as snapshot:
                mujoco.mj_setState(f.model,f.data,snapshot['state'],mujoco.mjtState.mjSTATE_INTEGRATION)
                f.closed_q=snapshot['closed_q'].copy()
            f.data.time=0.;f.initial_qpos=f.data.qpos.copy();f.qref=f.data.qpos[f.qadr].copy()
            f.planner.plan=plan_ns['plan'].__get__(f.planner)
            f.set_contact_phase();mujoco.mj_forward(f.model,f.data)
            if not np.array_equal(f.data.eq_active[f.new],[True,True,False,True,True]) or f.connected()[5]:
                raise RuntimeError('reverse snapshot has incorrect starting topology')
        return f
    except BaseException:
        f.close();raise


def swap(previous,fresh):
    """Keep the existing viewer model and data objects alive across resets."""
    model,data=previous.model,previous.data
    try:
        for name in ('eq_solref','eq_obj1id','pair_geom1','pair_geom2',
                     'actuator_gainprm','actuator_biasprm','site_pos','site_quat'):
            getattr(model,name)[:]=getattr(fresh.model,name)
        mujoco.mj_copyData(data,model,fresh.data)
        fresh.model=model;fresh.data=data
    except BaseException:
        fresh.close();raise
    previous.close();return fresh


def forward_from_current(previous):
    """Start a forward run from the current physical state without resetting it."""
    if (not np.all(previous.connected()[:5]) or previous.connected()[5]
            or np.any(previous.data.eq_active[previous.new])
            or np.max(np.abs(previous.data.qpos[previous.qadr]))>=.02
            or np.max(np.abs(previous.data.qvel[previous.dofs]))>=.05):
        raise ValueError('Forward requires a stationary straight all-180 chain with head-tail open')
    fresh=create('forward')
    try:
        mujoco.mj_copyData(fresh.data,fresh.model,previous.data)
        fresh.phase_start=float(fresh.data.time)
        fresh.initial_qpos=fresh.data.qpos.copy()
        fresh.qref=fresh.data.qpos[fresh.qadr].copy()
        fresh.next_log=fresh.phase_start;fresh.next_progress=fresh.phase_start
        fresh.set_contact_phase();fresh.set_pd()
        mujoco.mj_forward(fresh.model,fresh.data)
        return fresh
    except BaseException:
        fresh.close();raise


def reverse_from_current(previous):
    """Start a reverse run without resetting the current physical state."""
    expected=np.array([True,True,False,True,True])
    if (not np.all(previous.connected()[:5]) or previous.connected()[5]
            or not np.array_equal(previous.data.eq_active[previous.new],expected)
            or np.max(np.abs(previous.data.qpos[previous.qadr]))>=.02
            or np.max(np.abs(previous.data.qvel[previous.dofs]))>=.05):
        raise ValueError('Reverse requires a stationary straight mixed chain with head-tail open')
    fresh=create('reverse')
    try:
        mujoco.mj_copyData(fresh.data,fresh.model,previous.data)
        fresh.phase_start=float(fresh.data.time)
        fresh.initial_qpos=fresh.data.qpos.copy()
        fresh.qref=fresh.data.qpos[fresh.qadr].copy()
        fresh.next_log=fresh.phase_start;fresh.next_progress=fresh.phase_start
        fresh.set_contact_phase();fresh.set_pd()
        mujoco.mj_forward(fresh.model,fresh.data)
        return fresh
    except BaseException:
        fresh.close();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--direction',choices=('forward','reverse'),default='forward')
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/headtail_reverse_2026_10_29/latest.json'))
    args=p.parse_args();direction=args.direction;f=create(direction)
    try:
        if args.headless:
            while f.phase not in ('DONE','FAILED') and f.data.time<120:f.step()
            result=f.report();result['direction']=direction;result['phase_active_loads']=f.phase_loads
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
            print('RESULT',direction,f.phase,'time',f.data.time,'failure',f.failure,flush=True)
            return 0 if f.phase=='DONE' else 1
        import mujoco.viewer
        keys=queue.SimpleQueue()
        def on_key(key):
            if key in (13,257,335,49,50,321,322):keys.put(key)
        with mujoco.viewer.launch_passive(f.model,f.data,key_callback=on_key) as viewer:
            viewer.cam.lookat[:]=[0.,0.,.08];viewer.cam.distance=1.4;viewer.cam.elevation=-55
            anchor=time.perf_counter();anchor_sim=float(f.data.time);next_status=0.;notice=''
            while viewer.is_running():
                if not keys.empty():
                    key=keys.get()
                    try:
                        if key in (49,321):
                            if direction=='forward' and f.phase not in ('DONE','FAILED'):
                                fresh=None;notice='Forward is already running'
                            else:
                                fresh=forward_from_current(f);direction='forward';notice=''
                        elif key in (50,322):
                            if direction=='reverse' and f.phase not in ('DONE','FAILED'):
                                fresh=None;notice='Reverse is already running'
                            else:
                                fresh=reverse_from_current(f);direction='reverse';notice=''
                        else:
                            fresh=create(direction);notice=''
                        if fresh is not None:
                            with viewer.lock():f=swap(f,fresh)
                            anchor=time.perf_counter();anchor_sim=float(f.data.time)
                    except ValueError as error:
                        notice=str(error)
                    next_status=0.
                if f.phase not in ('DONE','FAILED'):
                    desired=anchor_sim+time.perf_counter()-anchor;deadline=time.perf_counter()+.012
                    while f.data.time<desired and time.perf_counter()<deadline and f.phase not in ('DONE','FAILED'):f.step()
                if time.perf_counter()>=next_status:
                    peak=max((x['load_N'] for x in f.phase_loads.values()),default=0.)
                    labels='Direction\nTime\nPhase\nTarget face\nFace loads 1-6 (N)\nPeak other (N)\nMin margin (N)\nConnected 1-6\nControls'
                    values=(f'{direction}\n{f.data.time:.2f}s\n{f.phase}\n{f.target+1}\n'
                            +' '.join(f'{x:.1f}' for x in f.loads)+f'\n{peak:.2f}\n{100-peak:.2f}\n'
                            +' '.join('1' if x else '0' for x in f.connected())+'\n1 forward / 2 reverse / Enter reset')
                    if f.failure:labels+='\nFailure';values+='\n'+f.failure[:160]
                    if notice:labels+='\nNotice';values+='\n'+notice
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,labels,values));next_status=time.perf_counter()+.1
                viewer.sync();time.sleep(.002)
        return 0
    finally:f.close()


if __name__=='__main__':raise SystemExit(main())
