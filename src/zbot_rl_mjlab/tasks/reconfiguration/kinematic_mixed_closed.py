"""Independent connected-ring FSM: 180^5 -> [270,270,180,90,90].

Geometry only. Every moving module root derives from joint FK on the one
connected chain; the head-tail brace stays at its original registration.
"""
import math
import mujoco
import numpy as np
from .kinematic_rz270 import KinematicRz270,KinematicPlanningError,rz,inverse,smooth

EXPERIMENT_ID='Zbot-Closed180-To-270-270-180-90-90-Kinematic-v1'
TARGET_DEG=(270,270,180,90,90)
DEFAULT_ORDER=(0,1,3,4)


class MixedClosedKinematics(KinematicRz270):
    def __init__(self,steps_per_phase=20,phase_duration=1.,order=DEFAULT_ORDER):
        if sorted(order)!=list(DEFAULT_ORDER):raise ValueError('Only internal connections 1,2,4,5 may be reconfigured')
        super().__init__(steps_per_phase,phase_duration)
        self.target_order=tuple(order)
        self.target_delta=np.deg2rad(np.array(TARGET_DEG)-180)
        m=self.model
        for i in (3,4):
            old=m.site(f'module_{i}/port_out').id
            new=m.site(f'module_{i}/port_out_rz270').id
            delta=self.target_delta[i];turn=rz(delta)[:3,:3]
            inlet=m.site_pos[m.site(f'module_{i+1}/port_in').id]
            R=np.empty(9);mujoco.mju_quat2Mat(R,m.site_quat[old]);R=R.reshape(3,3)
            m.site_pos[new]=m.site_pos[old]+R@(turn@inlet-inlet)
            q=np.empty(4);mujoco.mju_mulQuat(q,m.site_quat[old],np.array([math.cos(delta/2),0.,0.,math.sin(delta/2)]))
            m.site_quat[new]=q

    def internal_angles(self):return (180+np.rad2deg(self.registration[:5])).tolist()

    def record(self,phase,connector,p=0.,a=0.,event=False):
        # Base graph guard rejects detached fragments and double registrations.
        super().record(phase,connector,p,a,event=False)
        frame=self.frames[-1]
        if not frame.eq_active[self.old_eq[5]] or frame.eq_active[self.new_eq[5]]:
            self.frames.pop();raise KinematicPlanningError('Original head-tail brace must stay closed')
        if not frame.eq_active[self.old_eq[2]] or frame.eq_active[self.new_eq[2]]:
            self.frames.pop();raise KinematicPlanningError('Connection 3 must remain at 180 degrees')
        if event:
            info={'time_s':self.time,'phase':phase,'connection':None if connector<0 else connector+1,
                  'internal_angles_deg':self.internal_angles(),'joint_angle_rad':self.q.tolist(),
                  'position_error_m':p,'angle_error_deg':math.degrees(a),'connectivity':self.connectivity()}
            self.events.append(info);print('MIXED_FSM',info,flush=True)

    def plan(self):
        self.record('INITIAL_CLOSED',-1,event=True)
        for cut in self.target_order:
            anchor=self.roots[(cut+1)%6].copy();angle_start=self.registration[cut];angle_goal=self.target_delta[cut]
            self.active[self.old_eq[cut]]=False
            self.record('DISCONNECT',cut,event=True)
            approach_radius=.05
            for phase in ('SEPARATE','ROTATE','APPROACH'):
                if phase=='APPROACH':
                    _,source,destination=self.fk(self.q,cut,anchor)
                    approach_radius=float(np.linalg.norm((inverse(destination)@source)[:3,3]))
                maximum=np.zeros(2)
                for step in range(1,self.steps+1):
                    u=smooth(step/self.steps)
                    angle=angle_start if phase=='SEPARATE' else (angle_start+(angle_goal-angle_start)*u if phase=='ROTATE' else angle_goal)
                    radius=.05*u if phase=='SEPARATE' else (.05 if phase=='ROTATE' else approach_radius*(1-u))
                    if phase=='APPROACH' and step==self.steps:
                        p,a=self.solve(cut,anchor,rz(-angle)[:3,:3],position=np.zeros(3))
                    else:p,a=self.solve(cut,anchor,rz(-angle)[:3,:3],gap=radius)
                    self.time+=self.duration/self.steps
                    self.record(phase,cut,p,a);maximum=np.maximum(maximum,[p,a])
                self.stage_errors.append({'connection':cut+1,'phase':phase,'duration_s':self.duration,
                                          'max_position_error_m':maximum[0],'max_angle_error_rad':maximum[1]})
                self.record(phase+'_DONE',cut,p,a,event=True)
            if p>1e-6 or a>1e-5:raise KinematicPlanningError(f'Connection {cut+1}: strict closure tolerance not met')
            self.registration[cut]=angle_goal;self.active[self.new_eq[cut]]=True
            self.record('REBUILD_WELD',cut,p,a,event=True)
        self.record('DONE',-1,event=True)
        return self.frames

    def target_checks(self):
        constraints=self.active_constraint_errors()
        expected=np.array([True,True,False,True,True])
        return {**constraints,'internal_angles_deg':self.internal_angles(),
                'target_angles_match':bool(np.allclose(self.internal_angles(),TARGET_DEG,atol=1e-9)),
                'variant_selection_matches':bool(np.array_equal(self.active[self.new_eq[:5]].astype(bool),expected)),
                'head_tail_original_closed':bool(self.active[self.old_eq[5]] and not self.active[self.new_eq[5]]),
                'middle_connection_original':bool(self.active[self.old_eq[2]] and not self.active[self.new_eq[2]]),
                'connected_all_frames':all(self.connectivity(f.eq_active)['components']==1 for f in self.frames),
                'min_connected_faces':min(self.connectivity(f.eq_active)['connected_faces'] for f in self.frames),
                'final_connected_faces':self.connectivity()['connected_faces']}
