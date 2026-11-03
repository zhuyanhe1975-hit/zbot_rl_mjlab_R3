"""Single-robot, measured-state constraint Jacobian loading; no learning."""
import copy
import math
import numpy as np
import mujoco
from scipy.optimize import minimize


def response(model,data,eq_ids):
    mujoco.mj_forward(model,data)
    J=data.efc_J.reshape(data.nefc,model.nv)
    B=np.zeros((model.nv,6))
    for i in range(6):
        j=model.joint(f'module_{i}/joint{i+1}').id
        B[model.jnt_dofadr[j],i]=1.
    M=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,data,M)
    active=(data.efc_state==mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)|(data.efc_type==0)
    Ja=J[active]
    H=Ja@np.linalg.solve(M,Ja.T)+np.diag(data.efc_R[active])
    sensitivity=np.zeros((data.nefc,6));sensitivity[active]=-np.linalg.solve(H,Ja@np.linalg.solve(M,B))
    A=[];w=[]
    for i,eq in enumerate(eq_ids):
        rows=np.where((data.efc_type==0)&(data.efc_id==eq))[0]
        if len(rows)!=6:raise ValueError('Loading requires six connected faces')
        j=model.joint(f'module_{i}/module_root').id;adr=model.jnt_dofadr[j];body=model.jnt_bodyid[j]
        site=model.site(f'module_{i}/port_out').id
        rootR=data.xmat[body].reshape(3,3);R=data.site_xmat[site].reshape(3,3)
        arm=data.site_xpos[site]-data.xpos[body]
        def convert(g):return np.r_[R.T@g[:3],R.T@(rootR@g[3:]-np.cross(arm,g[:3]))]
        local=J[rows,adr:adr+6]
        w.append(convert(local.T@data.efc_force[rows]))
        generalized=local.T@sensitivity[rows]
        A.append(np.stack([convert(generalized[:,k]) for k in range(6)],-1))
    return np.array(A),np.array(w)


def peel(w, radius=.05):return np.maximum(w[:,2],0.)+np.linalg.norm(w[:,3:5],axis=-1)/radius


class LiveLoadingController:
    def __init__(self,env, *, other_limit=60., increment=.35, load_step=6., safety_weight=0., search_neighbors=False, safe_fallback=False):
        self.other_limit=other_limit;self.increment=increment;self.load_step=load_step
        self.safety_weight=safety_weight;self.search_neighbors=search_neighbors;self.safe_fallback=safe_fallback
        self.env=env;self.model=copy.copy(env.sim.mj_model);self.data=mujoco.MjData(self.model)
        self.model.actuator_gainprm[:,0]=1.
        self.model.actuator_biasprm[:]=0.;self.model.actuator_biasprm[:,2]=-env.cfg.joint_damping
        self.torque=np.zeros(6);self.last_report={};self.direction=None

    def reset(self):self.torque[:]=0.;self.direction=None;self.last_report={}

    def update(self,s):
        e=self.env;d=self.data
        d.qpos[:]=e.sim.data.qpos[0].cpu().numpy();d.qvel[:]=e.sim.data.qvel[0].cpu().numpy()
        d.eq_active[:]=e.sim.data.eq_active[0].cpu().numpy()
        # Rebuilt welds have completed a per-world stiffness ramp. Match its
        # current compliance when predicting the next face's loading response.
        solref=e.sim.model.eq_solref
        self.model.eq_solref[:]=(solref[0] if solref.ndim==3 else solref).cpu().numpy()
        d.ctrl[:]=self.torque
        ids=s.constraint_ids()[0].cpu().numpy()
        A,predicted=response(self.model,d,ids)
        measured=np.c_[s.force[0].cpu().numpy(),s.moment[0].cpu().numpy()]
        # At reset no wrench sample exists yet; use the CPU forward baseline.
        w=predicted if s.last_substep is None else measured
        target=e.cfg.target_connector;others=[i for i in range(6) if i!=target]
        radius=e.cfg.connector_radius
        current=peel(w,radius);goal=min(84.,max(12.,current[target]+self.load_step))
        candidates=[]
        axes=[self.direction] if self.direction is not None else [np.array([math.cos(a),math.sin(a)]) for a in np.linspace(0,2*math.pi,16,endpoint=False)]
        if self.search_neighbors and self.direction is not None:
            angle=math.atan2(self.direction[1],self.direction[0])
            axes=[np.array([math.cos(angle+a),math.sin(angle+a)]) for a in (0.,-math.pi/8,math.pi/8)]
        # Local constrained increments, corrected from real measured load every step.
        for axis in axes:
            for tension in (0.,1.):
                row=tension*A[target,2]+axis@A[target,3:5]/radius
                value=tension*max(w[target,2],0.)+axis@w[target,3:5]/radius
                def objective(delta):
                    predicted_others=peel(w+A@delta,radius)[others]
                    return delta@delta+.002*(self.torque+delta)@(self.torque+delta)+self.safety_weight*(predicted_others@predicted_others)
                def constraints(delta):
                    estimate=w+A@delta
                    return np.r_[self.other_limit-peel(estimate,radius)[others],value+row@delta-goal]
                bounds=[(max(-self.increment,-14.-t),min(self.increment,14.-t)) for t in self.torque]
                result=minimize(objective,np.zeros(6),method='SLSQP',bounds=bounds,
                    constraints=[{'type':'ineq','fun':constraints}],options={'ftol':1e-8,'maxiter':60})
                if result.success and constraints(result.x).min()>-1e-4:
                    candidates.append((objective(result.x),result.x,axis))
        if candidates:
            _,delta,self.direction=min(candidates,key=lambda x:x[0]);self.torque+=delta
        else:
            delta=np.zeros(6)
            self.direction=None
            if self.safe_fallback:
                # Never keep increasing load when no safe target increment is
                # available. Compare holding with bounded torque retreat using
                # the same local Jacobian prediction; neither is a guarantee.
                retreat=np.clip(-self.torque,-self.increment,self.increment)
                if peel(w+A@retreat,radius)[others].max() < current[others].max():
                    delta=retreat;self.torque+=delta
        self.last_report={'target':target,'loads_N':current.tolist(),'goal_N':float(goal),
                          'torque_Nm':self.torque.tolist(),'feasible_increment':bool(candidates),
                          'predicted_other_limit_N':self.other_limit,'predicted_loads_N':peel(w+A@delta,radius).tolist()}
        return self.torque
