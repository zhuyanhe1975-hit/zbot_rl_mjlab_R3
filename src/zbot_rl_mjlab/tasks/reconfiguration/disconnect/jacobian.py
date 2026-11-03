"""Reference constraint-Jacobian force response for the fixed IK closed loop.

This is a local forward-dynamics linearization, NOT an exact global static
inverse: contact active sets and compliance can change. Runtime load correction
and unchanged measured safety limits protect its use as a residual-PPO baseline.
"""
import numpy as np
import mujoco
from scipy.optimize import minimize


def constraint_response(model):
    if model.opt.jacobian!=mujoco.mjtJacobian.mjJAC_DENSE:
        raise ValueError('Jacobian feedforward requires dense Jacobian')
    if model.opt.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError('Reference feedforward currently requires pyramidal contacts')
    data=mujoco.MjData(model);data.qpos[:]=model.qpos0;data.qvel[:]=0.;data.ctrl[:]=0.
    mujoco.mj_forward(model,data)
    J=data.efc_J.reshape(data.nefc,model.nv).copy()
    B=np.zeros((model.nv,6));joint_positions=[]
    for i in range(6):
        joint=model.joint(f'module_{i}/joint{i+1}').id
        matches=np.where(model.actuator_trnid[:,0]==joint)[0]
        if len(matches)!=1 or model.actuator_gear[matches[0],0]!=1.:
            raise ValueError('Expected six unit-gear motor joints')
        B[model.jnt_dofadr[joint],i]=1.
        joint_positions.append(data.qpos[model.jnt_qposadr[joint]])
    M=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,data,M)
    active=(data.efc_state==mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC) | (data.efc_type==0)
    Ja=J[active];R=data.efc_R[active]
    # Fixed active-set soft-constraint system:
    # (J M^-1 J^T + R) d(lambda) = -J M^-1 B d(tau).
    H=Ja@np.linalg.solve(M,Ja.T)+np.diag(R)
    sensitivity=np.zeros((data.nefc,6))
    sensitivity[active]=-np.linalg.solve(H,Ja@np.linalg.solve(M,B))
    A=[];baseline=[]
    for i in range(6):
        eq=model.equality(f'connector_{i}').id
        rows=np.where((data.efc_type==0)&(data.efc_id==eq))[0]
        if len(rows)!=6:raise ValueError('Reference model needs six active weld rows per connector')
        root_joint=model.joint(f'module_{i}/module_root').id
        adr=model.jnt_dofadr[root_joint];root=model.jnt_bodyid[root_joint]
        port=model.site(f'module_{i}/port_out').id
        rotation=data.xmat[root].reshape(3,3);frame=data.site_xmat[port].reshape(3,3)
        arm=data.site_xpos[port]-data.xpos[root]
        def convert(g):
            return np.r_[frame.T@g[:3],frame.T@(rotation@g[3:]-np.cross(arm,g[:3]))]
        local_J=J[rows,adr:adr+6]
        baseline.append(convert(local_J.T@data.efc_force[rows]))
        generalized=local_J.T@sensitivity[rows]
        A.append(np.stack([convert(generalized[:,k]) for k in range(6)],-1))
    return np.array(A),np.array(baseline),np.array(joint_positions)


def solve_reference(model,cfg):
    A,w0,qref=constraint_response(model)
    axis=np.asarray(cfg.jacobian_axis_xy,dtype=float)
    if axis.shape!=(2,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('jacobian_axis_xy must be a finite nonzero 2-vector')
    axis/=np.linalg.norm(axis)
    i=cfg.target_connector;others=[k for k in range(6) if k!=i]
    row=axis@A[i,3:5]
    goal=cfg.jacobian_seed_tilt_force*cfg.connector_radius-axis@w0[i,3:5]
    if row@row<1e-12:raise ValueError('Requested bending axis is not controllable in the reference Jacobian')
    def loads(w):return np.maximum(w[:,2],0.)+np.linalg.norm(w[:,3:5],axis=-1)/cfg.connector_radius
    def objective(tau):return tau@tau+.01*np.sum(loads(w0+A@tau)[others]**2)/cfg.break_force**2
    def constraints(tau):return np.r_[cfg.jacobian_other_load_limit-loads(w0+A@tau)[others],row@tau-goal]
    x0=row*goal/(row@row)
    limit=cfg.continuous_torque_limit
    result=minimize(objective,x0,method='SLSQP',bounds=[(-limit,limit)]*6,
                    constraints=[{'type':'ineq','fun':constraints}],options={'ftol':1e-10,'maxiter':200})
    if not result.success or constraints(result.x).min()<-1e-5:
        raise ValueError('No feasible reference Jacobian feedforward: '+str(result.message))
    report={'method':'constraint_J_M_R_active_set','torque_Nm':result.x.tolist(),
            'axis_xy':axis.tolist(),'seed_tilt_force_N':cfg.jacobian_seed_tilt_force,
            'predicted_loads_N':loads(w0+A@result.x).tolist(),
            'reference_joint_positions':qref.tolist()}
    return result.x,qref,report


def reference_wrenches(model,data):
    """Independent CPU readout used for finite-difference verification."""
    J=data.efc_J.reshape(data.nefc,model.nv)
    out=[]
    for i in range(6):
        eq=model.equality(f'connector_{i}').id
        rows=np.where((data.efc_type==0)&(data.efc_id==eq))[0]
        joint=model.joint(f'module_{i}/module_root').id
        adr=model.jnt_dofadr[joint];body=model.jnt_bodyid[joint]
        port=model.site(f'module_{i}/port_out').id
        g=J[rows,adr:adr+6].T@data.efc_force[rows]
        rootR=data.xmat[body].reshape(3,3);portR=data.site_xmat[port].reshape(3,3)
        out.append(np.r_[portR.T@g[:3],portR.T@(rootR@g[3:]-np.cross(data.site_xpos[port]-data.xpos[body],g[:3]))])
    return np.array(out)
