"""One-shot fixed-wrench loading optimization using the compiled HiGHS MILP solver."""
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_static_loading(A, w, target, *, break_load=100., other_limit=100.,
                         torque_limit=14., radius=.05, directions=32):
    """Minimize worst predicted non-target peel, then lightly penalize torque.

    The target's nonconvex superlevel set is a union of linear support
    halfspaces. Binary variables choose one; HiGHS solves the complete MILP.
    Other-face moment norms use a conservative circumscribed polygon.
    """
    A=np.asarray(A,dtype=float);w=np.asarray(w,dtype=float)
    if A.shape!=(6,6,6) or w.shape!=(6,6) or not np.isfinite(A).all() or not np.isfinite(w).all():
        raise ValueError('expected finite wrench response (6,6,6) and wrench (6,6)')
    if target not in range(6) or min(break_load,other_limit,torque_limit,radius)<=0 or directions<4:
        raise ValueError('invalid loading problem parameters')
    angles=2*np.pi*np.arange(directions)/directions
    axes=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    count=2*directions; n=13+count; z0=13
    objective=np.zeros(n);objective[6]=1.;objective[7:13]=1e-4
    lb=np.r_[np.full(6,-torque_limit),0.,np.zeros(6+count)]
    ub=np.r_[np.full(6,torque_limit),other_limit,np.full(6,torque_limit),np.ones(count)]
    integrality=np.r_[np.zeros(13,dtype=int),np.ones(count,dtype=int)]
    rows=[];lower=[];upper=[]
    def add(row,lo=-np.inf,hi=np.inf):rows.append(row);lower.append(lo);upper.append(hi)
    # |tau| auxiliaries allow a secondary, scale-small effort preference.
    for j in range(6):
        for sign in (-1.,1.):
            row=np.zeros(n);row[j]=sign;row[7+j]=-1.;add(row,hi=0.)
    sec=1./math.cos(math.pi/directions)
    for i in range(6):
        if i==target:continue
        for axis in axes:
            for tension in (0.,1.):
                row=np.zeros(n);row[:6]=tension*A[i,2]+sec*(axis@A[i,3:5])/radius;row[6]=-1.
                value=tension*w[i,2]+sec*(axis@w[i,3:5])/radius
                add(row,hi=-value)
    # Big-M values are calculated from bounds, not arbitrary constants.
    for k,(axis,tension) in enumerate((a,t) for a in axes for t in (0.,1.)):
        support=tension*A[target,2]+(axis@A[target,3:5])/radius
        value=tension*w[target,2]+(axis@w[target,3:5])/radius
        big=max(0.,break_load-value+torque_limit*np.abs(support).sum())+1e-6
        row=np.zeros(n);row[:6]=-support;row[z0+k]=big
        add(row,hi=big+value-break_load)
    row=np.zeros(n);row[z0:]=1.;add(row,lo=1.,hi=1.)
    result=milp(objective,integrality=integrality,bounds=Bounds(lb,ub),
                constraints=LinearConstraint(np.asarray(rows),np.asarray(lower),np.asarray(upper)),
                options={'time_limit':10.,'mip_rel_gap':1e-5})
    if result.x is None or result.status!=0:
        return {'success':False,'status':int(result.status),'message':str(result.message)}
    tau=result.x[:6];predicted=w+np.einsum('fij,j->fi',A,tau)
    loads=np.maximum(predicted[:,2],0.)+np.linalg.norm(predicted[:,3:5],axis=-1)/radius
    other=np.arange(6)!=target
    valid=loads[target]>=break_load-1e-5 and np.max(loads[other])<=other_limit+1e-5 and np.max(np.abs(tau))<=torque_limit+1e-6
    return {'success':bool(valid),'status':int(result.status),'message':str(result.message),
            'torque':tau,'predicted_loads':loads,'other_peak':float(np.max(loads[other])),
            'solver':'HiGHS MILP','directions':directions}
