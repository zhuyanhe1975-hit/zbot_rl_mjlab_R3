"""Model-scoped GPU contact filtering for welded/open interface pairs.

The installed Warp MULTICCD implementation does not support margins as a
per-world collision switch. Filter only the CONSTRAINT bit after narrowphase,
before constraint construction; all ordinary collisions remain unchanged.
"""
from functools import wraps
import numpy as np
import warp as wp

_MARKER='_zbot_mixed_weld_contact_filter'


def prune_initial_closure_pairs(env):
    """Skip narrowphase for faces whose constraints this task always discards.

    Initial closure ends at the first LOAD; none of its reachable phases enables
    phase_contacts. Keep the model pairs for lookup/diagnostics, but omit them
    from Warp's NXN candidates. Other collisions and sensor pairs stay intact.
    """
    if not (getattr(env.cfg,'full_stop_after_initial_closure',False)
            and getattr(env.cfg,'full_initial_closure_collision_pruning',False)):
        return
    from mujoco_warp._src.types import BroadphaseType
    model=env.sim.wp_model
    if model.opt.broadphase!=BroadphaseType.NXN:
        return
    pairs=model.nxn_geom_pair_filtered.numpy()
    ids=model.nxn_pairid_filtered.numpy()
    # Head-tail penetration monitoring needs raw narrowphase distances even
    # when the phase filter removes this pair's physical solver constraint.
    count=5 if getattr(env.cfg,'full_headtail_penetration_guard',False) else 6
    faces={int(env.sim.mj_model.pair(f'open_face_{i}').id) for i in range(count)}
    keep=np.array([int(contact) not in faces or sensor>=0 for contact,sensor in ids])
    model.nxn_geom_pair_filtered=wp.array(pairs[keep],dtype=wp.vec2i,device=env.sim.wp_device)
    model.nxn_pairid_filtered=wp.array(ids[keep],dtype=wp.vec2i,device=env.sim.wp_device)


@wp.kernel
def _filter_closed_faces(nacon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],
                         kind:wp.array[int],active:wp.array2d[wp.bool],lookup:wp.array2d[int],
                         original:wp.array[int],variant:wp.array[int]):
    i=wp.tid()
    if i>=nacon[0]:return
    w=world[i];g=geom[i]
    if w<0 or g[0]<0 or g[1]<0:return
    face=lookup[g[0],g[1]]
    if face>=0:
        if active[w,original[face]] or active[w,variant[face]]:
            # Preserve SENSOR (2), remove CONSTRAINT (1).
            kind[i]=kind[i]&2



@wp.kernel
def _filter_phase_faces(nacon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],
                        kind:wp.array[int],active:wp.array2d[wp.bool],lookup:wp.array2d[int],
                        original:wp.array[int],variant:wp.array[int],enabled:wp.array2d[wp.bool]):
    i=wp.tid()
    if i>=nacon[0]:return
    w=world[i];g=geom[i]
    if w<0 or g[0]<0 or g[1]<0:return
    face=lookup[g[0],g[1]]
    if face>=0:
        if not enabled[w,face] or active[w,original[face]] or active[w,variant[face]]:
            kind[i]=kind[i]&2

def install_model_contact_filter(env,phase_enabled=None,include_headtail=False):
    from mujoco_warp._src import collision_driver
    if not getattr(collision_driver.collision,'_zbot_mixed_filter',False):
        original=collision_driver.collision
        @wraps(original)
        def collision(model,data,*args,**kwargs):
            result=original(model,data,*args,**kwargs)
            params=getattr(model.opt,_MARKER,None)
            if params is not None:
                lookup,old,new=params
                inputs=[data.nacon,data.contact.worldid,data.contact.geom,data.contact.type,
                        data.eq_active,lookup,old,new]
                phase=getattr(model.opt,'_zbot_mixed_phase_contacts',None)
                kernel=_filter_closed_faces if phase is None else _filter_phase_faces
                if phase is not None:inputs.append(phase)
                wp.launch(kernel,dim=data.contact.dist.shape[0],inputs=inputs,device=data.contact.dist.device)
            return result
        collision._zbot_mixed_filter=True
        collision_driver.collision=collision
    m=env.sim.mj_model
    lookup=np.full((m.ngeom,m.ngeom),-1,dtype=np.int32)
    count=6 if include_headtail else 5
    for i in range(count):
        pair=m.pair(f'open_face_{i}').id;a=int(m.pair_geom1[pair]);b=int(m.pair_geom2[pair])
        lookup[a,b]=i;lookup[b,a]=i
    old=np.array([m.equality(f'connector_{i}').id for i in range(count)],dtype=np.int32)
    new=np.array([m.equality(f'connector_{i}_rz270').id for i in range(5)]+([int(old[5])] if include_headtail else []),dtype=np.int32)
    params=tuple(wp.array(x,dtype=wp.int32,device=env.sim.wp_device) for x in (lookup,old,new))
    setattr(env.sim.wp_model.opt,_MARKER,params)
    if phase_enabled is not None:
        setattr(env.sim.wp_model.opt,'_zbot_mixed_phase_contacts',wp.from_torch(phase_enabled,dtype=wp.bool))
        prune_initial_closure_pairs(env)
    # Simulation captures its CUDA step/forward graphs before action terms
    # install this model-specific filter. Rebuild them after installing it;
    # otherwise graph replay bypasses the filter entirely.
    env.sim.create_graph()
