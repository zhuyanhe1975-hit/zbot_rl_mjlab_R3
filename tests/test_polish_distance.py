import os
import subprocess
import sys

import mujoco
import numpy as np

from scripts.optimize_headtail_mjbatch import STATE, lib_open, ready


def test_native_prediction_nearest_distance_matches_direct_geometry(tmp_path):
    library=tmp_path/'strict.so'
    subprocess.run([sys.executable,'scripts/build_mjbatch_rollout.py','--source',
                    'scripts/mjbatch_curl_rollout_strict.cpp','--output',str(library)],check=True)
    f=ready('forward',False,kp=15.,kd=1.5)
    try:
        m=f.model
        state=np.empty(mujoco.mj_stateSize(m,STATE));mujoco.mj_getState(m,f.data,state,STATE)
        path=tmp_path/'model.mjb';mujoco.mj_saveModel(m,str(path),None)
        steps=10;controls=np.ascontiguousarray(np.repeat(f.data.qpos[f.qadr][None,None,:],steps,axis=1))
        lengths=np.array([steps],dtype=np.int32);result=np.zeros((1,13));ends=np.zeros((1,len(state)))
        metrics=np.zeros((1,4));distances=np.zeros(1);lib=lib_open(library)
        ptr=lib.zbatch_create(os.fsencode(path),2)
        assert ptr
        try:lib.zbatch_eval_distance(ptr,state,1,steps,lengths,controls,result,ends,80.,metrics,0,distances)
        finally:lib.zbatch_free(ptr)
        assert result[0,0]==0
        data=mujoco.MjData(m);mujoco.mj_setState(m,data,state,STATE);mujoco.mj_forward(m,data)
        closest=.05
        for ctrl in controls[0]:
            data.ctrl[f.actuators]=ctrl;mujoco.mj_step(m,data);mujoco.mj_kinematics(m,data)
            for i in range(1,m.ngeom):
                for j in range(i+1,m.ngeom):
                    if f.geom_module[i]<0 or f.geom_module[j]<0 or f.geom_module[i]==f.geom_module[j]:
                        continue
                    face=next((k for k,pair in enumerate(f.pair_geoms) if {i,j}==set(map(int,pair))),None)
                    if face is not None and data.eq_active[f.old[face]]:
                        continue
                    closest=min(closest,float(mujoco.mj_geomDistance(m,data,i,j,.05,None)))
        assert abs(closest-distances[0]) < 1e-9
    finally:
        f.close()
