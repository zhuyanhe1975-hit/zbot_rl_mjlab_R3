"""Full reconstruction GPU scene initialization (not yet a training task)."""
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from mjlab.managers.event_manager import EventTermCfg
from .batched_state import add_batched_connectors
from .disconnect.state import state

INITIAL=Path(__file__).resolve().parents[4]/'assets/reconfiguration/full_straight_start.json'

def add_full_batched_connectors(spec):
    add_batched_connectors(spec)
    spec.add_exclude(bodyname1='module_5/foot_1',bodyname2='module_0/foot_0')
    spec.add_pair(name='open_face_5',geomname1='module_5/b6_geom',geomname2='module_0/a1_geom',
                  condim=3,margin=0.,gap=0.,friction=(.6,.6,.1,.1,.1))

def add_full_batched_connectors_headtail_90(spec):
    """Use the same registered head-tail site as the refined CPU teacher."""
    add_full_batched_connectors(spec)
    source=spec.site('module_5/port_out');inlet=spec.site('module_0/port_in')
    delta=-math.pi/2.;c=math.cos(delta);s=math.sin(delta)
    turn=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,source.quat)
    position=source.pos+rotation.reshape(3,3)@(turn@inlet.pos-inlet.pos)
    quat=np.empty(4);mujoco.mju_mulQuat(quat,source.quat,np.array([math.cos(delta/2),0.,0.,math.sin(delta/2)]))
    source.parent.add_site(name='module_5/port_out_refined90',pos=position,quat=quat,size=(.004,)*3)
    spec.equality('connector_5').name1='module_5/port_out_refined90'

def add_online_headtail_connectors(spec):
    """Add GPU closest-point sensors for the four actual head-tail mesh pairs."""
    add_full_batched_connectors_headtail_90(spec)
    for i,(a,b) in enumerate((('a','a'),('a','b'),('b','a'),('b','b'))):
        args=dict(objtype=mujoco.mjtObj.mjOBJ_GEOM,objname=f'module_0/{a}1_geom',
                  reftype=mujoco.mjtObj.mjOBJ_GEOM,refname=f'module_5/{b}6_geom',cutoff=.15)
        spec.add_sensor(name=f'headtail_distance_{i}',type=mujoco.mjtSensor.mjSENS_GEOMDIST,**args)
        spec.add_sensor(name=f'headtail_fromto_{i}',type=mujoco.mjtSensor.mjSENS_GEOMFROMTO,**args)

def reset_headtail_open(env,env_ids):
    s=state(env)
    env.sim.data.eq_active[env_ids,s.eq_ids[5]]=False

def configure_straight_initial_scene(cfg):
    data=json.loads(INITIAL.read_text())
    if data.get('version')!=1 or len(data['roots'])!=6:raise ValueError('invalid full reconstruction initial pose')
    for i,root in enumerate(data['roots']):
        init=cfg.scene.entities[f'module_{i}'].init_state
        init.pos=tuple(root[:3]);init.rot=tuple(root[3:]);init.lin_vel=(0.,0.,0.);init.ang_vel=(0.,0.,0.)
        init.joint_pos={f'joint{i+1}':float(data['joint_angles'][i])};init.joint_vel={f'joint{i+1}':0.}
    cfg.scene.spec_fn=add_full_batched_connectors
    cfg.events['reset_headtail_open']=EventTermCfg(func=reset_headtail_open,mode='reset')
    return cfg
