"""Construct the six-module closure model without importing MJLab."""
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'assets/zbot_6/ZBOT6.xml'
SNAPSHOT=ROOT/'assets/reconfiguration/closed_loop_start.json'
START=ROOT/'assets/reconfiguration/full_straight_start.json'
ROOT_NAMES=('foot_0','a2','a3','base','a5','a6')
TIP_NAMES=('b1','b2','b3','b4','b5','foot_1')

def _numbers(v):return ' '.join(str(float(x)) for x in v)

def _module_spec(index,snapshot):
    xml=ET.parse(ASSET).getroot();option=xml.find('option')
    if option is not None:xml.remove(option)
    xml.find('compiler').set('meshdir',str(ASSET.parent/'meshes'))
    world=xml.find('worldbody');original=next(b for b in world.iter('body') if b.get('name')==ROOT_NAMES[index])
    root=copy.deepcopy(original);tip=next(b for b in root.findall('body') if b.get('name')==TIP_NAMES[index])
    for child in list(tip.findall('body')):tip.remove(child)
    for child in list(root.findall('freejoint')):root.remove(child)
    saved=snapshot['modules'][index];root.set('pos',_numbers(saved['root_state'][:3]));root.set('quat',_numbers(saved['root_state'][3:7]))
    ET.SubElement(root,'freejoint',name='module_root')
    reference=saved['joint_position'];quat=np.empty(4)
    mujoco.mju_mulQuat(quat,np.fromstring(tip.get('quat'),sep=' '),np.array([math.cos(reference/2),0,0,math.sin(reference/2)]))
    tip.set('quat',_numbers(quat))
    for j in tip.findall('joint'):
        j.set('range',f'{-math.pi} {math.pi}');j.set('ref',str(reference));j.set('limited','false')
    for geom in root.iter('geom'):geom.set('condim','3');geom.set('friction','.6 .1 .1')
    inlet=np.array([-.00022282,-.00010463,0.]);ET.SubElement(root,'site',name='port_in',pos=_numbers(inlet),size='.004')
    if index<5:
        relative=original.find('body').find('body');pos=np.fromstring(relative.get('pos'),sep=' ');q=np.fromstring(relative.get('quat'),sep=' ')
        R=np.empty(9);mujoco.mju_quat2Mat(R,q);pos+=R.reshape(3,3)@inlet
    else:
        sole=tip.find("site[@name='foot_1_sole']");pos=np.fromstring(sole.get('pos'),sep=' ');q=np.fromstring(sole.get('quat'),sep=' ')
        angle=snapshot['tail_registration_quarter']*math.pi/2;out=np.empty(4)
        mujoco.mju_mulQuat(out,q,np.array([math.cos(angle/2),0,0,math.sin(angle/2)]));q=out
    ET.SubElement(tip,'site',name='port_out',pos=_numbers(pos),quat=_numbers(q),size='.004')
    world.clear();world.append(root)
    spec=mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'))
    spec.add_actuator(name=f'joint{index+1}',target=f'joint{index+1}',
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,gainprm=(15.,0,0,0,0,0,0,0,0,0),
                      biasprm=(0,-15.,-1.5,0,0,0,0,0,0,0),forcelimited=True,forcerange=(-20.,20.))
    return spec

def make_model(timestep=.001):
    snapshot=json.loads(SNAPSHOT.read_text());spec=mujoco.MjSpec()
    spec.option.timestep=timestep;spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver=mujoco.mjtSolver.mjSOL_NEWTON;spec.option.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    spec.option.iterations=50;spec.option.ls_iterations=20;spec.option.ccd_iterations=50
    spec.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(10,10,.1),friction=(.6,.005,.0001))
    for i in range(6):spec.attach(_module_spec(i,snapshot),prefix=f'module_{i}/',frame=spec.worldbody.add_frame())
    for i in range(5):spec.add_exclude(bodyname1=f'module_{i}/{TIP_NAMES[i]}',bodyname2=f'module_{i+1}/{ROOT_NAMES[i+1]}')
    # Exclude the native parent-body pairing, then explicitly restore its raw contact pair.
    spec.add_exclude(bodyname1='module_5/foot_1',bodyname2='module_0/foot_0')
    for i in range(6):
        tail=spec.site(f'module_{i}/port_out');inlet=spec.site(f'module_{(i+1)%6}/port_in')
        site=tail.name
        if i==5:
            angle=-math.pi/2;turn=np.array([[0,1,0],[-1,0,0],[0,0,1]])
            R=np.empty(9);mujoco.mju_quat2Mat(R,tail.quat);q=np.empty(4)
            mujoco.mju_mulQuat(q,tail.quat,np.array([math.cos(angle/2),0,0,math.sin(angle/2)]))
            pos=tail.pos+R.reshape(3,3)@(turn@inlet.pos-inlet.pos)
            site='module_5/search_port';tail.parent.add_site(name=site,pos=pos,quat=q,size=(.004,)*3)
        spec.add_equality(name=f'connector_{i}',type=mujoco.mjtEq.mjEQ_WELD,objtype=mujoco.mjtObj.mjOBJ_SITE,
                          name1=site,name2=inlet.name,active=(i<5),data=(*([0.]*10),1.),
                          solref=(.005,1.),solimp=(.999,.9999,.001,.5,2.))
    for label,site in (('head','module_0/port_in'),('tail','module_5/search_port')):
        for kind,stype in (('linear',mujoco.mjtSensor.mjSENS_FRAMELINVEL),('angular',mujoco.mjtSensor.mjSENS_FRAMEANGVEL)):
            spec.add_sensor(name=f'{label}_{kind}',type=stype,objtype=mujoco.mjtObj.mjOBJ_SITE,objname=site)
    model=spec.compile();data=mujoco.MjData(model);start=json.loads(START.read_text())
    for i in range(6):
        root=int(model.joint(f'module_{i}/module_root').qposadr[0]);joint=int(model.joint(f'module_{i}/joint{i+1}').qposadr[0])
        data.qpos[root:root+7]=start['roots'][i];data.qpos[joint]=start['joint_angles'][i]
    data.eq_active[:]=model.eq_active0;mujoco.mj_forward(model,data)
    return model,data
