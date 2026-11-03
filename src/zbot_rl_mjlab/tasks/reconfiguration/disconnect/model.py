"""Six independent motorized modules joined by six passive breakable welds."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ....robot import ASSET_PATH

ROOT_NAMES = ('foot_0','a2','a3','base','a5','a6')
TIP_NAMES = ('b1','b2','b3','b4','b5','foot_1')
SNAPSHOT_PATH = Path(__file__).resolve().parents[5]/'assets/reconfiguration/closed_loop_start.json'
# A small weld length scale made active connectors behave like loose ball
# joints: centimetre-scale flanges could rotate tens of degrees under gravity.
TORQUE_SCALE = 1.


def load_snapshot(path=SNAPSHOT_PATH):
    snapshot = json.loads(Path(path).read_text())
    if snapshot.get('version') != 1 or len(snapshot.get('modules',[])) != 6:
        raise ValueError('Unsupported closed-loop transfer snapshot')
    for module in snapshot['modules']:
        if not np.isfinite(module['root_state']+[module['joint_position'],module['joint_velocity'],module['holding_torque']]).all():
            raise ValueError('Nonfinite transferred module state')
    return snapshot


def _numbers(values):
    return ' '.join(str(float(v)) for v in values)


def module_spec(index):
    if not 0 <= index < 6:
        raise ValueError('Module index must be 0..5')
    xml = ET.parse(ASSET_PATH).getroot()
    # Scene-level simulation settings are authoritative after MjSpec.attach.
    option=xml.find('option')
    if option is not None:
        xml.remove(option)
    xml.find('compiler').set('meshdir',str(ASSET_PATH.parent/'meshes'))
    world = xml.find('worldbody')
    original = next(body for body in world.iter('body') if body.get('name')==ROOT_NAMES[index])
    root = copy.deepcopy(original)
    tip = next(body for body in root.findall('body') if body.get('name')==TIP_NAMES[index])
    for child in list(tip.findall('body')):
        tip.remove(child)
    for free in root.findall('freejoint'):
        root.remove(free)
    transferred=load_snapshot()['modules'][index]
    root.set('pos',_numbers(transferred['root_state'][:3]))
    root.set('quat',_numbers(transferred['root_state'][3:7]))
    ET.SubElement(root,'freejoint',name='module_root')
    # Make the compiled qpos0 itself a valid closed configuration. The joint
    # reference and rotated body frame cancel, preserving the original meaning
    # of every absolute joint angle while avoiding an overlapping bootstrap.
    reference=transferred['joint_position']
    base_quaternion=np.fromstring(tip.get('quat'),sep=' ')
    composed=np.empty(4)
    mujoco.mju_mulQuat(composed,base_quaternion,np.array([math.cos(reference/2),0.,0.,math.sin(reference/2)]))
    tip.set('quat',_numbers(composed))
    for joint in tip.findall('joint'):
        joint.set('range',f'{-math.pi} {math.pi}')
        joint.set('ref',str(reference))
    for geom in root.iter('geom'):
        geom.set('condim','3')
        geom.set('friction','.6 .1 .1')
    inlet = np.array([-.00022282,-.00010463,0.])
    ET.SubElement(root,'site',name='port_in',pos=_numbers(inlet),size='.004')
    if index < 5:
        relative = original.find('body').find('body')
        position = np.fromstring(relative.get('pos'),sep=' ')
        quaternion = np.fromstring(relative.get('quat'),sep=' ')
        rotation = np.empty(9); mujoco.mju_quat2Mat(rotation,quaternion)
        position = position+rotation.reshape(3,3)@inlet
    else:
        sole = tip.find("site[@name='foot_1_sole']")
        position = np.fromstring(sole.get('pos'),sep=' ')
        quaternion = np.fromstring(sole.get('quat'),sep=' ')
        quarter = load_snapshot()['tail_registration_quarter']
        angle = quarter*math.pi/2.
        composed = np.empty(4)
        mujoco.mju_mulQuat(composed,quaternion,np.array([math.cos(angle/2),0.,0.,math.sin(angle/2)]))
        quaternion = composed
    ET.SubElement(tip,'site',name='port_out',pos=_numbers(position),quat=_numbers(quaternion),size='.004')
    world.clear();world.append(root)
    return mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'))


def add_connectors(spec):
    # Native affine bias can supply implicit PD when enabled by the Action.
    # Zero bias preserves the earlier torque-only actuator exactly.
    for actuator in spec.actuators:
        actuator.biastype=mujoco.mjtBias.mjBIAS_AFFINE
        actuator.biasprm[:]=0.
        actuator.gainprm[0]=1.
        actuator.ctrllimited=False  # force range still clamps NET torque to 20Nm

    # Preserve the original chain's parent-child collision filtering. Splitting
    # it into free roots otherwise introduces overlapping flange contacts and
    # artificial rupture loads at reset. This first head-tail experiment keeps
    # these five internal pairs excluded even after a failure (which terminates).
    for index in range(5):
        spec.add_exclude(bodyname1=f'module_{index}/{TIP_NAMES[index]}',
                         bodyname2=f'module_{index+1}/{ROOT_NAMES[index+1]}')
    for index in range(6):
        spec.add_equality(name=f'connector_{index}',type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_SITE,
                          name1=f'module_{index}/port_out',name2=f'module_{(index+1)%6}/port_in',
                          active=True,data=(*([0.]*10),TORQUE_SCALE),
                          solref=(.005,1.),solimp=(.999,.9999,.001,.5,2.))
