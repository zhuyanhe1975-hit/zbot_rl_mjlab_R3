"""Transfer the closure feature backbone; learn a fresh torque action head."""
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlOnPolicyRunnerCfg


@dataclass
class DisconnectTorqueRunnerCfg(RslRlOnPolicyRunnerCfg):
    transfer_closure_backbone: bool=True


CONTROL_MODE="torque_feedforward_p0_static_tilt_loading_obs31_v10"


class DisconnectTorqueRunner(MjlabOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._closure_transferred=False
        self.transfer_report={}

    def initialize_transfer(self):
        cfg=self.env.unwrapped.cfg
        source=Path(cfg.closure_checkpoint)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=cfg.closure_checkpoint_sha256:
            raise ValueError('Closure checkpoint differs from the transferred physical start; regenerate the snapshot')
        saved=torch.load(source,map_location=self.device,weights_only=False)['actor_state_dict']
        actor=self.alg.get_policy()
        current=actor.state_dict()
        if saved['mlp.0.weight'].shape[1]!=59 or current['mlp.0.weight'].shape[1]!=31:
            raise ValueError('Expected a 59-dimensional closure policy and 31-dimensional feedforward torque policy')
        # Only q/dq are transferred. Account for q now being divided by pi.
        old_columns=list(range(12))
        new_columns=list(range(12))
        with torch.no_grad():
            current['mlp.0.weight'].zero_()
            current['mlp.0.weight'][:,new_columns]=saved['mlp.0.weight'][:,old_columns]
            for name in ('mlp.0.bias','mlp.2.weight','mlp.2.bias'):
                if current[name].shape!=saved[name].shape:
                    raise ValueError('Closure and torque hidden layer dimensions differ')
                current[name].copy_(saved[name])
            for name in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std'):
                if name in current and name in saved:
                    current[name][...,new_columns]=saved[name][...,old_columns]
            if 'obs_normalizer._std' in current:
                current['obs_normalizer._mean'][...,:6].div_(math.pi)
                current['obs_normalizer._var'][...,:6].div_(math.pi**2)
                current['obs_normalizer._std'][...,:6].div_(math.pi)
                # Preserve normalized source features including the additive eps.
                std=saved['obs_normalizer._std'][...,:6].reshape(-1)
                eps=actor.obs_normalizer.eps
                current['mlp.0.weight'][:,:6].mul_((std+math.pi*eps)/(std+eps))
            else:
                current['mlp.0.weight'][:,:6].mul_(math.pi)
            if 'obs_normalizer.count' in current:
                current['obs_normalizer.count'].copy_(saved['obs_normalizer.count'])
            # Source outputs are not reused as Nm. Start with zero feedforward.
            current['mlp.4.weight'].zero_();current['mlp.4.bias'].zero_()
            actor.load_state_dict(current)
        self._closure_transferred=True
        self.transfer_report={'source_checkpoint':str(source),'sha256':digest,
                              'migrated_input_channels':len(new_columns),'torque_head_reset':True}
        return self.transfer_report

    def learn(self,*args,**kwargs):
        if not self._closure_transferred and self.cfg.get('transfer_closure_backbone',True):
            print('DISCONNECT_TRANSFER',self.initialize_transfer())
        return super().learn(*args,**kwargs)

    def save(self,path,infos=None):
        infos={**(infos or {}),'disconnect_transfer':self.transfer_report,
               'closure_transferred':self._closure_transferred,'disconnect_control_mode':CONTROL_MODE}
        super().save(path,infos)

    def load(self,path,*args,**kwargs):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if (saved.get('infos') or {}).get('disconnect_control_mode')!=CONTROL_MODE:
            raise ValueError('该检查点使用旧控制/断裂判据/观测结构或未记录模式，不能作为力矩前馈P0阻尼策略回放/续训；请从新力矩前馈P0阻尼实验训练模型。旧模型已保留。')
        infos=super().load(path,*args,**kwargs)
        if infos and infos.get('closure_transferred'):
            self._closure_transferred=True
            self.transfer_report=infos.get('disconnect_transfer',{})
        return infos
