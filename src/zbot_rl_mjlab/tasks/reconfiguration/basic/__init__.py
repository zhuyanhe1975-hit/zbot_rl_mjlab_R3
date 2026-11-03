"""Basic restored to its 2026-10-26 13:11 baseline, isolated from PBT."""


def validate_checkpoint(path):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    weights = saved.get('actor_state_dict', {}).get('mlp.0.weight')
    if weights is not None and weights.shape[1] != 59:
        raise ValueError(f'初始Basic需要59维观测，该检查点为{weights.shape[1]}维；请重新训练或选择初始59维模型。')
