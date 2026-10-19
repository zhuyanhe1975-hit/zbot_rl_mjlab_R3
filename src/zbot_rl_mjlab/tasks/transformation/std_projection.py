"""将有界 Gaussian 探索参数保持在可微区间内部。"""

import math

import torch
from rsl_rl.modules import GaussianDistribution
from torch.utils.hooks import RemovableHandle


class ActionStdProjection:
    """原位投影参数，不替换参数对象或优化器状态。

    当前安装的 torch 版本中，边界处 clamp 的导数为零。
    因此使用优化器后置钩子，在构建下一次策略分布之前将参数严格保持在声明范围内。
    """

    def __init__(self, distribution):
        self.parameter = None
        if not isinstance(distribution, GaussianDistribution):
            return
        self.is_log = distribution.std_type == "log"
        if self.is_log:
            self.parameter = distribution.log_std_param
            self.bounds = tuple(distribution.log_std_range)
        else:
            self.parameter = distribution.std_param
            self.bounds = tuple(distribution.std_range)
        # 安装时即检查失败，避免在优化器更新中途报错。
        self._projection_bounds = self._interior_bounds()

    def _interior_bounds(self):
        parameter = self.parameter
        if parameter is None:
            return None
        low, high = self.bounds
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Action std bounds must be finite and strictly ordered")
        lower = parameter.new_tensor(low)
        upper = parameter.new_tensor(high)
        inner_lower = torch.nextafter(lower, upper)
        inner_upper = torch.nextafter(upper, lower)
        if self.is_log:
            # nextafter(0, +/-inf) 是次正规数，CUDA 可能将其清零。
            # 尽可能使用普通 epsilon，并采用正规数下界。
            info = torch.finfo(parameter.dtype)
            inset = max(info.tiny, min(info.eps, (high - low) / 4))
            if low == 0:
                inner_lower = parameter.new_tensor(inset)
            if high == 0:
                inner_upper = parameter.new_tensor(-inset)
        valid = (
            (lower < inner_lower) & (inner_lower <= inner_upper) & (inner_upper < upper)
        )
        if not valid.item():
            raise ValueError(
                "Action std range has no representable strict interior "
                f"for {parameter.dtype}: {self.bounds}"
            )
        return lower, upper, inner_lower, inner_upper

    @torch.no_grad()
    def project(self) -> None:
        if self.parameter is None:
            return
        bounds = self._projection_bounds
        if bounds is None:
            return
        lower, upper, inner_lower, inner_upper = bounds
        self.parameter.copy_(
            torch.where(
                self.parameter <= lower,
                inner_lower,
                torch.where(self.parameter >= upper, inner_upper, self.parameter),
            )
        )

    def install(self, optimizer: torch.optim.Optimizer) -> RemovableHandle:
        """注册一个原生优化器步后钩子；调用方负责句柄生命周期。"""

        def project_after_step(optimizer, args, kwargs):
            self.project()

        return optimizer.register_step_post_hook(project_after_step)
