import torch
import torch.distributed as dist
from typing import Any, Type
from torch.optim import Optimizer

class DDPOpSharing(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        self._param_to_owner = {} # id(param) -> owner rank
        self._local_param_groups = []
        self.local_optimizer = None

        super().__init__(params, defaults={})
        self.local_optimizer = optimizer_cls(self._local_param_groups, **kwargs)
        self.state = self.local_optimizer.state


    def add_param_group(self, param_group:dict[str,Any]):
        super().add_param_group(param_group)
        params = param_group["params"]
        n = len(params)

        base = n // self.world_size
        reminder = n % self.world_size

        start = self.rank * base + min(self.rank, reminder)
        end = start + base + (1 if self.rank < reminder else 0)

        for r in range(self.world_size):
            r_start = r * base + min(r, reminder)
            r_end = r_start + base + (1 if r < reminder else 0)
            for p in params[r_start:r_end]:
                self._param_to_owner[id(p)] = r

        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = params[start:end]

        if self.local_optimizer is None:
            self._local_param_groups.append(local_group)
        else:
            self.local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = self.local_optimizer.step(closure=closure, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                dist.broadcast(p.data, src=self._param_to_owner[id(p)])

        return loss


