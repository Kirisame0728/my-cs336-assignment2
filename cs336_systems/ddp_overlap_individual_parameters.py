import torch
import torch.distributed as dist


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()
        self._handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_grad_hook())


    def _make_grad_hook(self):
        def hook(param):
            self._handles.append((dist.all_reduce(param.grad, async_op=True), param))
        return hook


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, param in self._handles:
            handle.wait()
            param.grad /= self.world_size
        self._handles.clear()
