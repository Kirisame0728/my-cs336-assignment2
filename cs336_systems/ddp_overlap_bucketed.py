import torch
import torch.distributed as dist

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDPBucket(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        self._handles = []
        self._bucket_to_ready_count = {}
        self._param_to_bucket = {}

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        self._buckets = self._build_buckets()

        for bucket_idx, bucket in enumerate(self._buckets):
            self._bucket_to_ready_count[bucket_idx] = 0
            for param in bucket:
                self._param_to_bucket[param] = bucket_idx

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_grad_hook())

    def _build_buckets(self):
        buckets = []
        current_bucket = []
        current_bucket_size = 0

        params = [p for p in self.module.parameters() if p.requires_grad]

        for param in reversed(params):
            param_size = param.numel() * param.element_size()

            if len(current_bucket) > 0 and current_bucket_size + param_size > self.bucket_size_bytes:
                buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0

            current_bucket.append(param)
            current_bucket_size += param_size

        if len(current_bucket) > 0:
            buckets.append(current_bucket)

        return buckets

    def _make_grad_hook(self):
        def hook(param):
            bucket_idx = self._param_to_bucket[param]
            self._bucket_to_ready_count[bucket_idx] += 1

            bucket = self._buckets[bucket_idx]
            if self._bucket_to_ready_count[bucket_idx] == len(bucket):
                grads = [p.grad for p in bucket]
                flat_grad = _flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat_grad, async_op=True)
                self._handles.append((handle, flat_grad, bucket))

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, flat_grad, bucket in self._handles:
            handle.wait()
            flat_grad /= self.world_size
            synced_grads = _unflatten_dense_tensors(flat_grad, [p.grad for p in bucket])
            for param, synced_grad in zip(bucket, synced_grads):
                param.grad.copy_(synced_grad)

        self._handles.clear()
        for bucket_idx in self._bucket_to_ready_count:
            self._bucket_to_ready_count[bucket_idx] = 0
