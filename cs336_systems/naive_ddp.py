import torch
import os
import socket
import torch.distributed as dist
import torch.nn.functional as F

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]

def setup(rank, world_size, backend, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    elif backend == "gloo":
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def get_device(rank, backend):
    if backend == "nccl":
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")

def get_init_params(in_dim, out_dim, rank, backend):
    weight = torch.randn(in_dim, out_dim, device=get_device(rank, backend))
    return torch.nn.Parameter(weight)

def cleanup():
    torch.distributed.destroy_process_group()

def data_parallelism_main(rank, backend, world_size, data, num_layers, num_steps, return_dict):
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = batch_size // world_size
    start_index = rank * local_batch_size
    end_index = (rank + 1) * local_batch_size
    data = data[start_index: end_index].to(get_device(rank, backend))
    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank, backend) for i in range(num_layers)]
    # Synchronize parameters
    for p in params:
        dist.broadcast(p.data, src=0)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for step in range(num_steps):
        optimizer.zero_grad()
        x = data
        for param in params:
            x = x @ param
            x = F.relu(x)
        loss = x.square().mean()
        loss.backward()
        # Sync gradients
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
        optimizer.step()
    if rank == 0:
        return_dict["ddp_params"] = [p.detach().cpu().clone() for p in params]
    cleanup()

def single_process_train(data, num_layers, num_steps, init_params=None):
    num_dim = data.size(1)
    device = torch.device("cpu")

    if init_params is None:
        params = [
            torch.nn.Parameter(torch.randn(num_dim, num_dim, device=device))
            for _ in range(num_layers)
        ]
    else:
        params = [
            torch.nn.Parameter(p.detach().cpu().clone())
            for p in init_params
        ]

    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        optimizer.zero_grad()
        x = data.to(device)
        for param in params:
            x = x @ param
            x = F.relu(x)
        loss = x.square().mean()
        loss.backward()
        optimizer.step()

    return [p.detach().cpu().clone() for p in params]

