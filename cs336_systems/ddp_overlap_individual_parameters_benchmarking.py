from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp_overlap_individual_parameters import DDP

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import socket
import os
import timeit


def cleanup():
    dist.destroy_process_group()


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
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def get_device(rank, backend):
    if backend == "nccl":
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def get_model(rank, backend):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=128,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0,
    )
    model.to(get_device(rank, backend))
    return model


def generate_sample_data():
    batch_size = 4
    seq_len = 128
    vocab_size = 10000
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return x, y


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_training_step(ddp_model, optimizer, x, y, vocab_size, device):
    synchronize_if_needed(device)
    begin = timeit.default_timer()

    optimizer.zero_grad(set_to_none=True)

    outputs = ddp_model(x)
    loss = F.cross_entropy(outputs.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()

    synchronize_if_needed(device)
    begin_finish_wait = timeit.default_timer()

    ddp_model.finish_gradient_synchronization()

    synchronize_if_needed(device)
    end_finish_wait = timeit.default_timer()

    optimizer.step()

    synchronize_if_needed(device)
    end = timeit.default_timer()

    step_time_ms = (end - begin) * 1000.0
    finish_wait_ms = (end_finish_wait - begin_finish_wait) * 1000.0
    finish_wait_prop = finish_wait_ms / step_time_ms

    return step_time_ms, finish_wait_ms, finish_wait_prop


def benchmark_ddp_main(rank, backend, world_size, x, y, return_dict):
    batch_size = x.size(0)
    vocab_size = 10000

    device = get_device(rank, backend)
    local_batch_size = batch_size // world_size

    start_index = rank * local_batch_size
    end_index = (rank + 1) * local_batch_size
    x = x[start_index:end_index].to(device)
    y = y[start_index:end_index].to(device)

    model = get_model(rank, backend)
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.module.parameters(), lr=1e-3)

    warmup_steps = 5
    benchmark_steps = 20

    dist.barrier()
    synchronize_if_needed(device)
    for _ in range(warmup_steps):
        run_training_step(ddp_model, optimizer, x, y, vocab_size, device)

    times = []
    finish_wait_times = []
    finish_wait_props = []

    dist.barrier()
    synchronize_if_needed(device)
    for _ in range(benchmark_steps):
        step_time_ms, finish_wait_ms, finish_wait_prop = run_training_step(
            ddp_model, optimizer, x, y, vocab_size, device
        )
        times.append(step_time_ms)
        finish_wait_times.append(finish_wait_ms)
        finish_wait_props.append(finish_wait_prop)

    local_result = {
        "rank": rank,
        "avg_step_time_ms": sum(times) / len(times),
        "avg_finish_wait_ms": sum(finish_wait_times) / len(finish_wait_times),
        "avg_finish_wait_prop": sum(finish_wait_props) / len(finish_wait_props),
    }

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_result)

    if rank == 0:
        return_dict["results"] = gathered


def ddp_worker(rank, backend, world_size, master_port, x, y, return_dict):
    setup(rank, world_size, backend, master_port)
    benchmark_ddp_main(rank, backend, world_size, x, y, return_dict)
    cleanup()


def main():
    world_size = 2
    backend = "nccl"
    master_port = find_free_port()

    x, y = generate_sample_data()

    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        ddp_worker,
        args=(backend, world_size, master_port, x, y, return_dict),
        nprocs=world_size,
        join=True,
    )

    results = return_dict["results"]

    print("Per-rank results:")
    for item in results:
        print(
            f"rank={item['rank']}, "
            f"avg_step_time_ms={item['avg_step_time_ms']:.3f}, "
            f"avg_finish_wait_ms={item['avg_finish_wait_ms']:.3f}, "
            f"avg_finish_wait_prop={item['avg_finish_wait_prop']:.3f}"
        )

    mean_step_time = sum(r["avg_step_time_ms"] for r in results) / len(results)
    mean_finish_wait_ms = sum(r["avg_finish_wait_ms"] for r in results) / len(results)
    mean_finish_wait_prop = sum(r["avg_finish_wait_prop"] for r in results) / len(results)

    print(f"\nMean step time across ranks: {mean_step_time:.3f} ms")
    print(f"Mean finish-wait time across ranks: {mean_finish_wait_ms:.3f} ms")
    print(f"Mean finish-wait proportion across ranks: {mean_finish_wait_prop:.3f}")


if __name__ == "__main__":
    main()