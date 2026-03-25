from cs336_basics.model import BasicsTransformerLM
import torch
import torch.distributed as dist
import socket
import os
import timeit
import torch.nn.functional as F
import torch.multiprocessing as mp

def cleanup():
    torch.distributed.destroy_process_group()


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

def get_model(rank, backend):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=128,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0)
    model.to(get_device(rank, backend))
    return model

def generate_sample_data():
    batch_size = 4
    seq_len = 128
    vocab_size=10000
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return x, y

def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def benchmark_ddp_main(rank, backend, world_size, x, y, return_dict):
    batch_size = x.size(0)
    vocab_size = 10000

    device = get_device(rank, backend)
    local_batch_size = batch_size // world_size

    start_index = rank * local_batch_size
    end_index = (rank + 1) * local_batch_size
    x = x[start_index: end_index].to(device)
    y = y[start_index: end_index].to(device)

    model = get_model(rank, backend)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Warmup
    dist.barrier()
    synchronize_if_needed(device)
    for _ in range(5):
        _, _ = run_training_step(model, optimizer, x, y, vocab_size, device, world_size)
    # Benchmarking
    times = []
    props = []
    dist.barrier()
    synchronize_if_needed(device)
    for _ in range(20):
        time, prop = run_training_step(model, optimizer, x, y, vocab_size, device, world_size)
        times.append(time)
        props.append(prop)
    # Gather timing for all ranks
    local_result = {
        "rank": rank,
        "avg_step_time_ms": sum(times) / len(times),
        "avg_comm_prop": sum(props) / len(props),
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_result)

    if rank == 0:
        return_dict["results"] = gathered



def run_training_step(model, optimizer, x, y, vocab_size, device, world_size):
    begin = timeit.default_timer()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(x)
    loss = F.cross_entropy(outputs.reshape(-1, vocab_size), y.reshape(-1))
    synchronize_if_needed(device)
    loss.backward()

    synchronize_if_needed(device)
    begin_sync = timeit.default_timer()
    # Sync gradients
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
    synchronize_if_needed(device)
    end_sync = timeit.default_timer()

    optimizer.step()

    synchronize_if_needed(device)
    end = timeit.default_timer()

    sync_time = (end_sync - begin_sync) * 1000.0
    time = (end - begin) * 1000.0
    propotion = sync_time / time

    return time, propotion

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
        join=True
    )

    results = return_dict["results"]

    print("Per-rank results:")
    for item in results:
        print(
            f"rank={item['rank']}, "
            f"avg_step_time_ms={item['avg_step_time_ms']:.3f}, "
            f"avg_comm_prop={item['avg_comm_prop']:.3f}"
        )

    mean_step_time = sum(r["avg_step_time_ms"] for r in results) / len(results)
    mean_comm_prop = sum(r["avg_comm_prop"] for r in results) / len(results)

    print(f"\nMean step time across ranks: {mean_step_time:.3f} ms")
    print(f"Mean communication proportion across ranks: {mean_comm_prop:.3f}")


if __name__ == "__main__":
    main()
