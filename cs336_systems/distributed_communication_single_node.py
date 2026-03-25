import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import socket

MB = 1024 ** 2
GB = 1024 ** 3

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

def cleanup():
    torch.distributed.destroy_process_group()

def get_device(rank, backend):
    if backend == "nccl":
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")

def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def all_reduce_benchmark(rank, world_size, backend, size, warmup_iters, measure_iters, return_dict, master_port):
    setup(rank, world_size, backend, master_port)
    device = get_device(rank, backend)

    num_elements = size // 4
    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
    # Warmup
    dist.barrier()
    synchronize_if_needed(device)
    for _ in range(warmup_iters):
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
        synchronize_if_needed(device)

    # Benchmarking
    times = []
    dist.barrier()
    synchronize_if_needed(device)
    for _ in range(measure_iters):
        dist.barrier()
        begin = timeit.default_timer()
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
        synchronize_if_needed(device)
        end = timeit.default_timer()
        times.append((end - begin) * 1000.0)
    # Gather timing for all ranks
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, times)

    if rank == 0:
        flat_times = [x for rank_times in gathered for x in rank_times]
        result = {
            "backend": backend,
            "world_size": world_size,
            "size_mb": size / MB,
            "global_mean_ms": float(pd.Series(flat_times).mean()),
        }
        key = f"{backend}_{world_size}_{size}"
        return_dict[key] = result
    cleanup()

def run_exps(
    backends=("nccl",),
    world_sizes=(2, 4, 6),
    sizes=(1 * MB, 10 * MB, 100 * MB, 1 * GB),
    warmup_iters=5,
    measure_iters=10,
):
    manager = mp.Manager()
    return_dict = manager.dict()

    for backend in backends:
        for world_size in world_sizes:
            if backend == "nccl" and (
                (not torch.cuda.is_available()) or world_size > torch.cuda.device_count()
            ):
                print(f"Skip backend={backend}, world_size={world_size}: not enough GPUs")
                continue

            for size in sizes:
                master_port = find_free_port()
                print(
                    f"Running backend={backend}, world_size={world_size}, "
                    f"size={size / MB:.0f}MB, port={master_port}"
                )

                mp.spawn(
                    all_reduce_benchmark,
                    args=(
                        world_size,
                        backend,
                        size,
                        warmup_iters,
                        measure_iters,
                        return_dict,
                        master_port,
                    ),
                    nprocs=world_size,
                    join=True,
                )

    df = pd.DataFrame(list(return_dict.values()))
    if len(df) == 0:
        raise RuntimeError("No benchmark results collected.")

    df = df.sort_values(["backend", "world_size", "size_mb"]).reset_index(drop=True)
    return df

def print_markdown_table(df):
    markdown_df = df[
        [
            "backend",
            "world_size",
            "size_mb",
            "global_mean_ms",
        ]
    ].copy()
    print(markdown_df.to_markdown(index=False))

def save_plots(df, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    for backend in sorted(df["backend"].unique()):
        sub_df = df[df["backend"] == backend].sort_values(["world_size", "size_mb"])

        plt.figure(figsize=(8, 5))
        for ws in sorted(sub_df["world_size"].unique()):
            ws_df = sub_df[sub_df["world_size"] == ws].sort_values("size_mb")
            plt.plot(
                ws_df["size_mb"],
                ws_df["global_mean_ms"],
                marker="o",
                label=f"world_size={ws}",
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Tensor size (MB)")
        plt.ylabel("All-reduce runtime (ms)")
        plt.title(f"All-reduce benchmark ({backend})")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"all_reduce_{backend}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {save_path}")
def main():
    df = run_exps()
    print_markdown_table(df)
    save_plots(df, out_dir="plots")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()