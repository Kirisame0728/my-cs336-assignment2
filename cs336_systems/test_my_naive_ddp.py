import torch
import torch.multiprocessing as mp

from cs336_systems.naive_ddp import (
    find_free_port,
    setup,
    cleanup,
    data_parallelism_main,
    single_process_train,
)


def compare_params(params_a, params_b, atol=1e-6, rtol=1e-5):
    assert len(params_a) == len(params_b)
    for i, (a, b) in enumerate(zip(params_a, params_b)):
        assert torch.allclose(a, b, atol=atol, rtol=rtol), \
            f"param[{i}] mismatch, max_diff={(a - b).abs().max().item()}"


def ddp_worker(rank, backend, world_size, master_port, data, num_layers, num_steps, return_dict):
    torch.manual_seed(0)
    setup(rank, world_size, backend, master_port)
    data_parallelism_main(rank, backend, world_size, data, num_layers, num_steps, return_dict)


def test_naive_ddp_matches_single_process():
    world_size = 2
    backend = "gloo"
    num_layers = 3
    num_steps = 5
    batch_size = 20
    num_dim = 10

    assert batch_size % world_size == 0

    # toy data
    torch.manual_seed(0)
    data = torch.randn(batch_size, num_dim)

    torch.manual_seed(0)
    init_params = [
        torch.nn.Parameter(torch.randn(num_dim, num_dim))
        for _ in range(num_layers)
    ]
    baseline_params = single_process_train(
        data=data,
        num_layers=num_layers,
        num_steps=num_steps,
        init_params=init_params,
    )

    master_port = find_free_port()
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        ddp_worker,
        args=(backend, world_size, master_port, data, num_layers, num_steps, return_dict),
        nprocs=world_size,
        join=True,
    )

    ddp_params = return_dict["ddp_params"]
    compare_params(baseline_params, ddp_params)