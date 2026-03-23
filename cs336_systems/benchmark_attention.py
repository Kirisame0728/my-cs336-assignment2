import torch
import timeit
import statistics
import pandas as pd

def make_qkv(batch_size, seq_len, d_model, device):
    q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    return q, k, v

def pytorch_attention(q, k, v, d):
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out


def time_forward(batch_size, seq_len, d_model, device, model, num_passes=100):
    q, k, v = make_qkv(batch_size, seq_len, d_model, device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(q, k, v, d_model)
            torch.cuda.synchronize()
        times = []
        for _ in range(num_passes):
            start = timeit.default_timer()
            _ = model(q, k, v, d_model)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append((end - start) * 1000)
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)

def memory_before_backward(batch_size, seq_len, d_model, model, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q, k, v = make_qkv(batch_size, seq_len, d_model, device)
    out = model(q, k, v, d_model)
    loss = out.sum()
    torch.cuda.synchronize()
    mem_allocated = torch.cuda.memory_allocated(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    del q, k, v, out, loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return mem_allocated / (1024 ** 2), peak_allocated / (1024 ** 2)

def time_backward(batch_size, seq_len, d_model, device, model, num_passes=100):
    q, k, v = make_qkv(batch_size, seq_len, d_model, device)
    for _ in range(5):
        if q.grad is not None: q.grad = None
        if k.grad is not None: k.grad = None
        if v.grad is not None: v.grad = None
        out = model(q, k, v, d_model)
        loss = out.sum()
        loss.backward(,
        torch.cuda.synchronize()

    times = []
    for _ in range(num_passes):
        if q.grad is not None: q.grad = None
        if k.grad is not None: k.grad = None
        if v.grad is not None: v.grad = None

        start = timeit.default_timer()
        out = model(q, k, v, d_model)
        loss = out.sum()
        loss.backward(,
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append((end - start) * 1000)
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)

def benchmark(batch_size, seq_len, d_model, device, model, num_passes=100):
    result = {
        "d_model": d_model,
        "seq_len": seq_len,
        "forward_mean_ms": None,
        "forward_std_ms": None,
        "forward_memory": None,
        "forward_peak_mem": None,
        "backward_mean_ms": None,
        "backward_std_ms": None,
        "status": "ok",
        "OOM_stage": None,
        "variant": None,
    }
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fwd_time_mean, fwd_time_std = time_forward(batch_size, seq_len, d_model, device, model, num_passes)
        result["forward_mean_ms"] = fwd_time_mean
        result["forward_std_ms"] = fwd_time_std
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        result["OOM_stage"] = "forward"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "OOM"
            result["OOM_stage"] = "forward"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return result
        raise

    try:
        memory, peak = memory_before_backward(batch_size, seq_len, d_model, model, device)
        result["forward_memory"] = memory
        result["forward_peak_mem"] = peak
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        result["OOM_stage"] = "memory_before_backward"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "OOM"
            result["OOM_stage"] = "memory_before_backward"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return result
        raise

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        bwd_mean, bwd_std = time_backward(batch_size, seq_len, d_model, device, model, num_passes)
        result["backward_mean_ms"] = bwd_mean
        result["backward_std_ms"] = bwd_std
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        result["OOM_stage"] = "backward"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "OOM"
            result["OOM_stage"] = "backward"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return result
        raise
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result

def main():
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192]
    b_size = 8
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    results = []
    eager_attention = pytorch_attention
    compiled_attn = torch.compile(pytorch_attention)
    models = [("eager", eager_attention), ("compiled", compiled_attn)]
    total = len(d_models) * len(seq_lens) * len(models)
    idx = 0
    for d_model in d_models:
        for seq_len in seq_lens:
            for variant_name, model in models:
                idx += 1
                print(f"[{idx}/{total}] Running seq_len={seq_len}, d_model={d_model}")
                result = benchmark(b_size, seq_len, d_model, device, model, num_passes=100)
                result["variant"] = variant_name
                results.append(result)
                df = pd.DataFrame(results)
                print(df.to_markdown(index=False))
    print("\nFinal results:")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()