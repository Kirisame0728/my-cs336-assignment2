import torch
import argparse
from cs336_basics.model import BasicsTransformerLM
import torch.nn.functional as F
import timeit
import statistics
import pandas as pd
import torch.cuda.nvtx as nvtx

MODEL_CONFIGS = {
    "small": {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "train"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    return parser.parse_args()

def generate_data(args):
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=args.device, dtype=torch.long)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=args.device, dtype=torch.long)
    return  x, y

def make_model(args):
    configs = MODEL_CONFIGS[args.model_size]
    d_model = configs["d_model"]
    num_layers = configs["num_layers"]
    num_heads = configs["num_heads"]
    d_ff = configs["d_ff"]

    model = BasicsTransformerLM(args.vocab_size, args.seq_len, d_model, num_layers, num_heads, d_ff, args.rope_theta).to(args.device)
    return model

def run_forward(model, x):
    with nvtx.range("forward"):
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()


def run_forward_backward(model, optimizer, x, y, vocab_size):
    optimizer.zero_grad(set_to_none=True)
    with nvtx.range("forward"):
        outputs = model(x)
        torch.cuda.synchronize()

    loss = F.cross_entropy(outputs.reshape(-1, vocab_size), y.reshape(-1))

    with nvtx.range("backward"):
        loss.backward()
        torch.cuda.synchronize()

    with nvtx.range("optimizer_step"):
        optimizer.step()
        torch.cuda.synchronize()

def benchmark(args):
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    model = make_model(args)
    if args.mode == "train":
        model.train()
    else:
        model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) if args.mode == "train" else None
    x, y = generate_data(args)

    torch.cuda.synchronize()
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            if args.mode == "train":
                run_forward_backward(model, optimizer, x, y, args.vocab_size)
            else:
                run_forward(model, x)
    times = []
    with nvtx.range("measured_region"):
        for i in range(args.measure_steps):
            start = timeit.default_timer()
            with nvtx.range(f"measure_step_{i}"):
                if args.mode == "train":
                    run_forward_backward(model, optimizer, x, y, args.vocab_size)
                else:
                    run_forward(model, x)
            end = timeit.default_timer()
            times.append(end - start)
    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times)

    summary_df = pd.DataFrame([
        {
            "mode": args.mode,
            "model_size": args.model_size,
            "seq_len": args.seq_len,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "mean_ms": mean_t * 1000,
            "std_ms": std_t * 1000,
        }
    ])
    print("Summary:")
    print(summary_df.to_markdown(index=False))

def main():
    args = parse_args()
    benchmark(args)

if __name__ == "__main__":
    main()



