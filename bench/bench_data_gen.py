#!/usr/bin/env python3
import argparse
import time
import torch


def generate_batch(
    batch_size: int,
    num_points: int,
    num_features: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    sigma_b0: float,
    sigma_beta: float,
    mu_sigma: float,
    sigma_sigma: float,
):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    b0 = (
        torch.randn(
            (batch_size, 1),
            device=device,
            generator=gen,
            dtype=torch.float32,
        )
        * sigma_b0
    )
    beta = (
        torch.randn(
            (batch_size, num_features),
            device=device,
            generator=gen,
            dtype=torch.float32,
        )
        * sigma_beta
    )

    log_sigma = mu_sigma + sigma_sigma * torch.randn(
        (batch_size, 1),
        device=device,
        generator=gen,
        dtype=torch.float32,
    )
    sigma_noise = torch.exp(log_sigma)

    x = torch.randn(
        (batch_size, num_points, num_features),
        device=device,
        generator=gen,
        dtype=torch.float32,
    )
    eps = (
        torch.randn(
            (batch_size, num_points),
            device=device,
            generator=gen,
            dtype=torch.float32,
        )
        * sigma_noise
    )

    y = b0 + (x * beta.unsqueeze(1)).sum(dim=-1) + eps

    x = x.to(dtype)
    y = y.to(dtype)

    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark on-the-fly data generation."
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--points", type=int, default=102048)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)

    parser.add_argument("--sigma-b0", type=float, default=1.0)
    parser.add_argument("--sigma-beta", type=float, default=1.0)
    parser.add_argument("--mu-sigma", type=float, default=-1.0)
    parser.add_argument("--sigma-sigma", type=float, default=0.3)

    parser.add_argument("--split-m", type=int, default=None)
    parser.add_argument("--split-n", type=int, default=None)

    args = parser.parse_args()

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    for _ in range(args.warmup):
        generate_batch(
            args.batch_size,
            args.points,
            args.features,
            dtype,
            args.device,
            args.seed,
            args.sigma_b0,
            args.sigma_beta,
            args.mu_sigma,
            args.sigma_sigma,
        )
        if args.device == "cuda":
            torch.cuda.synchronize()

    times = []
    for r in range(args.repeats):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x, y = generate_batch(
            args.batch_size,
            args.points,
            args.features,
            dtype,
            args.device,
            args.seed + r,
            args.sigma_b0,
            args.sigma_beta,
            args.mu_sigma,
            args.sigma_sigma,
        )

        if args.split_m is not None and args.split_n is not None:
            m = args.split_m
            n = args.split_n
            x_m = x[:, :m]
            x_n = x[:, m : m + n]
            y_m = y[:, :m]
            y_n = y[:, m : m + n]
            _ = (x_m, x_n, y_m, y_n)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    points = args.batch_size * args.points
    points_per_sec = points / avg

    print(
        f"device={args.device} dtype={args.dtype} B={args.batch_size} "
        f"points={args.points} features={args.features}"
    )
    print(f"avg_time_ms={avg * 1000:.3f} std_ms={std * 1000:.3f}")
    print(f"points_per_sec={points_per_sec:,.2f}")

    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"peak_mem_bytes={peak:,}")


if __name__ == "__main__":
    main()
