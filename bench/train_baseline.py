#!/usr/bin/env python3
import argparse
import gc
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model_baseline import Batch, BaselineTransformerNP


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def resolve_amp_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    return torch.bfloat16


def sdp_backend_status() -> str:
    flags = {}
    for name, fn in (
        ("flash", "flash_sdp_enabled"),
        ("mem_efficient", "mem_efficient_sdp_enabled"),
        ("math", "math_sdp_enabled"),
    ):
        if hasattr(torch.backends.cuda, fn):
            flags[name] = getattr(torch.backends.cuda, fn)()
    if not flags:
        return "backend_flags_unavailable"
    return ", ".join(f"{k}={v}" for k, v in flags.items())


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
    m: int,
    n: int,
):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    b0 = torch.randn((batch_size, 1), device=device, generator=gen) * sigma_b0
    beta = (
        torch.randn((batch_size, num_features), device=device, generator=gen)
        * sigma_beta
    )

    log_sigma = mu_sigma + sigma_sigma * torch.randn(
        (batch_size, 1), device=device, generator=gen
    )
    sigma_noise = torch.exp(log_sigma)

    x = torch.randn(
        (batch_size, num_points, num_features), device=device, generator=gen
    )
    eps = (
        torch.randn((batch_size, num_points), device=device, generator=gen)
        * sigma_noise
    )

    y = b0 + (x * beta.unsqueeze(1)).sum(dim=-1) + eps
    y = y.unsqueeze(-1)

    x = x.to(dtype)
    y = y.to(dtype)

    xc = x[:, :m]
    yc = y[:, :m]
    xt = x[:, m : m + n]
    yt = y[:, m : m + n]

    return Batch(xc=xc, yc=yc, xt=xt, yt=yt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline training step benchmark (ACE-style)."
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )
    parser.add_argument("--amp", action="store_true", help="Enable autocast AMP.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--find-max-batch", action="store_true")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--batch-search-warmup", type=int, default=1)
    parser.add_argument("--batch-search-repeats", type=int, default=2)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--m", type=int, default=100000)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--num-tasks", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--sigma-b0", type=float, default=1.0)
    parser.add_argument("--sigma-beta", type=float, default=1.0)
    parser.add_argument("--mu-sigma", type=float, default=-1.0)
    parser.add_argument("--sigma-sigma", type=float, default=0.3)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1365)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--emb-depth", type=int, default=2)
    parser.add_argument("--emb-hidden", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    dtype = resolve_dtype(args.dtype)
    device = args.device
    amp_enabled = args.amp and device == "cuda"
    amp_dtype = resolve_amp_dtype(args.dtype)
    use_grad_scaler = amp_enabled and amp_dtype == torch.float16

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if device == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = BaselineTransformerNP(
        dim_x=args.features,
        dim_y=1,
        d_model=args.d_model,
        dim_feedforward=args.d_ff,
        n_head=args.n_head,
        num_layers=args.layers,
        num_components=args.components,
        emb_depth=args.emb_depth,
        emb_hidden=args.emb_hidden,
        dropout=0.0,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    total_points = args.m + args.n
    if args.steps is None:
        steps = math.ceil(args.num_tasks / args.batch_size)
        total_tasks = args.num_tasks
    else:
        steps = args.steps
        total_tasks = steps * args.batch_size

    sdp_flags = None
    if device == "cuda":
        sdp_flags = sdp_backend_status()
        print(f"sdp_backend_flags: {sdp_flags}")
        print(
            f"amp_enabled={amp_enabled} amp_dtype={amp_dtype} grad_scaler={use_grad_scaler}"
        )

    def sdp_context():
        if device != "cuda":
            return nullcontext()
        if hasattr(torch.nn, "attention") and hasattr(
            torch.nn.attention, "sdpa_kernel"
        ):
            if hasattr(torch.nn.attention, "SDPBackend"):
                try:
                    backends = [
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    ]
                    return torch.nn.attention.sdpa_kernel(backends)
                except TypeError:
                    return torch.nn.attention.sdpa_kernel(
                        enable_flash=True,
                        enable_mem_efficient=True,
                        enable_math=False,
                    )
            return torch.nn.attention.sdpa_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=False,
            )
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=True,
            enable_math=False,
        )

    def amp_context():
        if not amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def timed_section(fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        t1 = time.perf_counter()
        return out, t1 - t0

    def clear_cuda():
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def run_steps(
        batch_size: int,
        num_steps: int,
        warmup_steps: int,
        seed_base: int,
        print_every: int = 0,
    ) -> dict:
        model.train()
        for i in range(warmup_steps):
            batch = generate_batch(
                batch_size,
                total_points,
                args.features,
                dtype,
                device,
                seed_base + i,
                args.sigma_b0,
                args.sigma_beta,
                args.mu_sigma,
                args.sigma_sigma,
                args.m,
                args.n,
            )
            with sdp_context(), amp_context():
                out = model(batch)
            loss = out["loss"]
            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        data_time = 0.0
        fwd_time = 0.0
        bwd_time = 0.0
        opt_time = 0.0
        step_start = time.perf_counter()
        for step_idx in range(num_steps):
            batch_seed = seed_base + warmup_steps + step_idx
            batch, dt = timed_section(
                lambda: generate_batch(
                    batch_size,
                    total_points,
                    args.features,
                    dtype,
                    device,
                    batch_seed,
                    args.sigma_b0,
                    args.sigma_beta,
                    args.mu_sigma,
                    args.sigma_sigma,
                    args.m,
                    args.n,
                )
            )
            data_time += dt

            with sdp_context(), amp_context():
                out, dt = timed_section(lambda: model(batch))
            fwd_time += dt
            loss = out["loss"]
            if use_grad_scaler:
                _, dt = timed_section(lambda: scaler.scale(loss).backward())
                bwd_time += dt

                _, dt = timed_section(
                    lambda: (
                        scaler.step(optimizer),
                        scaler.update(),
                        optimizer.zero_grad(set_to_none=True),
                    )
                )
                opt_time += dt
            else:
                _, dt = timed_section(lambda: loss.backward())
                bwd_time += dt

                _, dt = timed_section(
                    lambda: (optimizer.step(), optimizer.zero_grad(set_to_none=True))
                )
                opt_time += dt

            elapsed = time.perf_counter() - step_start
            done = step_idx + 1
            eta = (elapsed / done) * (num_steps - done) if done > 0 else 0
            loss_val = loss.item()
            bar_len = 30
            filled = int(bar_len * done / num_steps)
            bar = "=" * filled + ">" * (1 if filled < bar_len else 0) + "." * (bar_len - filled - 1)
            print(
                f"\r  [{bar}] {done}/{num_steps} "
                f"loss={loss_val:.4e} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                end="", flush=True,
            )
            if done == num_steps:
                print()

        total_time = data_time + fwd_time + bwd_time + opt_time
        compute_time = fwd_time + bwd_time + opt_time
        tasks = batch_size * num_steps
        tokens = tasks * total_points
        avg_step_ms = (total_time / num_steps) * 1000 if num_steps > 0 else float("nan")

        return {
            "data_time": data_time,
            "fwd_time": fwd_time,
            "bwd_time": bwd_time,
            "opt_time": opt_time,
            "total_time": total_time,
            "compute_time": compute_time,
            "tasks": tasks,
            "tokens": tokens,
            "avg_step_ms": avg_step_ms,
        }

    def try_batch(batch_size: int, seed_base: int) -> dict:
        try:
            run_steps(
                batch_size,
                args.batch_search_repeats,
                args.batch_search_warmup,
                seed_base,
                print_every=0,
            )
            return {"ok": True}
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                clear_cuda()
                return {"ok": False, "error": "oom"}
            raise

    sweep_history = []
    max_batch = None
    batch_size = args.batch_size

    if args.find_max_batch:
        last_good = None
        first_bad = None
        candidate = args.min_batch

        print(f"--- batch size search (range {args.min_batch}..{args.max_batch}) ---")
        while candidate <= args.max_batch:
            print(f"  trying batch_size={candidate} ... ", end="", flush=True)
            result = try_batch(candidate, args.seed)
            entry = {"batch_size": candidate, "ok": result["ok"]}
            if result["ok"]:
                print("OK")
                last_good = candidate
                candidate *= 2
            else:
                print("OOM")
                first_bad = candidate
                candidate = args.max_batch + 1
            sweep_history.append(entry)

        if last_good is None:
            raise RuntimeError("OOM at min-batch; no valid batch size found.")

        if first_bad is not None:
            lo = last_good + 1
            hi = first_bad - 1
            print(f"  binary search between {lo}..{hi}")
            while lo <= hi:
                mid = (lo + hi) // 2
                print(f"  trying batch_size={mid} ... ", end="", flush=True)
                result = try_batch(mid, args.seed + 1000)
                entry = {"batch_size": mid, "ok": result["ok"]}
                if result["ok"]:
                    print("OK")
                    last_good = mid
                    lo = mid + 1
                else:
                    print("OOM")
                    hi = mid - 1
                sweep_history.append(entry)

        max_batch = last_good
        batch_size = max_batch
        print(f"  => max batch size = {max_batch}")
        steps = math.ceil(args.num_tasks / batch_size)
        print(f"--- training {steps} steps (batch={batch_size}, tasks={args.num_tasks}) ---")
        metrics = run_steps(
            batch_size,
            steps,
            args.warmup,
            args.seed + 2000,
            print_every=args.print_every,
        )
        data_time = metrics["data_time"]
        fwd_time = metrics["fwd_time"]
        bwd_time = metrics["bwd_time"]
        opt_time = metrics["opt_time"]
        total_tasks = args.num_tasks
    else:
        print(f"--- training {steps} steps (batch={batch_size}, tasks={total_tasks}) ---")
        metrics = run_steps(
            args.batch_size,
            steps,
            args.warmup,
            args.seed,
            print_every=args.print_every,
        )
        data_time = metrics["data_time"]
        fwd_time = metrics["fwd_time"]
        bwd_time = metrics["bwd_time"]
        opt_time = metrics["opt_time"]
        total_tasks = metrics["tasks"] if args.steps is not None else total_tasks

    total_time = data_time + fwd_time + bwd_time + opt_time
    tokens = total_tasks * total_points
    compute_time = fwd_time + bwd_time + opt_time
    avg_step_ms = (total_time / steps) * 1000 if steps > 0 else float("nan")
    total_tokens_per_sec = tokens / total_time if total_time > 0 else float("nan")
    compute_tokens_per_sec = tokens / compute_time if compute_time > 0 else float("nan")
    tasks_per_sec = total_tasks / total_time if total_time > 0 else float("nan")
    tasks_per_sec_compute = (
        total_tasks / compute_time if compute_time > 0 else float("nan")
    )
    points_per_sec = tokens / total_time if total_time > 0 else float("nan")
    points_per_sec_compute = tokens / compute_time if compute_time > 0 else float("nan")

    print("--- baseline summary ---")
    print(f"device={device} dtype={args.dtype} batch={batch_size}")
    print(f"m={args.m} n={args.n} features={args.features} total_points={total_points}")
    print(f"tasks={total_tasks} steps={steps}")
    if args.find_max_batch:
        print(f"max_batch={max_batch} sweep_trials={len(sweep_history)}")
    print(f"data_gen_s={data_time:.3f}")
    print(f"forward_s={fwd_time:.3f} backward_s={bwd_time:.3f} optim_s={opt_time:.3f}")
    print(f"total_s={total_time:.3f} compute_only_s={compute_time:.3f}")
    print(f"avg_step_ms={avg_step_ms:.3f}")
    print(f"tasks_per_sec={tasks_per_sec:,.2f}")
    print(f"tasks_per_sec_compute={tasks_per_sec_compute:,.2f}")
    print(f"tokens_per_sec={total_tokens_per_sec:,.2f}")
    print(f"tokens_per_sec_compute={compute_tokens_per_sec:,.2f}")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"peak_mem_bytes={peak:,}")
    else:
        peak = None

    def as_json_number(value: float) -> float | None:
        return value if math.isfinite(value) else None

    if args.json_out:
        out = {
            "config": {
                "device": device,
                "dtype": args.dtype,
                "amp": amp_enabled,
                "amp_dtype": str(amp_dtype) if amp_enabled else None,
                "grad_scaler": use_grad_scaler,
                "batch_size": batch_size,
                "features": args.features,
                "m": args.m,
                "n": args.n,
                "total_points": total_points,
                "num_tasks": total_tasks,
                "steps": steps,
                "find_max_batch": args.find_max_batch,
                "min_batch": args.min_batch,
                "max_batch": args.max_batch,
                "batch_search_warmup": args.batch_search_warmup,
                "batch_search_repeats": args.batch_search_repeats,
                "max_batch_found": max_batch,
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "n_head": args.n_head,
                "layers": args.layers,
                "components": args.components,
                "emb_depth": args.emb_depth,
                "emb_hidden": args.emb_hidden,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "sdp_backend_flags": sdp_flags,
            "timing_s": {
                "data_gen": data_time,
                "forward": fwd_time,
                "backward": bwd_time,
                "optimizer": opt_time,
                "total": total_time,
                "compute_only": compute_time,
            },
            "throughput": {
                "tasks_per_sec": as_json_number(tasks_per_sec),
                "tasks_per_sec_compute": as_json_number(tasks_per_sec_compute),
                "points_per_sec": as_json_number(points_per_sec),
                "points_per_sec_compute": as_json_number(points_per_sec_compute),
                "tokens_per_sec": as_json_number(total_tokens_per_sec),
                "tokens_per_sec_compute": as_json_number(compute_tokens_per_sec),
            },
            "avg_step_ms": as_json_number(avg_step_ms),
            "peak_mem_bytes": peak,
            "sweep_history": sweep_history if args.find_max_batch else None,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
