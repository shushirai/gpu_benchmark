#!/usr/bin/env python3
import argparse, time, platform, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sync(dev):
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        try: torch.mps.synchronize()
        except Exception: pass

def pick_device(name):
    if name == "auto":
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)

def log_env(dev):
    print(f"\n=== Environment Info ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch : {torch.__version__}")
    print(f"Device: {dev}")
    if dev.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    elif dev.type == "mps":
        print(f"GPU   : Apple M-series GPU (Metal)")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print("=========================\n")

# === ベンチ対象 ===
@torch.no_grad()
def matmul_bench(dev, n=6144, runs=30, warmup=5, batched=False):
    B = 8 if batched else 1
    x = torch.randn(B, n, n, device=dev)
    for _ in range(warmup): _ = x @ x
    sync(dev)
    t0 = time.time()
    for _ in range(runs): _ = x @ x
    sync(dev)
    sec = time.time() - t0
    flops = B * 2 * (n**3) * runs
    print(f"[MatMul] n={n} B={B} runs={runs} → {sec:.3f}s ({flops/sec/1e12:.2f} TFLOPS)")

class DeepCNN(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(base*4, 1000)
    def forward(self, x): return self.fc(self.seq(x).flatten(1))

def cnn_forward(dev, hw=384, bs=64, runs=50, warmup=10):
    model = DeepCNN().to(dev)
    x = torch.randn(bs, 3, hw, hw, device=dev)
    for _ in range(warmup): _ = model(x)
    sync(dev)
    t0 = time.time()
    for _ in range(runs): _ = model(x)
    sync(dev)
    sec = time.time() - t0
    print(f"[CNN FWD] hw={hw} batch={bs} runs={runs} → {sec:.3f}s ({bs*runs/sec:.1f} imgs/s)")

def cnn_train(dev, hw=384, bs=64, steps=100, warmup=10, amp=False):
    model = DeepCNN().to(dev)
    x = torch.randn(bs, 3, hw, hw, device=dev)
    y = torch.randint(0, 1000, (bs,), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev.type=="cuda" and amp))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type=="cuda" and amp))
    for _ in range(warmup):
        with amp_ctx: out = model(x); loss = F.cross_entropy(out, y)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
    sync(dev)
    t0 = time.time()
    for _ in range(steps):
        with amp_ctx: out = model(x); loss = F.cross_entropy(out, y)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
    sync(dev)
    sec = time.time() - t0
    print(f"[CNN Train] hw={hw} batch={bs} steps={steps} amp={amp} → {sec:.3f}s ({bs*steps/sec:.1f} imgs/s)")

@torch.no_grad()
def transformer_block(dev, seq=384, dim=1024, heads=16, bs=16, runs=20, warmup=5):
    dh = dim // heads
    q = torch.randn(bs, heads, seq, dh, device=dev)
    k = torch.randn(bs, heads, seq, dh, device=dev)
    v = torch.randn(bs, heads, seq, dh, device=dev)
    for _ in range(warmup):
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(dh), dim=-1)
        _ = attn @ v
    sync(dev)
    t0 = time.time()
    for _ in range(runs):
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(dh), dim=-1)
        _ = attn @ v
    sync(dev)
    sec = time.time() - t0
    print(f"[Transformer] seq={seq} dim={dim} heads={heads} batch={bs} → {sec:.3f}s")

def fft_einsum(dev, size=4096, runs=50, warmup=5):
    x = torch.randn(size, size, device=dev)
    for _ in range(warmup):
        _ = torch.fft.fft2(x); _ = torch.einsum("ik,kj->ij", x, x)
    sync(dev)
    t0 = time.time()
    for _ in range(runs):
        _ = torch.fft.fft2(x); _ = torch.einsum("ik,kj->ij", x, x)
    sync(dev)
    sec = time.time() - t0
    print(f"[FFT+Einsum] size={size} runs={runs} → {sec:.3f}s")

# === メイン ===
def main():
    ap = argparse.ArgumentParser("CLI PyTorch Heavy Benchmark")
    ap.add_argument("--device", default="auto", choices=["auto","mps","cuda","cpu"])
    ap.add_argument("--preset", default="heavy", choices=["medium","heavy","extreme"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batched", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    log_env(dev)

    if args.preset == "medium":  n, hw, bs, seq, dim, heads = 4096, 224, 64, 196, 768, 12
    elif args.preset == "heavy": n, hw, bs, seq, dim, heads = 6144, 384, 96, 384, 1024, 16
    else:                        n, hw, bs, seq, dim, heads = 8192, 512, 128, 512, 1536, 24

    print(f"=== Starting Bench ({args.preset.upper()}) ===")
    matmul_bench(dev, n=n, runs=30, batched=args.batched)
    cnn_forward(dev, hw=hw, bs=bs, runs=40)
    cnn_train(dev, hw=hw, bs=bs, steps=80, amp=args.amp)
    transformer_block(dev, seq=seq, dim=dim, heads=heads, bs=16)
    fft_einsum(dev, size=n//2)
    print("==========================\nBenchmark completed.\n")

if __name__ == "__main__":
    main()
