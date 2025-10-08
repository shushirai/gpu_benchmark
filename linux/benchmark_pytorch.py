import torch
import time
from torchvision.models import resnet50

def benchmark_on_device(device, batch_size=64, input_shape=(3, 224, 224), num_classes=10,
                        warmup_batches=5, timing_batches=20, use_dataparallel=False):
    print(f"\n=== Benchmarking on {device} ===")
    if device.type == 'cuda':
        print(f"🖥️ GPU Name: {torch.cuda.get_device_name(device)}")

    # モデル構築
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # 複数GPU対応
    if use_dataparallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"✅ Using DataParallel with {torch.cuda.device_count()} GPUs")

    model.to(device)
    model.train()

    # 入力とラベルの作成
    inputs = torch.randn((batch_size, *input_shape)).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def train_step():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Warm-up
    print("🔥 Warming up...")
    for _ in range(warmup_batches):
        train_step()
        sync()

    # Benchmark
    print("🚀 Benchmarking...")
    start = time.time()
    for _ in range(timing_batches):
        train_step()
        sync()
    end = time.time()

    avg_time = (end - start) / timing_batches
    samples_per_sec = batch_size / avg_time

    # VRAM usage (only for CUDA)
    if device.type == 'cuda':
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        reserv = torch.cuda.memory_reserved(device) / 1024**2
        print(f"📦 VRAM Allocated: {alloc:.2f} MB")
        print(f"📦 VRAM Reserved:  {reserv:.2f} MB")

    print(f"⏱️ Avg time per batch: {avg_time:.4f} sec")
    print(f"📈 Throughput: {samples_per_sec:.2f} images/sec")

# ==== 実行 ====
if __name__ == "__main__":
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            benchmark_on_device(device)
        
        # DataParallelベンチマーク（全GPU使用）
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda")
            benchmark_on_device(device, use_dataparallel=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        benchmark_on_device(torch.device("mps"))
    else:
        benchmark_on_device(torch.device("cpu"))
