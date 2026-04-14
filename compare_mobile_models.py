import argparse
import copy
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


@dataclass
class TrainResult:
    best_val_acc: float
    best_model: nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MobileNet vs ShuffleNet vs EfficientNet for mobile-ready benchmarks."
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--teacher-model", type=str, default="efficientnet_b0")
    parser.add_argument("--distill-alpha", type=float, default=0.4)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--benchmark-onnx", action="store_true")
    parser.add_argument("--onnx-opset", type=int, default=17)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    img_size: int,
    workers: int,
    val_split: float,
    seed: int,
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    full_train = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, transform=eval_transform, download=True)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    # Validation should not use train-time augmentation.
    val_ds.dataset = copy.copy(full_train)
    val_ds.dataset.transform = eval_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}")


def model_title(name: str) -> str:
    mapping = {
        "mobilenet_v3_small": "MobileNetV3-Small",
        "shufflenet_v2_x1_0": "ShuffleNetV2 x1.0",
        "efficientnet_b0": "EfficientNet-B0",
    }
    return mapping.get(name, name)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / max(total_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    teacher_model: Optional[nn.Module] = None,
    distill_alpha: float = 0.4,
    distill_temperature: float = 2.0,
) -> TrainResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    kd_criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                ce_loss = criterion(logits, labels)
                loss = ce_loss

                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_model(images)

                    t = distill_temperature
                    student_log_probs = torch.log_softmax(logits / t, dim=1)
                    teacher_probs = torch.softmax(teacher_logits / t, dim=1)
                    kd_loss = kd_criterion(student_log_probs, teacher_probs) * (t * t)
                    loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_acc = accuracy_from_logits(logits, labels)
            running_loss += loss.item()
            running_acc += batch_acc

            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

        scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = running_acc / max(len(train_loader), 1)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return TrainResult(best_val_acc=best_val_acc, best_model=model)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), path)
        size_bytes = os.path.getsize(path)
        return size_bytes / (1024 * 1024)
    finally:
        if os.path.exists(path):
            os.remove(path)


def benchmark_latency_ms(
    model: nn.Module,
    device: torch.device,
    img_size: int,
    runs: int = 100,
    warmup: int = 20,
) -> float:
    model = model.to(device)
    model.eval()

    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / runs) * 1000.0


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    model_cpu = copy.deepcopy(model).cpu().eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def estimate_complexity(model: nn.Module, img_size: int) -> Dict[str, float]:
    params_m = count_trainable_params(model) / 1_000_000.0
    macs_g = float("nan")
    flops_g = float("nan")

    if thop_profile is not None:
        probe = copy.deepcopy(model).cpu().eval()
        sample = torch.randn(1, 3, img_size, img_size)
        try:
            macs, _ = thop_profile(probe, inputs=(sample,), verbose=False)
            macs_g = float(macs) / 1_000_000_000.0
            flops_g = (2.0 * float(macs)) / 1_000_000_000.0
        except Exception:
            pass

    return {
        "params_m": params_m,
        "macs_g": macs_g,
        "flops_g": flops_g,
    }


def export_model_to_onnx(model: nn.Module, output_path: str, img_size: int, opset: int) -> None:
    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def benchmark_onnx_latency_ms(
    onnx_path: str,
    img_size: int,
    runs: int = 100,
    warmup: int = 20,
) -> float:
    if ort is None:
        return float("nan")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sample = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: sample})

    start = time.perf_counter()
    for _ in range(runs):
        _ = session.run(None, {input_name: sample})
    elapsed = time.perf_counter() - start

    return (elapsed / runs) * 1000.0


def _save_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    out_path: str,
    title: str,
    ylabel: str,
) -> None:
    plot_df = df[["model_title", value_col]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(value_col, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot_df["model_title"], plot_df[value_col], color=["#2E7D32", "#1565C0", "#EF6C00", "#6A1B9A"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plot_df = df[["model_title", x_col, y_col]].dropna().copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = (plot_df[y_col].fillna(plot_df[y_col].median()) * 180).clip(lower=60, upper=700)
    ax.scatter(plot_df[x_col], plot_df[y_col], s=sizes, c="#1565C0", alpha=0.8, edgecolors="black", linewidth=0.5)

    for _, row in plot_df.iterrows():
        ax.annotate(str(row["model_title"]), (row[x_col], row[y_col]), textcoords="offset points", xytext=(6, 4))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_result_plots(df: pd.DataFrame, output_dir: str) -> List[str]:
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    generated: List[str] = []

    acc_path = os.path.join(plots_dir, "accuracy_comparison.png")
    _save_bar_plot(
        df=df,
        value_col="test_acc",
        out_path=acc_path,
        title="Test Accuracy Comparison",
        ylabel="Accuracy",
    )
    if os.path.exists(acc_path):
        generated.append(acc_path)

    latency_path = os.path.join(plots_dir, "cpu_latency_comparison_ms.png")
    _save_bar_plot(
        df=df,
        value_col="cpu_latency_ms",
        out_path=latency_path,
        title="CPU Latency Comparison",
        ylabel="Latency (ms/image)",
    )
    if os.path.exists(latency_path):
        generated.append(latency_path)

    pareto_path = os.path.join(plots_dir, "pareto_accuracy_latency.png")
    _save_scatter_plot(
        df=df,
        x_col="cpu_latency_ms",
        y_col="test_acc",
        out_path=pareto_path,
        title="Pareto View: Accuracy vs CPU Latency",
        xlabel="CPU Latency (ms/image)",
        ylabel="Test Accuracy",
    )
    if os.path.exists(pareto_path):
        generated.append(pareto_path)

    size_acc_path = os.path.join(plots_dir, "size_vs_accuracy.png")
    _save_scatter_plot(
        df=df,
        x_col="size_mb",
        y_col="test_acc",
        out_path=size_acc_path,
        title="Model Size vs Accuracy",
        xlabel="Model Size (MB)",
        ylabel="Test Accuracy",
    )
    if os.path.exists(size_acc_path):
        generated.append(size_acc_path)

    if "quant_cpu_latency_ms" in df.columns and df["quant_cpu_latency_ms"].notna().any():
        q_latency_path = os.path.join(plots_dir, "quantized_latency_comparison_ms.png")
        _save_bar_plot(
            df=df,
            value_col="quant_cpu_latency_ms",
            out_path=q_latency_path,
            title="Quantized CPU Latency Comparison",
            ylabel="Latency (ms/image)",
        )
        if os.path.exists(q_latency_path):
            generated.append(q_latency_path)

    if "onnx_cpu_latency_ms" in df.columns and df["onnx_cpu_latency_ms"].notna().any():
        onnx_latency_path = os.path.join(plots_dir, "onnx_cpu_latency_comparison_ms.png")
        _save_bar_plot(
            df=df,
            value_col="onnx_cpu_latency_ms",
            out_path=onnx_latency_path,
            title="ONNX Runtime CPU Latency Comparison",
            ylabel="Latency (ms/image)",
        )
        if os.path.exists(onnx_latency_path):
            generated.append(onnx_latency_path)

    return generated


def run_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, object]:
    print(f"\n=== Training {model_name} ===")
    model = build_model(model_name)

    teacher_model: Optional[nn.Module] = None
    trained_with_distillation = False
    if args.distill and model_name != args.teacher_model:
        teacher_model = build_model(args.teacher_model)
        teacher_ckpt = os.path.join(args.output_dir, f"{args.teacher_model}_teacher.pt")
        if os.path.exists(teacher_ckpt):
            teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
            teacher_model = teacher_model.to(device).eval()
            print(f"Loaded teacher checkpoint: {teacher_ckpt}")
        else:
            print(
                f"Teacher checkpoint not found at {teacher_ckpt}. "
                "Students will train without distillation for this run."
            )
            teacher_model = None
        trained_with_distillation = teacher_model is not None

    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        teacher_model=teacher_model,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
    )

    best_model = train_result.best_model
    val_acc = train_result.best_val_acc
    test_acc = evaluate(best_model, test_loader, device)

    complexity = estimate_complexity(best_model, img_size=args.img_size)
    size_mb = get_model_size_mb(best_model.cpu())

    cpu_device = torch.device("cpu")
    cpu_latency = benchmark_latency_ms(
        best_model.cpu(),
        device=cpu_device,
        img_size=args.img_size,
        runs=args.benchmark_runs,
    )

    result: Dict[str, object] = {
        "model": model_name,
        "model_title": model_title(model_name),
        "distilled": trained_with_distillation,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "params_m": complexity["params_m"],
        "macs_g": complexity["macs_g"],
        "flops_g": complexity["flops_g"],
        "size_mb": size_mb,
        "cpu_latency_ms": cpu_latency,
    }

    if not args.skip_quant:
        quant_model = apply_dynamic_quantization(best_model)
        quant_size_mb = get_model_size_mb(quant_model)
        quant_test_acc = evaluate(quant_model, test_loader, cpu_device)
        quant_cpu_latency = benchmark_latency_ms(
            quant_model,
            device=cpu_device,
            img_size=args.img_size,
            runs=args.benchmark_runs,
        )

        result.update(
            {
                "quant_test_acc": quant_test_acc,
                "quant_size_mb": quant_size_mb,
                "quant_cpu_latency_ms": quant_cpu_latency,
            }
        )

    if args.export_onnx:
        onnx_dir = os.path.join(args.output_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
        export_model_to_onnx(best_model, onnx_path, img_size=args.img_size, opset=args.onnx_opset)
        result["onnx_path"] = onnx_path

        if args.benchmark_onnx:
            result["onnx_cpu_latency_ms"] = benchmark_onnx_latency_ms(
                onnx_path=onnx_path,
                img_size=args.img_size,
                runs=args.benchmark_runs,
            )

    return result


def train_teacher_if_needed(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    if not args.distill:
        return None

    print(f"\n=== Training Teacher: {args.teacher_model} ===")
    teacher_model = build_model(args.teacher_model)
    teacher_result = train_model(
        model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_teacher = teacher_result.best_model
    teacher_ckpt = os.path.join(args.output_dir, f"{args.teacher_model}_teacher.pt")
    torch.save(best_teacher.state_dict(), teacher_ckpt)
    print(f"Saved teacher checkpoint: {teacher_ckpt}")

    teacher_test_acc = evaluate(best_teacher, test_loader, device)
    teacher_complexity = estimate_complexity(best_teacher, img_size=args.img_size)
    teacher_size_mb = get_model_size_mb(best_teacher.cpu())
    teacher_cpu_latency = benchmark_latency_ms(
        best_teacher.cpu(),
        device=torch.device("cpu"),
        img_size=args.img_size,
        runs=args.benchmark_runs,
    )

    return {
        "model": args.teacher_model,
        "model_title": f"{model_title(args.teacher_model)} (teacher)",
        "distilled": False,
        "val_acc": teacher_result.best_val_acc,
        "test_acc": teacher_test_acc,
        "params_m": teacher_complexity["params_m"],
        "macs_g": teacher_complexity["macs_g"],
        "flops_g": teacher_complexity["flops_g"],
        "size_mb": teacher_size_mb,
        "cpu_latency_ms": teacher_cpu_latency,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        workers=args.workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model_names: List[str] = [
        "mobilenet_v3_small",
        "shufflenet_v2_x1_0",
        "efficientnet_b0",
    ]

    all_results: List[Dict[str, object]] = []

    teacher_summary = train_teacher_if_needed(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )
    if teacher_summary is not None:
        all_results.append(teacher_summary)

    for model_name in model_names:
        result = run_single_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            args=args,
        )
        all_results.append(result)

    df = pd.DataFrame(all_results)
    df = df.sort_values(by=["test_acc", "cpu_latency_ms"], ascending=[False, True]).reset_index(drop=True)

    numeric_cols = [
        "val_acc",
        "test_acc",
        "params_m",
        "macs_g",
        "flops_g",
        "size_mb",
        "cpu_latency_ms",
        "quant_test_acc",
        "quant_size_mb",
        "quant_cpu_latency_ms",
        "onnx_cpu_latency_ms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    csv_path = os.path.join(args.output_dir, "model_comparison.csv")
    md_path = os.path.join(args.output_dir, "model_comparison.md")

    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print("\n=== Final Comparison ===")
    print(df.to_markdown(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")

    plot_paths = generate_result_plots(df, args.output_dir)
    if plot_paths:
        print("Generated plots:")
        for p in plot_paths:
            print(f"- {p}")


if __name__ == "__main__":
    main()
