import torch
from thop import profile
import argparse
import warnings
import copy
import numpy as np

# Import loralib to identify its custom layer types
import loralib as lora

# --- Model Imports ---
# The proposed model
from model.FUSE import FUSE

# Image-only baseline model
from model.depth_anything_v2.dpt_align import DepthAnythingV2

# Attention-based fusion baseline model
from model.FUSE_atten import FUSE as FUSE_atten

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


# --- Custom Thop Hooks for LoRA Layers ---
# The standard `thop` library does not recognize custom layers from `loralib`.
# These hook functions teach `thop` how to calculate FLOPs and parameters for them.


def count_relu(m, x, y):
    """A dummy hook for ReLU layers, as they have zero FLOPs and parameters."""
    pass


def count_lora_linear(m: lora.Linear, x: (torch.Tensor,), y: torch.Tensor):
    """Calculates FLOPs and params for a `lora.Linear` layer."""
    x_in = x[0]
    # FLOPs = 2 * MACs. MACs = (input_features * output_features)
    macs = y.numel() * m.in_features

    # Add FLOPs from the LoRA path (A and B matrices) if it exists
    if m.r > 0:
        tokens_count = np.prod(y.shape[:-1])
        macs += tokens_count * (m.in_features * m.r + m.r * m.out_features)

    m.total_ops += torch.DoubleTensor([macs])
    # `thop` automatically counts parameters for registered modules, but this ensures custom ones are included.
    m.total_params += sum(p.numel() for p in m.parameters())


def count_lora_merged_linear(m: lora.MergedLinear, x: (torch.Tensor,), y: torch.Tensor):
    """Calculates FLOPs and params for a `lora.MergedLinear` layer."""
    x_in = x[0]
    # Main path MACs
    macs = y.numel() * m.in_features

    # Handle both integer and list formats for the 'r' attribute
    r_values = m.r if isinstance(m.r, list) else [m.r] * len(m.enable_lora)
    tokens_count = np.prod(y.shape[:-1])

    # Add FLOPs from all enabled LoRA paths
    if any(r > 0 for r in r_values):
        for i, r in enumerate(r_values):
            if r > 0 and i < len(m.lora_B):
                out_features_i = m.lora_B[i].shape[0]
                macs += tokens_count * (m.in_features * r + r * out_features_i)

    m.total_ops += torch.DoubleTensor([macs])
    m.total_params += sum(p.numel() for p in m.parameters())


def measure_fps(model, dummy_input, iterations=300):
    """
    Measures a model's inference speed in Frames Per Second (FPS).
    Uses CUDA events for accurate GPU timing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Running on CPU. FPS measurements might not be accurate."
        )
        return 0, 0

    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()

    # Warm-up runs to stabilize GPU performance
    warmup_iter = 10
    with torch.no_grad():
        for _ in range(warmup_iter):
            _ = model(dummy_input)

    # Precise timing using CUDA events
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    timings = torch.zeros((iterations,))

    with torch.no_grad():
        for i in range(iterations):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            curr_time = starter.elapsed_time(ender)  # Time in milliseconds
            timings[i] = curr_time

    mean_latency_ms = torch.mean(timings).item()
    fps = 1000.0 / mean_latency_ms

    return fps, mean_latency_ms


def main():
    """
    Main function to run the benchmark.
    It compares the proposed FUSE model against two baselines:
    1. An image-only model (DepthAnythingV2).
    2. An attention-based fusion model (FUSE_atten).
    Metrics reported: Parameters, GFLOPs, Latency, and FPS.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark FUSE model against baselines."
    )
    parser.add_argument(
        "--encoder",
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder model type.",
    )
    parser.add_argument(
        "--img-size", default=266, type=int, help="Input image size for the model."
    )
    parser.add_argument(
        "--batch-size", default=1, type=int, help="Batch size for inference."
    )
    parser.add_argument(
        "--event-voxel-chans",
        default=3,
        type=int,
        help="Number of event voxel channels.",
    )
    args = parser.parse_args()

    # Suppress warnings from thop about unhandled operations
    warnings.filterwarnings("ignore", category=UserWarning, module="thop")

    print("--- Starting Benchmark ---")
    print(
        f"Encoder: {args.encoder}, Image Size: {args.img_size}, Batch Size: {args.batch_size}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Register the custom hooks for thop to handle LoRA layers
    custom_ops = {
        torch.nn.ReLU: count_relu,
        lora.Linear: count_lora_linear,
        lora.MergedLinear: count_lora_merged_linear,
    }

    # === 1. Image-Only Baseline (Depth Anything V2) ===
    print("\n--- Benchmarking Baseline 1: Image-Only (Depth Anything V2) ---")
    baseline_model = DepthAnythingV2(**model_configs[args.encoder]).to(device).eval()
    dummy_image_input = torch.randn(
        args.batch_size, 3, args.img_size, args.img_size, device=device
    )

    # Calculate parameters using a reliable method for verification
    baseline_params_actual = sum(p.numel() for p in baseline_model.parameters())

    # Use a deep copy for FLOPs calculation to avoid modifying the original model with thop's hooks
    model_for_flops_baseline = copy.deepcopy(baseline_model)
    baseline_macs, _ = profile(
        model_for_flops_baseline,
        inputs=(dummy_image_input,),
        custom_ops=custom_ops,
        verbose=False,
    )
    del model_for_flops_baseline

    baseline_gflops = baseline_macs * 2 / 1e9
    print(f"Image-Only Model - Parameters (Actual): {baseline_params_actual/1e6:.2f} M")
    print(f"Image-Only Model - GFLOPs: {baseline_gflops:.2f} G")

    baseline_fps, baseline_latency = measure_fps(baseline_model, dummy_image_input)
    print(f"Image-Only Model - Inference Latency: {baseline_latency:.2f} ms")
    print(f"Image-Only Model - FPS: {baseline_fps:.2f}")

    del baseline_model, dummy_image_input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Prepare a shared dummy input for the fusion models
    dummy_fuse_input = torch.randn(
        args.batch_size,
        3 + args.event_voxel_chans,
        args.img_size,
        args.img_size,
        device=device,
    )

    # === 2. Attention Fusion Baseline (FUSE_atten) ===
    atten_params_actual, atten_gflops, atten_fps, atten_latency = 0, 0, 0, 0
    if FUSE_atten is not None:
        print("\n--- Benchmarking Baseline 2: Attention Fusion (FUSE_atten) ---")
        atten_model = (
            FUSE_atten(
                model_name=args.encoder, event_voxel_chans=args.event_voxel_chans
            )
            .to(device)
            .eval()
        )

        atten_params_actual = sum(p.numel() for p in atten_model.parameters())
        model_for_flops_atten = copy.deepcopy(atten_model)
        atten_macs, _ = profile(
            model_for_flops_atten,
            inputs=(dummy_fuse_input,),
            custom_ops=custom_ops,
            verbose=False,
        )
        del model_for_flops_atten

        atten_gflops = atten_macs * 2 / 1e9
        print(
            f"Attention Fusion Model - Parameters (Actual): {atten_params_actual/1e6:.2f} M"
        )
        print(f"Attention Fusion Model - GFLOPs: {atten_gflops:.2f} G")

        atten_fps, atten_latency = measure_fps(atten_model, dummy_fuse_input)
        print(f"Attention Fusion Model - Inference Latency: {atten_latency:.2f} ms")
        print(f"Attention Fusion Model - FPS: {atten_fps:.2f}")

        del atten_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === 3. Proposed Model (FUSE with FreDFuse) ===
    print("\n--- Benchmarking Your Model: FreDFuse Fusion (FUSE) ---")
    fuse_model = (
        FUSE(model_name=args.encoder, event_voxel_chans=args.event_voxel_chans)
        .to(device)
        .eval()
    )

    fuse_params_actual = sum(p.numel() for p in fuse_model.parameters())
    model_for_flops_fuse = copy.deepcopy(fuse_model)
    fuse_macs, _ = profile(
        model_for_flops_fuse,
        inputs=(dummy_fuse_input,),
        custom_ops=custom_ops,
        verbose=False,
    )
    del model_for_flops_fuse

    fuse_gflops = fuse_macs * 2 / 1e9
    print(f"FreDFuse Model - Parameters (Actual): {fuse_params_actual/1e6:.2f} M")
    print(f"FreDFuse Model - GFLOPs: {fuse_gflops:.2f} G")

    fuse_fps, fuse_latency = measure_fps(fuse_model, dummy_fuse_input)
    print(f"FreDFuse Model - Inference Latency: {fuse_latency:.2f} ms")
    print(f"FreDFuse Model - FPS: {fuse_fps:.2f}")

    del fuse_model, dummy_fuse_input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === 4. Summary of Results ===
    print("\n\n--- Benchmark Summary ---")
    print("\n--- Comparison 1: Your FUSE Model vs. Image-Only Baseline ---")
    if baseline_params_actual > 0 and baseline_gflops > 0 and baseline_latency > 0:
        param_overhead = (
            (fuse_params_actual - baseline_params_actual) / baseline_params_actual * 100
        )
        gflops_overhead = (fuse_gflops - baseline_gflops) / baseline_gflops * 100
        latency_overhead = (fuse_latency - baseline_latency) / baseline_latency * 100
        fps_drop = (baseline_fps - fuse_fps) / baseline_fps * 100
        print(
            f"Parameter Overhead: +{(fuse_params_actual - baseline_params_actual)/1e6:.2f} M ({param_overhead:+.2f}%)"
        )
        print(
            f"GFLOPs Overhead: +{fuse_gflops - baseline_gflops:.2f} G ({gflops_overhead:+.2f}%)"
        )
        print(
            f"Latency Increase: +{fuse_latency - baseline_latency:.2f} ms ({latency_overhead:+.2f}%)"
        )
        print(f"FPS Drop: -{baseline_fps - fuse_fps:.2f} FPS ({fps_drop:.2f}%)")

    if FUSE_atten is not None and atten_params_actual > 0:
        print("\n--- Comparison 2: Your FreDFuse vs. Attention Fusion Baseline ---")
        param_overhead_fred = (
            (fuse_params_actual - atten_params_actual) / atten_params_actual * 100
        )
        gflops_overhead_fred = (fuse_gflops - atten_gflops) / atten_gflops * 100
        latency_overhead_fred = (fuse_latency - atten_latency) / atten_latency * 100
        fps_drop_fred = (atten_fps - fuse_fps) / atten_fps * 100
        print("This comparison isolates the cost of the FreDFuse module itself.")
        print(
            f"Parameter Overhead: +{(fuse_params_actual - atten_params_actual)/1e6:.2f} M ({param_overhead_fred:+.2f}%)"
        )
        print(
            f"GFLOPs Overhead: +{fuse_gflops - atten_gflops:.2f} G ({gflops_overhead_fred:+.2f}%)"
        )
        print(
            f"Latency Increase: +{fuse_latency - atten_latency:.2f} ms ({latency_overhead_fred:+.2f}%)"
        )
        print(f"FPS Drop: -{atten_fps - fuse_fps:.2f} FPS ({fps_drop_fred:.2f}%)")


if __name__ == "__main__":
    main()
