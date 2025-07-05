import torch
from torchvision.models import ResNet18_Weights
from modules.LCPR import LCPR

def profile_lcpr():
    """
    Profiles the LCPR model to identify computational bottlenecks.
    """
    # Ensure the model is on the correct device (CPU for this script, can be changed to CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")

    # --- Model Initialization ---
    # Initialize the LCPR model with default ResNet18 weights
    model = LCPR.create(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    # --- Dummy Input Data ---
    # Create realistic dummy inputs based on the project's configuration
    # Camera input: Batch x Num_Cameras x Channels x Height x Width
    # (B, N, C, H, W) = (1, 6, 3, 256, 704) based on config.yaml and README
    dummy_camera_input = torch.randn(1, 6, 3, 256, 704, device=device)

    # LiDAR input: Batch x Channels x Height x Width
    # (B, C, H, W) = (1, 1, 32, 1056) based on tools/gen_range.py
    dummy_lidar_input = torch.randn(1, 1, 32, 1056, device=device)

    # --- Profiling ---
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            model(dummy_camera_input, dummy_lidar_input)

    # --- Print Results ---
    print("--- Profiler Results (CPU time) ---")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))

    if torch.cuda.is_available():
        print("\n--- Profiler Results (CUDA time) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n--- Profiler Results (Memory Usage) ---")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))


if __name__ == '__main__':
    profile_lcpr()
