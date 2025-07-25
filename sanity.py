import torch
from torchvision.models import ResNet18_Weights
from modified_lcpr import LCPR_MatViT
import random
from loguru import logger
import traceback

logger.add("sanity_check.log", format="<green>{time} {level} {message}</green>", level="INFO")

def run_sanity_check():
    """
    Performs a sanity check and computational cost comparison on the LCPR_MatViT model 
    by running a forward pass with different granularity configurations.
    """
    logger.info("--- Running Sanity Check & Cost Analysis for LCPR-MatViT Model ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Initialize the universal model
    try:
        model = LCPR_MatViT.create(weights=ResNet18_Weights.DEFAULT).to(device)
        model.eval()
        logger.info("[SUCCESS] Model created and moved to device.")
    except Exception as e:
        logger.error(f"[FAILURE] Could not create model: {e}")
        return

    # 2. Create dummy input data
    dummy_camera_input = torch.randn(1, 6, 3, 256, 704, device=device)
    dummy_lidar_input = torch.randn(1, 1, 32, 1056, device=device)
    print(f"Created dummy camera input with shape: {dummy_camera_input.shape}")
    print(f"Created dummy LiDAR input with shape: {dummy_lidar_input.shape}")
    logger.info(f"Created dummy camera input with shape: {dummy_camera_input.shape}")
    logger.info(f"Created dummy LiDAR input with shape: {dummy_lidar_input.shape}")

    # 3. Define granularities to test
    test_granularities = {
        "Smallest (ViT Depth=0.25, Width=0.25, Fusion=0.25)": {
            'vit_depth_scale': 0.25,
            'vit_width_scale': 0.25,
            'fusion_granularities': [{'scale': 0.25, 'mid_channel_scale': 0.25}] * 4
        },
        "Medium (ViT Depth=0.75, Width=0.75 Fusion=0.75)": {
            'vit_depth_scale': 0.75,
            'vit_width_scale': 0.75,
            'fusion_granularities': [{'scale': 0.75, 'mid_channel_scale': 0.75}] * 4
        },
        "Largest (ViT Depth=1.0, Width=1.0, Fusion=1.0)": {
            'vit_depth_scale': 1.0,
            'vit_width_scale': 1.0,
            'fusion_granularities': [{'scale': 1.0, 'mid_channel_scale': 1.0}] * 4
        },
        "Mixed (Greedy Search Style)": {
            'vit_depth_scale': 0.5,
            'vit_width_scale': 0.75, # Example depth
            'fusion_granularities': [
                {'scale': 0.25, 'mid_channel_scale': 0.5},
                {'scale': 0.5, 'mid_channel_scale': 0.75},
                {'scale': 0.75, 'mid_channel_scale': 1.0},
                {'scale': 1.0, 'mid_channel_scale': 1.0}
            ]
        }
    }

    # 4. Test forward pass and profile each granularity
    all_passed = True
    results = {}
    for name, granularity_config in test_granularities.items():
        logger.info(f"\n--- Testing granularity: {name} ---")
        try:
            model.configure_subnetwork(granularity_config)
            print("Model configured for sub-network.")
            logger.info(f"Model configured for sub-network: {name}")

            # Profile the forward pass
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False,
                profile_memory=False
            ) as prof:
                with torch.profiler.record_function("model_inference"):
                    with torch.no_grad():
                        output = model(dummy_camera_input, dummy_lidar_input)
            
            # Record results
            cpu_time = prof.key_averages().self_cpu_time_total
            results[name] = cpu_time
            print(f"Computational Cost (CPU Time): {cpu_time / 1e6:.4f} seconds")

            expected_shape = torch.Size([1, 256])
            if output.shape == expected_shape:
                print(f"[SUCCESS] Forward pass completed. Output shape: {output.shape}")
                logger.info(f"[SUCCESS] Forward pass completed for {name}. Output shape: {output.shape}")    
            else:
                print(f"[FAILURE] Forward pass failed. Expected shape {expected_shape}, but got {output.shape}")
                logger.error(f"[FAILURE] Forward pass failed for {name}. Expected shape {expected_shape}, but got {output.shape}")
                all_passed = False

        except Exception as e:
            print(f"[FAILURE] An error occurred during forward pass: {e}")
            logger.error(f"[FAILURE] An error occurred during forward pass for {name}: {e}")
            logger.error(traceback.format_exc())
            all_passed = False
            
    print("\n--- Sanity Check & Cost Analysis Summary ---")
    logger.info("\n--- Sanity Check & Cost Analysis Summary ---")
    if all_passed:
        print("[SUCCESS] All forward pass checks completed successfully!")
        logger.info("[SUCCESS] All forward pass checks completed successfully!")
        logger.info("Computational costs for different granularities:")
        print("\n--- Computational Cost Comparison ---")
        largest_cost = results["Largest (ViT Depth=1.0, Fusion=1.0)"]
        for name, cost in results.items():
            reduction = (1 - (cost / largest_cost)) * 100
            print(f"- {name}: {cost / 1e6:.4f}s | Reduction: {reduction:.2f}%")
            logger.info(f"Computational costs for different granularities:- {name}: {cost / 1e6:.4f}s | Reduction: {reduction:.2f}%")
    else:
        print("[FAILURE] Some checks failed. Please review the errors above.")
        logger.error("[FAILURE] Some checks failed. Please review the errors above.")

if __name__ == '__main__':
    run_sanity_check()