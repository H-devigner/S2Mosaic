"""
Optimized test script for max_ndvi method on high-performance hardware.
System: 224 cores, 8x NVIDIA H100 80GB
"""
from s2mosaic import mosaic
from pathlib import Path

# High-performance settings
OCM_BATCH_SIZE = 32  # Larger batch for H100 GPUs
NO_DATA_THRESHOLD = 0.001  # Process more scenes

result = mosaic(
    grid_id="50HMH",
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=3,
    output_dir=Path("test_output_optimized"),
    sort_method="valid_data",
    mosaic_method="max_ndvi",
    required_bands=["visual"],
    no_data_threshold=NO_DATA_THRESHOLD,
    ocm_batch_size=OCM_BATCH_SIZE,
    ocm_inference_dtype="bf16",  # Best for H100
    overwrite=True
)

print(f"Success! Mosaic saved to: {result}")
