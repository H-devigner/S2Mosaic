import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Force CPU to avoid OOM on busy GPU

from s2mosaic import mosaic
from pathlib import Path
import numpy as np

# Test configuration
grid_id = "50HMH" # Known grid from examples
start_year = 2022
start_month = 1
start_day = 1
duration_months = 1
output_dir = Path("test_output")

print("Testing max_ndvi method...")
try:
    result = mosaic(
        grid_id=grid_id,
        start_year=start_year,
        start_month=start_month,
        start_day=start_day,
        duration_months=duration_months,
        output_dir=output_dir,
        sort_method="valid_data",
        mosaic_method="max_ndvi",
        required_bands=["visual"], # Should auto-fetch B04/B08 for calculation
        no_data_threshold=0.1,
        overwrite=True
    )
    print(f"Success! Mosaic saved to: {result}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
