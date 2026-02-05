import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Force CPU

from s2mosaic import mosaic
from pathlib import Path

# Test configuration: Small custom bbox (part of 50HMH/Perth approx)
# format: (minx, miny, maxx, maxy)
custom_bounds = (115.8, -32.0, 115.85, -31.95)

output_dir = Path("test_output_custom")

print("Testing custom bbox support...")
try:
    result = mosaic(
        grid_id=None,
        bounds=custom_bounds,
        start_year=2022,
        start_month=1,
        start_day=1,
        duration_months=1,
        output_dir=output_dir,
        sort_method="valid_data",
        mosaic_method="max_ndvi", # Test both new features together!
        required_bands=["visual"],
        no_data_threshold=None,
        overwrite=True
    )
    print(f"Success! Mosaic saved to: {result}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
