from shapely.geometry import Polygon
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Force CPU

from s2mosaic import mosaic
from pathlib import Path

# Triangle polygon near Perth
# format: [(lon, lat), ...]
triangle_coords = [
    (115.8, -32.0),
    (115.85, -32.0),
    (115.825, -31.95),
    (115.8, -32.0)
]
custom_poly = Polygon(triangle_coords)

output_dir = Path("test_output_poly")

print("Testing custom polygon support...")
try:
    result = mosaic(
        geometry=custom_poly,
        start_year=2022,
        start_month=1,
        start_day=1,
        duration_months=1,
        output_dir=output_dir,
        sort_method="valid_data",
        mosaic_method="max_ndvi",
        required_bands=["visual"],
        overwrite=True
    )
    print(f"Success! Mosaic saved to: {result}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
