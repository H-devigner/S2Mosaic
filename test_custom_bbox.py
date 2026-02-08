"""
Test script for custom bbox/geometry from a GeoJSON file.
Usage: python3 test_custom_bbox.py
"""
import json
from s2mosaic import mosaic
from pathlib import Path

# Load your GeoJSON file
geojson_path = Path("test.geojson")

with open(geojson_path) as f:
    geojson_data = json.load(f)

# Extract geometry (handles both Feature and FeatureCollection)
if geojson_data.get("type") == "FeatureCollection":
    geometry = geojson_data["features"][0]["geometry"]
elif geojson_data.get("type") == "Feature":
    geometry = geojson_data["geometry"]
else:
    geometry = geojson_data  # Already a geometry

print(f"Loaded geometry from {geojson_path}")
print(f"Geometry type: {geometry.get('type')}")

# Create mosaic
result = mosaic(
    grid_id=None,
    geometry=geometry,           # Pass GeoJSON directly
    start_year=2023,
    start_month=6,
    start_day=1,
    duration_months=3,
    output_dir=Path("output_custom"),
    mosaic_method="max_ndvi",
    required_bands=["visual"],
    ocm_batch_size=32,           # Optimized for H100
    overwrite=True
)

print(f"Success! Mosaic saved to: {result}")
