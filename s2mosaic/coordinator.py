import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

import numpy as np

from .frequent_coverage import get_frequent_coverage
from .helpers import (
    define_dates,
    export_tif,
    get_extent_from_grid_id,
    get_output_path,
    validate_inputs,
)
from .mosaic_core import download_bands_pool
from .stac_utils import add_item_info, search_for_items, sort_items

logger = logging.getLogger(__name__)


@overload
def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: None = None,
    sort_method: str = "valid_data",
    sort_function: Optional[Callable] = None,
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: Optional[List[str]] = None,
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
    additional_query: Optional[Dict[str, Any]] = None,
    percentile_value: Optional[float] = None,
    ignore_duplicate_items: bool = True,
    bounds: Optional[Union[List[float], Tuple[float, float, float, float]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]: ...


@overload
def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: Union[str, Path] = ...,
    sort_method: str = "valid_data",
    sort_function: Optional[Callable] = None,
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: Optional[List[str]] = None,
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
    additional_query: Optional[Dict[str, Any]] = None,
    percentile_value: Optional[float] = None,
    ignore_duplicate_items: bool = True,
    bounds: Optional[Union[List[float], Tuple[float, float, float, float]]] = None,
) -> Path: ...


def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: Optional[Union[Path, str]] = None,
    sort_method: str = "valid_data",
    sort_function: Optional[Callable] = None,
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: Optional[List[str]] = None,
    no_data_threshold: Union[float, None] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
    additional_query: Optional[Dict[str, Any]] = None,
    percentile_value: Optional[float] = None,
    ignore_duplicate_items: bool = True,
    bounds: Optional[Union[List[float], Tuple[float, float, float, float]]] = None,
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Path]:
    """
    Create a Sentinel-2 mosaic for a specified grid and time range.

    This function generates a mosaic from Sentinel-2 satellite imagery based on the provided
    grid ID and time range. It can either return the mosaic data and metadata or save it as
    a GeoTIFF file.

    Args:
        grid_id (Optional[str]): The ID of the grid area. Optional if bounds are provided.
        start_year (int): The start year of the time range.
        start_month (int, optional): The start month of the time range. Defaults to 1.
        start_day (int, optional): The start day of the time range. Defaults to 1.
        output_dir (Optional[Union[Path, str]], optional): Directory to save the output GeoTIFF.
        sort_method (str, optional): Method to sort scenes. Defaults to "valid_data".
        sort_function (Callable, optional): Custom sorting function.
        mosaic_method (str, optional): Method to create the mosaic. Defaults to "mean".
        duration_years (int, optional): Duration in years. Defaults to 0.
        duration_months (int, optional): Duration in months. Defaults to 0.
        duration_days (int, optional): Duration in days. Defaults to 0.
        required_bands (List[str], optional): List of bands. Defaults to RGB+NIR.
        no_data_threshold (float, optional): Threshold for no data. Defaults to 0.01.
        overwrite (bool, optional): Whether to overwrite. Defaults to True.
        ocm_batch_size (int, optional): Batch size. Defaults to 1.
        ocm_inference_dtype (str, optional): Data type. Defaults to "bf16".
        debug_cache (bool, optional): Whether to cache. Defaults to False.
        additional_query (Dict[str, Any], optional): Additional query parameters.
        percentile_value (Optional[float], optional): Percentile value. Defaults to None.
        ignore_duplicate_items (bool, optional): Remove duplicates. Defaults to True.
        bounds (Optional[Union[List[float], Tuple[float, float, float, float]]], optional): 
            Custom bounds (minx, miny, maxx, maxy). If provided, grid_id is optional.

    Returns:
        Union[Tuple[np.ndarray, Dict[str, Any]], Path]: If output_dir is None, returns a tuple
        containing the mosaic array and metadata dictionary. If output_dir is provided,
        returns the path to the saved GeoTIFF file.

    Raises:
        Exception: If no scenes are found for the specified grid ID and time range.

    Note:
        - The function uses the STAC API to search for Sentinel-2 scenes.
        - If 'visual' is included in required_bands, it will be replaced with 'Red', 'Green', 'Blue' in the output.
        - The time range for scene selection is inclusive of the start date and exclusive of the end date.
    """  # noqa: E501
    if required_bands is None:
        required_bands = ["B04", "B03", "B02", "B08"]

    if additional_query is None:
        additional_query = {"eo:cloud_cover": {"lt": 100}}

    if grid_id is None and bounds is None:
        raise ValueError("Either grid_id or bounds must be provided.")

    if grid_id is None:
        # Use a placeholder if grid_id is not provided but logic expects a string
        # Ideally we should use this "custom_bounds" for logging/naming
        pass

    if sort_function:
        sort_method = "custom"

    # If mosaic method is passed as "median",
    # it is converted to "percentile" with a value of 50.0
    if mosaic_method == "median":
        if percentile_value is not None:
            raise ValueError(
                "percentile_value should not be set when using mosaic_method='median'."
            )
        mosaic_method = "percentile"
        percentile_value = 50.0
    logger.info(
        f"Creating mosaic for grid {grid_id} "
        f"from {start_year}-{start_month:02d}-{start_day:02d} "
        f"to {duration_years} years, {duration_months} months, "
        f"{duration_days} days later using {mosaic_method} method "
        f"with bands {required_bands}."
    )

    validate_inputs(
        sort_method=sort_method,
        mosaic_method=mosaic_method,
        no_data_threshold=no_data_threshold,
        required_bands=required_bands,
        grid_id=grid_id,
        percentile_value=percentile_value,
    )
    logger.info("All inputs validated successfully.")

    start_date, end_date = define_dates(
        start_year,
        start_month,
        start_day,
        duration_years,
        duration_months,
        duration_days,
    )
    if output_dir:
        export_path = get_output_path(
            grid_id=grid_id,
            start_date=start_date,
            end_date=end_date,
            sort_method=sort_method,
            mosaic_method=mosaic_method,
            required_bands=required_bands,
            output_dir=output_dir,
        )

    if output_dir:
        if export_path.exists() and not overwrite:
            return export_path

    if bounds is None and grid_id is not None:
        bounds_poly = get_extent_from_grid_id(grid_id)
        logger.info(
            f"Searching for scenes in grid {grid_id} within bounds {bounds_poly} "
            f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
        )
    elif bounds is not None:
        from shapely.geometry import box
        bounds_poly = box(*bounds)
        logger.info(
            f"Searching for scenes in custom bounds {bounds} "
            f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
        )
    else:
        raise ValueError("Either grid_id or bounds must be provided")

    # Only buffer if using grid_id (legacy behavior to avoid edge effects)
    # For custom bounds, use exact bounds (buffer=0)
    search_bounds = bounds_poly.buffer(-0.05) if grid_id else bounds_poly

    items = search_for_items(
        bounds=search_bounds,
        grid_id=grid_id,
        start_date=start_date,
        end_date=end_date,
        additional_query=additional_query,
        ignore_duplicate_items=ignore_duplicate_items,
    )
    
    search_id = grid_id if grid_id else "custom_area"
    logger.info(f"Found {len(items)} scenes for {search_id}.")
    if len(items) == 0:
        raise Exception(
            f"No scenes found for {search_id} between {start_date.strftime('%Y-%m-%d')} "
            f"and {end_date.strftime('%Y-%m-%d')}"
        )

    # for scenes with only partial S2 coverage work out which pixels are covered
    coverage_mask = get_frequent_coverage(scene_bounds=bounds_poly, scenes=items)

    items_with_orbits = add_item_info(items)

    if not sort_function:
        sorted_items = sort_items(items=items_with_orbits, sort_method=sort_method)
    else:
        sorted_items = sort_function(items=items_with_orbits)

    logger.info(f"Sorted {len(sorted_items)} scenes using {sort_method} method.")

    mosaic, profile = download_bands_pool(
        sorted_scenes=sorted_items,
        required_bands=required_bands,
        no_data_threshold=no_data_threshold,
        mosaic_method=mosaic_method,
        ocm_batch_size=ocm_batch_size,
        ocm_inference_dtype=ocm_inference_dtype,
        debug_cache=debug_cache,
        coverage_mask=coverage_mask,
        percentile_value=percentile_value,
    )
    if "visual" in required_bands:
        required_bands = ["Red", "Green", "Blue"]
        nodata_value = None
    else:
        nodata_value = 0

    if output_dir:
        export_tif(
            array=mosaic,
            profile=profile,
            export_path=export_path,
            required_bands=required_bands,
            nodata_value=nodata_value,
        )
        return export_path

    return mosaic, profile
