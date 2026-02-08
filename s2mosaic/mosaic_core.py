import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from tqdm.auto import tqdm

from .data_reader import get_band_with_mask
from .helpers import (
    MOSAIC_FIRST,
    MOSAIC_MAX_NDVI,
    MOSAIC_MEAN,
    MOSAIC_PERCENTILE,
    format_progress,
)
from .masking import get_masks
from .mosaic_utils import calculate_percentile_mosaic

logger = logging.getLogger(__name__)


def download_bands_pool(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    coverage_mask: np.ndarray,
    no_data_threshold: Union[float, None],
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
    max_dl_workers: int = 16,  # Increased for high-core systems
    percentile_value: float | None = 50.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    s2_scene_size = 10980
    possible_pixel_count = coverage_mask.sum()

    logger.info(f"Possible pixel count: {possible_pixel_count}")

    if "visual" in required_bands:
        band_count = 3
        band_indexes = [1, 2, 3]
        required_bands = required_bands * 3

    else:
        band_count = len(required_bands)
        band_indexes = [1] * len(required_bands)

    if mosaic_method == MOSAIC_PERCENTILE:
        # For percentile, we need to store all values for each pixel
        all_scene_data = []
    elif mosaic_method == MOSAIC_MAX_NDVI:
        # For max_ndvi, we initialize mosaic normally but also need a buffer for NDVI values
        # Initialize with -1.0 (NDVI range is -1 to 1)
        mosaic = np.zeros((band_count, s2_scene_size, s2_scene_size), dtype=np.float32)
        max_ndvi_buffer = np.full((s2_scene_size, s2_scene_size), -2.0, dtype=np.float32)

        # Check if B04 and B08 are available in item assets, regardless of required_bands
        # This will be handled inside the loop for each item
    else:
        # For mean and first, use the existing approach
        mosaic = np.zeros((band_count, s2_scene_size, s2_scene_size), dtype=np.float32)

    good_pixel_tracker = np.zeros((s2_scene_size, s2_scene_size), dtype=np.uint16)

    pbar = tqdm(
        total=len(sorted_scenes),
        desc=format_progress(0, len(sorted_scenes), 100.0),
        leave=False,
        bar_format="{desc}",
    )

    for index, item in enumerate(sorted_scenes["item"].tolist()):
        non_cloud_pixels, valid_pixels = get_masks(
            item=item,
            batch_size=ocm_batch_size,
            inference_dtype=ocm_inference_dtype,
            debug_cache=debug_cache,
            max_dl_workers=max_dl_workers,
        )

        combo_mask = (non_cloud_pixels * valid_pixels).astype(bool)

        # if method is first, only download valid,
        # non cloudy pixels that have not been filled,
        # else download all valid non cloudy pixels
        if mosaic_method == MOSAIC_FIRST:
            combo_mask = (good_pixel_tracker == 0) & combo_mask
        
        # For max_ndvi, we need to download NDVI bands first to determine which pixels to keep
        current_ndvi_mask = None
        if mosaic_method == MOSAIC_MAX_NDVI:
            # We want to process pixels that are valid AND (either not filled yet OR have higher NDVI)
            # But to know if they have higher NDVI, we must download bands first.
            # So we use the standard combo_mask to download potential candidates.
            # We will filter them after calculating NDVI.
            pass

        good_pixel_tracker += combo_mask

        hrefs_and_indexes = [
            (item.assets[band].href, band_index)
            for band, band_index in zip(required_bands, band_indexes, strict=False)
        ]

        get_band_with_mask_partial = partial(
            get_band_with_mask,
            mask=combo_mask,
            debug_cache=debug_cache,
            mosaic_method=mosaic_method,
        )

        with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
            if mosaic_method == MOSAIC_MAX_NDVI:
                # 1. Download Red (B04) and NIR (B08) for NDVI calculation
                ndvi_bands = ["B04", "B08"]
                ndvi_hrefs_indexes = [
                    (item.assets[band].href, 1) # B04/B08 are usually band 1 in their assets
                    for band in ndvi_bands
                ]
                
                get_ndvi_bands_partial = partial(
                    get_band_with_mask,
                    mask=combo_mask,
                    debug_cache=debug_cache,
                    mosaic_method=mosaic_method,
                )
                
                ndvi_results = list(
                    executor.map(get_ndvi_bands_partial, ndvi_hrefs_indexes)
                )
                
                # Process Red and NIR
                red_data, _ = ndvi_results[0]
                nir_data, _ = ndvi_results[1]
                
                # Resize to scene size
                red_data = scipy.ndimage.zoom(
                    red_data,
                    (s2_scene_size / red_data.shape[0], s2_scene_size / red_data.shape[1]),
                    order=0,
                )
                nir_data = scipy.ndimage.zoom(
                    nir_data,
                    (s2_scene_size / nir_data.shape[0], s2_scene_size / nir_data.shape[1]),
                    order=0,
                )
                
                # Calculate NDVI
                # Prevent division by zero
                denom = (nir_data.astype(np.float32) + red_data.astype(np.float32))
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = (nir_data.astype(np.float32) - red_data.astype(np.float32)) / denom
                
                ndvi[denom == 0] = -2.0 # Handle no data/zero division
                
                # Determine better pixels (higher NDVI) within the valid cloud-free area
                # We only want updates where combo_mask is True AND NDVI > current max
                update_mask = combo_mask & (ndvi > max_ndvi_buffer)
                
                # Update NDVI buffer
                max_ndvi_buffer[update_mask] = ndvi[update_mask]
                
                # Now we only download the REQUIRED bands for the pixels in update_mask
                if not np.any(update_mask):
                    # No better pixels found, skip downloading required bands
                    bands_and_profiles = []
                    # We might need to handle 'last_profile' if it's not set, but let's assume valid flow
                else:
                     # Update the mask to only fetch better pixels
                    download_mask = update_mask
                    
                    get_band_update_partial = partial(
                        get_band_with_mask,
                        mask=download_mask,
                        debug_cache=debug_cache,
                        mosaic_method=mosaic_method,
                    )
                    
                    bands_and_profiles = list(
                        executor.map(get_band_update_partial, hrefs_and_indexes)
                    )

            else:
                # Standard flow for other methods
                bands_and_profiles = list(
                    executor.map(get_band_with_mask_partial, hrefs_and_indexes)
                )

        bands = []

        for band, profile in bands_and_profiles:
            bands.append(
                scipy.ndimage.zoom(
                    band,
                    (s2_scene_size / band.shape[0], s2_scene_size / band.shape[1]),
                    order=0,
                )
            )
            last_profile = profile

        scene_data = np.array(bands)

        if mosaic_method == MOSAIC_PERCENTILE:
            scene_data = np.where(combo_mask, scene_data, np.nan)
            all_scene_data.append(scene_data)
        elif mosaic_method == MOSAIC_MAX_NDVI:
            if np.any(update_mask) and len(bands) > 0:
                # We only have data for update_mask pixels
                # Directly update the mosaic array for these pixels
                # bands is shaped (bands, rows, cols)
                for b_idx in range(scene_data.shape[0]):
                    mosaic[b_idx][update_mask] = scene_data[b_idx][update_mask]
        else:
            mosaic += scene_data

        completed_of_possible = coverage_mask * (good_pixel_tracker != 0)
        no_data_sum = coverage_mask.sum() - completed_of_possible.sum()
        logger.info(f"No data sum: {no_data_sum}")

        no_data_pct = (1 - (completed_of_possible.sum() / possible_pixel_count)) * 100
        logger.info(f"No data pct: {no_data_pct}")

        pbar.set_description(
            format_progress(index + 1, len(sorted_scenes), no_data_pct)
        )

        if mosaic_method == MOSAIC_FIRST:
            if no_data_sum == 0:
                break

        # if no_data_threshold is set, stop if threshold is reached
        if no_data_threshold is not None:
            if no_data_sum < (possible_pixel_count * no_data_threshold):
                break
        pbar.update(1)

    remaining_scenes = pbar.total - pbar.n
    pbar.update(remaining_scenes)
    pbar.refresh()
    pbar.close()

    if mosaic_method == MOSAIC_PERCENTILE:
        if percentile_value is None:
            raise ValueError("Percentile must be provided for percentile mosaic method")

        max_workers = multiprocessing.cpu_count() // 2

        mosaic = calculate_percentile_mosaic(
            all_scene_data=all_scene_data,
            s2_scene_size=s2_scene_size,
            max_workers=max_workers,
            percentile_value=float(percentile_value),
        )

    # For max_ndvi, logic is done in-place, no post-processing needed except clipping
    
    if mosaic_method == MOSAIC_MEAN:
        mosaic = np.divide(
            mosaic,
            good_pixel_tracker,
            out=np.zeros_like(mosaic),
            where=good_pixel_tracker != 0,
        )

    if "visual" in required_bands:
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    else:
        mosaic = np.clip(mosaic, 0, 65535).astype(np.int16)

    return mosaic, last_profile
