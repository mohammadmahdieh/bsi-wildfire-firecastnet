#!/usr/bin/env python3
"""
Regenerate the cube_mean_std_dict_{h}.json files required by FireCastNet's
inference_fcn.py.

These statistics MUST come from the ORIGINAL SeasFire cube's training split
(2002-01-01 to 2018-01-01) -- the same data the published checkpoints were
trained on -- NOT from your own operational 2026 data. This script replicates
the exact preprocessing order used in seasfire/data.py of the firecastnet repo:

    1. log1p transform on `tp` and `pop_dens`        (DataModule.__init__)
    2. expand `lsm` along time                        (DataModule.__init__)
    3. shift input vars forward by `target_shift`
       with fill_value=0                              (sample_dataset)
    4. slice training split 2002-01-01 .. 2018-01-01  (sample_dataset)
    5. per-variable mean/std over the whole split     (sample_dataset)

Output: one JSON per horizon, e.g. cube_mean_std_dict_8.json, containing
{var}_mean / {var}_std for every input var, static var, and the target var.
(inference_fcn.py only consumes the input-var entries, but we keep the file
contents identical to what training would have written.)

Usage:
    # Download the SeasFire cube (v0.4, ~44GB) from Zenodo first:
    #   https://zenodo.org/records/13834057
    python compute_mean_std_dicts.py --cube-path /path/to/seasfire_v0.4.zarr

    # Or point directly at a copy in GCS:
    python compute_mean_std_dicts.py --cube-path gs://your-bucket/seasfire_v0.4.zarr

Requires: xarray, zarr, dask, numpy (and gcsfs if reading from GCS).
"""

import argparse
import json
import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Mirrors inference_fcn.py / configs in d-michail/firecastnet
INPUT_VARS = [
    "mslp",
    "tp",
    "vpd",
    "sst",
    "t2m_mean",
    "ssrd",
    "swvl1",
    "lst_day",
    "ndvi",
    "pop_dens",
]
STATIC_VARS = ["lsm"]
LOG_PREPROCESS_INPUT_VARS = ["tp", "pop_dens"]
TARGET_VAR = "gwis_ba"

TRAIN_SLICE = slice("2002-01-01", "2018-01-01")
DEFAULT_HORIZONS = [1, 2, 4, 8, 16, 24]


def compute_stats_for_horizon(
    cube: xr.Dataset,
    target_shift: int,
    target_var_per_area: bool = False,
    target_var_log_process: bool = False,
) -> dict:
    """Replicates sample_dataset() stats computation for one horizon."""
    ds = cube.copy()

    # --- step 3: shift input vars forward by target_shift, fill with 0 ---
    logger.info(f"[h={target_shift}] shifting input vars by {target_shift}")
    for var in INPUT_VARS:
        if target_shift > 0:
            ds[var] = ds[var].shift(time=target_shift, fill_value=0)

    # --- optional target transforms (defaults match the classification
    #     checkpoints; the regression configs may differ, but these only
    #     affect the gwis_ba entries, which inference does not use) ---
    if target_var_per_area:
        logger.info(f"[h={target_shift}] converting target to per-area (hectares)")
        area_ha = ds["area"] / 10000.0
        ds[TARGET_VAR] = ds[TARGET_VAR] / area_ha
    if target_var_log_process:
        logger.info(f"[h={target_shift}] log1p on target var")
        ds[TARGET_VAR] = np.log1p(ds[TARGET_VAR])

    # --- step 4: training split ---
    ds = ds.sel(time=TRAIN_SLICE)
    logger.info(
        f"[h={target_shift}] train split: "
        f"[{ds.time.values[0]}, {ds.time.values[-1]}], {ds.time.size} steps"
    )

    ds = ds[INPUT_VARS + STATIC_VARS + [TARGET_VAR]]

    # --- step 5: per-variable mean/std (xarray default: skipna=True, ddof=0),
    #     identical to data.py: ds[var].mean().values.item(0) ---
    mean_std_dict = {}
    for var in INPUT_VARS + STATIC_VARS + [TARGET_VAR]:
        logger.info(f"[h={target_shift}] computing stats for '{var}'")
        mean_std_dict[var + "_mean"] = float(ds[var].mean().compute().values.item(0))
        mean_std_dict[var + "_std"] = float(ds[var].std().compute().values.item(0))

    return mean_std_dict


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate FireCastNet cube_mean_std_dict_{h}.json files "
        "from the original SeasFire cube."
    )
    parser.add_argument(
        "--cube-path",
        required=True,
        help="Path or gs:// URI to the ORIGINAL SeasFire cube zarr (v0.4).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Target shifts (horizons) to compute, default: 1 2 4 8 16 24",
    )
    parser.add_argument(
        "--prefix",
        default="cube",
        help="Output filename prefix (default 'cube' -> cube_mean_std_dict_{h}.json)",
    )
    parser.add_argument(
        "--target-var-per-area",
        action="store_true",
        help="Divide target by cell area before stats (regression configs).",
    )
    parser.add_argument(
        "--target-var-log-process",
        action="store_true",
        help="log1p the target before stats (regression configs).",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logger.info(f"Opening SeasFire cube: {args.cube_path}")
    # consolidated=False matches how the firecastnet repo opens the cube.
    cube = xr.open_zarr(args.cube_path, consolidated=False)

    # --- step 1: log1p on tp and pop_dens (kept lazy via xr.apply_ufunc-free
    #     dask arithmetic; np.log1p on a DataArray stays lazy with dask) ---
    for var_name in LOG_PREPROCESS_INPUT_VARS:
        logger.info(f"log1p transform on input var: {var_name}")
        cube[var_name] = np.log1p(cube[var_name])

    # --- step 2: expand static vars along time ---
    for static_v in STATIC_VARS:
        if "time" not in cube[static_v].dims:
            logger.info(f"Expanding time dimension on static var: {static_v}")
            cube[static_v] = cube[static_v].expand_dims(
                dim={"time": cube.time}, axis=0
            )

    for h in args.horizons:
        stats = compute_stats_for_horizon(
            cube,
            target_shift=h,
            target_var_per_area=args.target_var_per_area,
            target_var_log_process=args.target_var_log_process,
        )
        out_name = f"{args.prefix}_mean_std_dict_{h}.json"
        with open(out_name, "w") as f:
            json.dump(stats, f)
        logger.info(f"Wrote {out_name}")
        logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
