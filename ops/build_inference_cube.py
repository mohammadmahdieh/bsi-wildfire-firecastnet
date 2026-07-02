#!/usr/bin/env python3
"""
Build a SeasFire-compliant inference cube from per-period Zarr slices in GCS,
ready for the unmodified inference_fcn.py from d-michail/firecastnet.

What it does:
  1. Lists period files under --prefix (or reads explicit URIs from --uris-file).
  2. Sorts chronologically, keeps the latest --timeseries (default 24).
  3. Verifies the periods are CONTIGUOUS under the SeasFire calendar
     (46 periods/year, starting Jan 1, every 8 days; the final period of each
     year is 5-6 days; the sequence resets every Jan 1). Aborts on any gap.
  4. Merges along time; drops extra vars (e.g. d2m); casts float64 -> float32.
  5. Squeezes `lsm` to a static 2D (latitude, longitude) variable, as the
     model and inference_fcn.py both require.
  6. Injects `gfed_region` (static 2D) from --static-path. The model's
     checkpoint loader reads this variable unconditionally.
  7. Adds a zeros `gwis_ba` placeholder (only its shape is used).
  8. Pads the time axis with max(horizons) dummy future periods (NaN inputs).
     inference_fcn.py's target-shift mechanic means dummy values are never
     part of any model input window; they only exist so the future target
     dates are addressable on the time axis.
  9. Writes inference_cube.zarr and a run_inference.sh with one correctly
     parameterized inference_fcn.py command per horizon.

Usage example:
  python build_inference_cube.py \
      --prefix "gs://bsi-wildfire-prediction/output/firecastnet_data_slice_" \
      --static-path gs://bsi-wildfire-prediction/static/seasfire_static.zarr \
      --output-dir ./work

  bash ./work/run_inference.sh   # after placing ckpts + mean/std JSONs

Requires: xarray, zarr, gcsfs, numpy, pandas.
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("build_inference_cube")

INPUT_VARS = [
    "mslp", "tp", "vpd", "sst", "t2m_mean",
    "ssrd", "swvl1", "lst_day", "ndvi", "pop_dens",
]
LSM_VAR = "lsm"
TARGET_VAR = "gwis_ba"
GFED_VAR = "gfed_region"
HORIZONS = [1, 2, 4, 8, 16, 24]


# ---------------------------------------------------------------- calendar --
def period_starts_for_year(year: int):
    """The 46 SeasFire period-start dates for a given year."""
    jan1 = date(year, 1, 1)
    return [jan1 + timedelta(days=8 * k) for k in range(46)]


def next_period_start(d: date) -> date:
    """Successor of a period start under the SeasFire calendar."""
    starts = period_starts_for_year(d.year)
    if d not in starts:
        raise ValueError(
            f"{d} is not a valid SeasFire period start (must be Jan 1 + k*8 days)"
        )
    idx = starts.index(d)
    if idx < 45:
        return starts[idx + 1]
    return date(d.year + 1, 1, 1)  # short Dec period -> Jan 1 reset


def advance_periods(d: date, n: int) -> date:
    for _ in range(n):
        d = next_period_start(d)
    return d


def to_date(ts) -> date:
    return pd.Timestamp(ts).date()


# ------------------------------------------------------------------- build --
def list_period_uris(prefix: str):
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    # match e.g. .../firecastnet_data_slice_20251125_to_20251202.zarr
    pattern = prefix.replace("gs://", "") + "*"
    hits = sorted(fs.glob(pattern))
    uris = ["gs://" + h for h in hits if h.rstrip("/").endswith(".zarr")]
    if not uris:
        raise FileNotFoundError(f"No .zarr stores found matching {pattern}")
    return uris


def open_period(uri: str) -> xr.Dataset:
    try:
        return xr.open_zarr(uri, consolidated=False)
    except Exception:
        return xr.open_zarr(uri)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prefix", help="gs:// prefix of period zarr stores")
    src.add_argument(
        "--uris-file",
        help="text file with one gs:// zarr URI per line (chronological or not; "
        "they get sorted by their time coordinate)",
    )
    p.add_argument(
        "--static-path",
        required=True,
        help="zarr store containing static 2D 'gfed_region' (and optionally a "
        "canonical 'lsm') exported from the original SeasFire cube",
    )
    p.add_argument("--timeseries", type=int, default=24)
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    p.add_argument("--output-dir", default=".")
    p.add_argument(
        "--ckpt-pattern",
        default="firecastnet-cls-ts24-h{h}.ckpt",
        help="checkpoint filename pattern used in the generated run script",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. collect URIs -----------------------------------------------
    if args.prefix:
        uris = list_period_uris(args.prefix)
    else:
        with open(args.uris_file) as f:
            uris = [ln.strip() for ln in f if ln.strip()]
    logger.info(f"Found {len(uris)} period stores")

    # ---- 2. open, sort by time, keep latest N ---------------------------
    datasets = []
    for uri in uris:
        ds = open_period(uri)
        if ds.time.size != 1:
            logger.warning(f"{uri} has {ds.time.size} timesteps (expected 1)")
        datasets.append((to_date(ds.time.values[0]), uri, ds))
    datasets.sort(key=lambda t: t[0])
    datasets = datasets[-args.timeseries:]

    if len(datasets) < args.timeseries:
        logger.error(
            f"Need {args.timeseries} periods, only found {len(datasets)}. Aborting."
        )
        sys.exit(1)

    dates = [d for d, _, _ in datasets]
    logger.info(f"Using periods {dates[0]} .. {dates[-1]}")

    # ---- 3. contiguity check under the SeasFire calendar ----------------
    for a, b in zip(dates, dates[1:]):
        expected = next_period_start(a)
        if b != expected:
            logger.error(
                f"GAP DETECTED: after period {a}, expected {expected} but found {b}. "
                "Refusing to build a cube with a broken timeseries."
            )
            sys.exit(1)
    logger.info("Contiguity check passed.")

    # ---- 4. merge --------------------------------------------------------
    keep = INPUT_VARS + [LSM_VAR]
    slices = []
    for d, uri, ds in datasets:
        missing = [v for v in keep if v not in ds]
        if missing:
            logger.error(f"{uri} is missing variables {missing}. Aborting.")
            sys.exit(1)
        slices.append(ds[keep])
    cube = xr.concat(slices, dim="time")

    for v in cube.data_vars:
        if cube[v].dtype == np.float64:
            logger.info(f"Casting {v} float64 -> float32")
            cube[v] = cube[v].astype(np.float32)

    # ---- 5. lsm -> static 2D --------------------------------------------
    if "time" in cube[LSM_VAR].dims:
        logger.info("Squeezing lsm to static 2D")
        lsm2d = cube[LSM_VAR].isel(time=0, drop=True)
        cube = cube.drop_vars(LSM_VAR)
        cube[LSM_VAR] = lsm2d

    # ---- 6. gfed_region from static export ------------------------------
    logger.info(f"Opening static store: {args.static_path}")
    static = open_period(args.static_path)
    if GFED_VAR not in static:
        logger.error(f"'{GFED_VAR}' not found in {args.static_path}. Aborting.")
        sys.exit(1)
    gfed = static[GFED_VAR]
    if "time" in gfed.dims:
        gfed = gfed.isel(time=0, drop=True)
    # align to our grid coords exactly (tolerates tiny float representation
    # differences by reindexing with nearest within half a cell)
    gfed = gfed.reindex_like(
        cube[LSM_VAR], method="nearest", tolerance=0.05
    )
    if gfed.isnull().any():
        logger.error("gfed_region did not align to the grid. Aborting.")
        sys.exit(1)
    cube[GFED_VAR] = gfed.astype(np.float32)

    # ---- 7 + 8. future padding, then gwis_ba placeholder -----------------
    max_h = max(args.horizons)
    last_real = dates[-1]
    future_dates = [advance_periods(last_real, k) for k in range(1, max_h + 1)]
    logger.info(
        f"Padding {max_h} future periods: {future_dates[0]} .. {future_dates[-1]}"
    )

    pad_time = pd.to_datetime(future_dates)
    pad = xr.Dataset(
        {
            v: xr.DataArray(
                np.full(
                    (max_h, cube.latitude.size, cube.longitude.size),
                    np.nan,
                    dtype=np.float32,
                ),
                dims=("time", "latitude", "longitude"),
            )
            for v in INPUT_VARS
        },
        coords={
            "time": pad_time,
            "latitude": cube.latitude,
            "longitude": cube.longitude,
        },
    )
    cube = xr.concat([cube[INPUT_VARS], pad], dim="time").merge(
        cube[[LSM_VAR, GFED_VAR]]
    )

    cube[TARGET_VAR] = xr.DataArray(
        np.zeros(
            (cube.time.size, cube.latitude.size, cube.longitude.size),
            dtype=np.float32,
        ),
        dims=("time", "latitude", "longitude"),
    )

    # ---- 9. write outputs -------------------------------------------------
    out_zarr = os.path.join(args.output_dir, "inference_cube.zarr")
    logger.info(f"Writing {out_zarr}")
    cube.chunk({"time": cube.time.size}).to_zarr(out_zarr, mode="w")

    run_sh = os.path.join(args.output_dir, "run_inference.sh")
    with open(run_sh, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        f.write(f"# Inputs: {dates[0]} .. {last_real} ({args.timeseries} periods)\n\n")
        for h in args.horizons:
            target = advance_periods(last_real, h)
            ckpt = args.ckpt_pattern.format(h=h)
            f.write(f"# horizon {h}: forecast for the 8-day period starting {target}\n")
            f.write(
                "time python inference_fcn.py "
                f"--cube-path {out_zarr} "
                f"--ckpt-path {ckpt} "
                f"--target-shift {h} --timeseries {args.timeseries} "
                f"--start-time {target} --end-time {target} "
                # published ckpts lack the _lsm_mask buffer; ocean is still
                # masked on the output map by inference_fcn.py regardless
                "--no-lsm-filter "
                f"--output-path predictions_h{h}.zarr\n\n"
            )
    os.chmod(run_sh, 0o755)
    logger.info(f"Wrote {run_sh}")

    print("\n=== SUMMARY ===")
    print(f"Input window : {dates[0]} .. {last_real}")
    for h in args.horizons:
        print(f"h={h:>2}  forecast for period starting {advance_periods(last_real, h)}")
    print(f"Cube         : {out_zarr}")
    print(f"Run script   : {run_sh}")


if __name__ == "__main__":
    main()
