#!/usr/bin/env python3
"""
Synthetic end-to-end smoke test for FireCastNet.

Validates, WITHOUT any real data:
  1. checkpoint deserialization under the installed torch/lightning versions
  2. DGL multi-mesh graph construction under the installed dgl version
  3. a full forward pass (predict_step) on random input
  4. output shape/range sanity

Usage (from the repo root, venv active):
  wget "https://huggingface.co/datasets/d-michail/firecastnet-artifacts/resolve/main/model_ckpts/firecastnet-cls-ts24-h8.ckpt"
  python ops/smoke_test_model.py --ckpt-path firecastnet-cls-ts24-h8.ckpt

Notes:
  - Needs ~6-8 GB free RAM (the input tensor alone is ~1.1 GB).
  - Runs on CPU; a laptop is fine. Expect the forward pass to take
    anywhere from tens of seconds to several minutes on CPU.
"""

import argparse
import logging
import os
import sys
import tempfile
import time

# Make the repo root importable regardless of where this script is run from,
# so `from seasfire...` works even though this file lives in ops/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")

N_VARS = 11        # 10 dynamic inputs + lsm
TIMESERIES = 24
H, W = 720, 1440


def make_fake_cube(path: str):
    """Minimal cube containing only what FireCastNetLit reads at load time."""
    lat = np.arange(89.875, -90, -0.25)
    lon = np.arange(-179.875, 180, 0.25)
    assert lat.size == H and lon.size == W

    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            # values in [0,1] so the 0.1 lsm threshold produces a mixed mask
            "lsm": (("latitude", "longitude"),
                    rng.random((H, W)).astype(np.float32)),
            "gfed_region": (("latitude", "longitude"),
                            np.zeros((H, W), dtype=np.float32)),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_zarr(path, mode="w")
    logger.info(f"Fake cube written to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--timeseries", type=int, default=TIMESERIES)
    args = p.parse_args()

    # import here so a broken env fails with a clear message
    from seasfire.firecastnet_lit import FireCastNetLit

    with tempfile.TemporaryDirectory() as tmp:
        cube_path = os.path.join(tmp, "fake_cube.zarr")
        make_fake_cube(cube_path)

        logger.info(f"Loading checkpoint {args.ckpt_path} (this builds the "
                    "DGL multi-mesh graph; may take a few minutes)...")
        t0 = time.time()
        model = FireCastNetLit.load_from_checkpoint(
            args.ckpt_path,
            cube_path=cube_path,
            map_location="cpu",
            # The published checkpoints predate the _lsm_mask persistent
            # buffer, so strict loading fails if the LSM filter is enabled.
            # Disabling it here is safe: the mask is not a learned weight,
            # and the production pipeline masks the ocean on the OUTPUT map
            # anyway (inference_fcn.py does this unconditionally).
            lsm_filter_enable=False,
        )
        model.eval()
        model.dglTo(model.device)
        t_load = time.time() - t0
        logger.info(f"CHECKPOINT LOAD OK ({t_load:.1f}s)")

        x = torch.randn(1, N_VARS, args.timeseries, H, W, dtype=torch.float32)
        logger.info(f"Input tensor: {tuple(x.shape)} "
                    f"({x.element_size() * x.nelement() / 1e9:.2f} GB)")

        t0 = time.time()
        with torch.no_grad():
            preds = model.predict_step({"x": x})
        t_fwd = time.time() - t0

        preds = preds.cpu().numpy().squeeze()
        logger.info(f"FORWARD PASS OK ({t_fwd:.1f}s)")
        logger.info(f"Output shape: {preds.shape}  (expected ({H}, {W}))")
        logger.info(f"Output range: [{np.nanmin(preds):.4f}, "
                    f"{np.nanmax(preds):.4f}]  (expected within [0, 1])")

        ok = preds.shape == (H, W) and np.nanmin(preds) >= 0 and np.nanmax(preds) <= 1
        print("\n=== SMOKE TEST", "PASSED ===" if ok else "FAILED ===")
        print(f"checkpoint load : {t_load:.1f}s")
        print(f"forward pass    : {t_fwd:.1f}s   (~ per-horizon inference cost)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
