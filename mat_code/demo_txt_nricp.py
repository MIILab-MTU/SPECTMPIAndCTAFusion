#!/usr/bin/env python3
"""
demo_txt_nricp.py
-----------------
Usage:
  python demo_txt_nricp.py --src SRC.txt --tgt TGT.txt --out out_prefix \
      --grid 8 8 8 --levels 2 --iters 8 --lam 0.01 --robust 0.02 --voxel 0.0 --skiprows 0

- SRC.txt / TGT.txt: plain text point clouds with at least 3 numeric columns (x y z ...). 
  The script will take the first three columns as XYZ.
- Outputs:
  - {out}_deformed.xyz        : deformed source (x y z)
  - {out}_metrics.txt         : simple metrics report
  - {out}_disp_grid.npz       : displacement grid & grid axes (for reuse)
"""

import argparse, os, sys
import numpy as np
from mat_code.pwtricubic_nricp import PWTricubicNRICP
from scipy.spatial import cKDTree

def load_txt_points(path, skiprows=0):
    # Auto delimiter: numpy.loadtxt handles whitespace and tabs; for CSV it also works if sep is commas.
    pts = np.loadtxt(path, dtype=float, ndmin=2, skiprows=skiprows)
    if pts.shape[1] < 3:
        raise ValueError(f"{path} must have >=3 columns (x y z ...)")
    return pts[:, :3]

def write_xyz(path, pts):
    np.savetxt(path, pts, fmt="%.6f")

def main(i, cloud_dir):


    ap = argparse.ArgumentParser()
    ap.add_argument("--tgt", default=rf'preprocess/name/{i}/ijkcta.txt', help="source TXT path")
    ap.add_argument("--src", default=rf'{cloud_dir}/{i}/cu.txt', help="target TXT path")
    ap.add_argument("--out", default=rf"{cloud_dir}/{i}/ffd_result", help="output prefix (no extension)")
    ap.add_argument("--grid", nargs=3, type=int, default=[8,8,8], help="control grid size nx ny nz")
    ap.add_argument("--levels", type=int, default=2, help="multi-scale levels (coarse->fine)")
    ap.add_argument("--iters", type=int, default=8, help="ICP iterations per level")
    ap.add_argument("--lam", type=float, default=1e-5, help="Laplacian regularization strength")
    ap.add_argument("--robust", type=float, default=0.5, help="Huber tau (0 for L2)")
    ap.add_argument("--voxel", type=float, default=0.5, help="optional random downsample placeholder (set 0)")
    ap.add_argument("--skiprows", type=int, default=0, help="lines to skip at file head")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    args = ap.parse_args()

    X = load_txt_points(args.src, skiprows=args.skiprows)
    Y = load_txt_points(args.tgt, skiprows=args.skiprows)

    print(f"Loaded: src {X.shape} | tgt {Y.shape}")
    reg = PWTricubicNRICP(
        grid_size=tuple(args.grid),
        levels=args.levels,
        iters_per_level=args.iters,
        lam=args.lam,
        robust_tau=args.robust,
        seed=args.seed,
        verbose=True
    )
    out = reg.fit(X, Y)
    Xd = out["deformed"]

    # Save outputs
    out_xyz = f"{args.out}.txt"
    write_xyz(out_xyz, Xd)
    # out_xyz = f"{args.out}_deformed.xyz"
    # write_xyz(out_xyz, Xd)
    np.savez(f"{args.out}_disp_grid.npz", disp_grid=out["disp_grid"], grid_axes=out["grid_axes"])

    # Simple nearest-neighbor RMSE
    kdt = cKDTree(Y)
    d, idx = kdt.query(Xd, k=1, n_jobs=-1)
    rmse = float(np.sqrt((d**2).mean()))
    with open(f"{args.out}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Points: src={len(X)}, tgt={len(Y)}\n")
        f.write(f"Grid: {tuple(args.grid)}, levels={args.levels}, iters/level={args.iters}\n")
        f.write(f"lam={args.lam}, robust_tau={args.robust}\n")
        f.write(f"Final NN-RMSE: {rmse:.6f}\n")
    print("Saved:")
    print(" ", out_xyz)
    print(" ", f"{args.out}_disp_grid.npz")
    print(" ", f"{args.out}_metrics.txt")


