#!/usr/bin/env python3
"""
evaluate_uncertainty.py

Evaluate per-point uncertainty (sigma) produced by run_mvs_pipeline.py
against a ground-truth LiDAR point cloud.

Pipeline:
  1. Load fused_all_cov.npz -> xyz (N,3), gt_cov (N,3,3), sigma (N,)
  2. Load LiDAR LAS/LAZ/PLY -> Mx3 reference point cloud
     IMPORTANT: the LiDAR must already be coarsely aligned to the
     MVS frame. ICP below uses a 2 m gating distance and will not
     converge across large absolute-coordinate offsets (e.g. raw UTM).
  3. Fine-align MVS -> LiDAR with point-to-point ICP (open3d).
  4. For each MVS point, compute distance to nearest LiDAR point.
  5. Report metrics: bounding rate, MAE, RMSE, sigma-distribution stats.

Usage:
  python scripts/evaluate_uncertainty.py \
      --npz   examples/UseGeo/Dataset-1/out/fused_all_cov.npz \
      --lidar /path/to/UseGeo_dataset1_aligned.las \
      [--max_dist 2.0] [--max_iter 200] \
      [--out_npz examples/UseGeo/Dataset-1/out/eval.npz]

Dependencies (in addition to the pipeline's): open3d, scipy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_lidar_xyz(path: Path) -> np.ndarray:
    """Load a LiDAR cloud (.las/.laz/.ply) -> (M, 3) float64 array."""
    suf = path.suffix.lower()
    if suf in (".las", ".laz"):
        import laspy
        las = laspy.read(str(path))
        return np.stack([
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ], axis=1)
    if suf == ".ply":
        from plyfile import PlyData
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        return np.column_stack([
            np.asarray(v["x"]),
            np.asarray(v["y"]),
            np.asarray(v["z"]),
        ]).astype(np.float64)
    sys.exit(f"[error] unsupported LiDAR format: {path.suffix}")


def icp_align(src_xyz: np.ndarray, tgt_xyz: np.ndarray,
              max_dist: float = 2.0, max_iter: int = 200,
              rel_fitness: float = 1e-6, rel_rmse: float = 1e-6):
    """Point-to-point ICP src -> tgt (open3d). Returns (T, fitness, rmse).

    The src cloud is **not** pre-centered. The caller is responsible for
    coarsely aligning src and tgt so that correspondences fall within
    `max_dist` (default 2 m).
    """
    import open3d as o3d

    def _pcd(pts):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return p

    reg = o3d.pipelines.registration
    result = reg.registration_icp(
        _pcd(src_xyz), _pcd(tgt_xyz),
        max_dist,
        np.eye(4),
        reg.TransformationEstimationPointToPoint(with_scaling=False),
        reg.ICPConvergenceCriteria(
            relative_fitness=rel_fitness,
            relative_rmse=rel_rmse,
            max_iteration=max_iter,
        ),
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def per_point_distance(src_xyz: np.ndarray, tgt_xyz: np.ndarray) -> np.ndarray:
    """Nearest-neighbor distance from each src point to the tgt cloud."""
    from scipy.spatial import cKDTree
    tree = cKDTree(tgt_xyz)
    dists, _ = tree.query(src_xyz, k=1)
    return dists.astype(np.float32)


def summarize(sigma: np.ndarray, dist: np.ndarray) -> dict:
    """Bounding rate, MAE, RMSE, and distribution stats."""
    valid = np.isfinite(sigma) & np.isfinite(dist)
    s, d = sigma[valid], dist[valid]
    bounded = (s > d).astype(np.float64)
    return {
        "n_points":       int(len(s)),
        "bounding_rate":  float(bounded.mean()),
        "mae":            float(np.mean(np.abs(s - d))),
        "rmse":           float(np.sqrt(np.mean((s - d) ** 2))),
        "sigma_min":      float(s.min()),
        "sigma_median":   float(np.median(s)),
        "sigma_max":      float(s.max()),
        "dist_min":       float(d.min()),
        "dist_median":    float(np.median(d)),
        "dist_max":       float(d.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True,
                    help="fused_all_cov.npz from run_mvs_pipeline.py")
    ap.add_argument("--lidar", required=True,
                    help=".las/.laz/.ply LiDAR ground-truth cloud, "
                         "already coarsely aligned to the MVS frame")
    ap.add_argument("--max_dist", type=float, default=2.0,
                    help="ICP correspondence gating distance, in scene units (default 2.0)")
    ap.add_argument("--max_iter", type=int, default=200,
                    help="ICP max iterations (default 200)")
    ap.add_argument("--out_npz", default=None,
                    help="Optional path to save per-point eval npz "
                         "(xyz, sigma, distance, bounded). "
                         "Default: <npz_dir>/eval.npz")
    args = ap.parse_args()

    npz_path   = Path(args.npz)
    lidar_path = Path(args.lidar)
    out_npz    = Path(args.out_npz) if args.out_npz else npz_path.parent / "eval.npz"

    print(f"[1/4] Loading MVS cov: {npz_path}")
    d = np.load(npz_path)
    if "xyz" not in d or ("gt_cov" not in d and "cov" not in d):
        sys.exit(f"[error] {npz_path} missing 'xyz' / covariance keys")
    xyz_mvs = d["xyz"]
    cov     = d["gt_cov"] if "gt_cov" in d else d["cov"]
    sigma   = (d["sigma"] if "sigma" in d
               else np.sqrt(cov[:, 0, 0] + cov[:, 1, 1] + cov[:, 2, 2]).astype(np.float32))
    print(f"      {len(xyz_mvs)} MVS points  | sigma median={np.median(sigma):.3f}")

    print(f"[2/4] Loading LiDAR: {lidar_path}")
    xyz_lidar = load_lidar_xyz(lidar_path)
    print(f"      {len(xyz_lidar)} LiDAR points")

    # Quick coarse-alignment sanity check
    mvs_c   = np.median(xyz_mvs,   axis=0)
    lidar_c = np.median(xyz_lidar, axis=0)
    delta   = lidar_c - mvs_c
    if np.linalg.norm(delta) > args.max_dist * 50:
        print(f"[warn] median offset MVS->LiDAR = {delta} (norm "
              f"{np.linalg.norm(delta):.1f}). ICP gate is {args.max_dist} m. "
              f"You likely need to pre-align the LiDAR before running this.")

    print(f"[3/4] Running ICP (point-to-point, max_dist={args.max_dist}, "
          f"max_iter={args.max_iter}) ...")
    T, fitness, inlier_rmse = icp_align(xyz_mvs, xyz_lidar,
                                        max_dist=args.max_dist,
                                        max_iter=args.max_iter)
    R, t = T[:3, :3], T[:3, 3]
    aligned = xyz_mvs @ R.T + t
    print(f"      fitness={fitness:.3f}  inlier_rmse={inlier_rmse:.4f}")

    print(f"[4/4] Computing per-point distance + metrics ...")
    dist    = per_point_distance(aligned, xyz_lidar)
    metrics = summarize(sigma, dist)

    print()
    print(f"  N points       : {metrics['n_points']}")
    print(f"  bounding rate  : {metrics['bounding_rate'] * 100:6.2f} %   (sigma > distance)")
    print(f"  MAE  |sigma-d| : {metrics['mae']:.4f} m")
    print(f"  RMSE           : {metrics['rmse']:.4f} m")
    print(f"  sigma  min/med/max : {metrics['sigma_min']:.4f} / "
          f"{metrics['sigma_median']:.4f} / {metrics['sigma_max']:.4f} m")
    print(f"  dist   min/med/max : {metrics['dist_min']:.4f} / "
          f"{metrics['dist_median']:.4f} / {metrics['dist_max']:.4f} m")

    bounded = (sigma > dist).astype(np.uint8)
    np.savez_compressed(out_npz,
                        xyz=aligned.astype(np.float32),
                        sigma=sigma,
                        distance=dist,
                        bounded=bounded,
                        icp_T=T,
                        icp_fitness=fitness,
                        icp_inlier_rmse=inlier_rmse,
                        **{f"metric_{k}": v for k, v in metrics.items()})
    print(f"\n  -> wrote {out_npz}")


if __name__ == "__main__":
    main()
