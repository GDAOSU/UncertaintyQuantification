#!/usr/bin/env python3
"""
run_mvs_pipeline.py

Full MVS pipeline implemented entirely in Python:
  1. Read COLMAP sparse model
  2. Select stereo pairs from co-visibility graph
  3. Run OpenCV SGBM stereo → disparity + SGM energy proxy
  4. Fuse depth maps into per-image XYZ grids
  5. Predict per-pixel measurement error  (energy-calibrated reprojection std)
  6. Propagate measurement error to 3-D point covariance (Gauss-Markov)
  7. Write per-pixel covariance maps (.tif) and summary .txt

Dependencies:
    pip install numpy scipy opencv-python-headless laspy

Usage:
    python run_mvs_pipeline.py \\
        --scene   path/to/colmap/sparse \\
        --images  path/to/images        \\
        --out     path/to/output        \\
        [--max_neighbors 6]             \\
        [--disp_range 128]              \\
        [--block_size 9]                \\
        [--multi_rays 6]                \\
        [--num_std 1]                   \\
        [--sample_interval 3]           \\
        [--cov_txt output_cov.txt]
"""

import argparse
import gc
import logging
import struct
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("mvs_pipeline")


def _mem_gb():
    """Return current RSS in GB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1048576  # kB → GB
    except Exception:
        return -1

# Import projection and Jacobian functions directly from
# compute_sensor_error_prop.py so we use the exact same implementation.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_sensor_error_prop import (
    aa_from_qvec,
    PRIOR_INV,
)
from rectify import compute_rectifying_rotation

# ============================================================================
# COLMAP binary readers
# ============================================================================

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),  3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid = struct.unpack("<I", f.read(4))[0]
            mid = struct.unpack("<I", f.read(4))[0]
            w   = struct.unpack("<Q", f.read(8))[0]
            h   = struct.unpack("<Q", f.read(8))[0]
            name, np_ = CAMERA_MODELS[mid]
            params = list(struct.unpack(f"<{np_}d", f.read(8 * np_)))
            cameras[cid] = dict(model=name, width=int(w), height=int(h), params=params)
    return cameras


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid  = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cid  = struct.unpack("<I", f.read(4))[0]
            bname = []
            while True:
                c = f.read(1)
                if c == b"\x00": break
                bname.append(c)
            name = b"".join(bname).decode()
            npts = struct.unpack("<Q", f.read(8))[0]
            # Each 2D point record is interleaved: x(f64), y(f64), point3D_id(i64) = 24 bytes
            pts2d_raw = f.read(24 * npts)
            pt3d_ids = set()
            pid_to_xy = {}
            for i in range(npts):
                x = struct.unpack_from("<d", pts2d_raw, 24 * i)[0]
                y = struct.unpack_from("<d", pts2d_raw, 24 * i + 8)[0]
                pid = struct.unpack_from("<q", pts2d_raw, 24 * i + 16)[0]
                if pid != -1:
                    pt3d_ids.add(pid)
                    pid_to_xy[pid] = (x, y)
            images[iid] = dict(
                qvec=qvec, tvec=tvec, cam_id=cid, name=name,
                opk=qvec_to_opk(qvec),
                pt3d_ids=pt3d_ids,
                pid_to_xy=pid_to_xy,
            )
    return images


def read_points3D_bin(path, min_track_len=3, z_outlier_percent=1.0):
    """Read COLMAP points3D.bin → {pid: xyz}.

    Two filters are applied in order:
      1. Track-length filter: drop points with fewer than `min_track_len`
         observing images. Default 3 follows the photogrammetry convention
         that a 3-view consensus is the minimum for a reliable point.
      2. Z-percentile clip: keep only points whose world Z lies within
         [p_lo, p_hi] where p_lo = z_outlier_percent and
         p_hi = 100 - z_outlier_percent. Removes the residual floating
         points and below-ground outliers that survive filter 1.

    Pass `z_outlier_percent=0` to disable filter 2.
    """
    raw = []  # (pid, xyz, track_len) for all points
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            f.read(3)  # rgb
            f.read(8)  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # (image_id, point2D_idx) pairs
            raw.append((pid, xyz, track_len))

    n_total = len(raw)
    after_track = [t for t in raw if t[2] >= min_track_len]
    n_after_track = len(after_track)

    if z_outlier_percent > 0 and n_after_track > 0:
        zs = np.array([t[1][2] for t in after_track])
        z_lo = float(np.percentile(zs, z_outlier_percent))
        z_hi = float(np.percentile(zs, 100.0 - z_outlier_percent))
        kept = [t for t in after_track if z_lo <= t[1][2] <= z_hi]
    else:
        z_lo = float("-inf")
        z_hi = float("inf")
        kept = after_track

    points = {pid: xyz for pid, xyz, _ in kept}
    log.info(f"      points3D: {n_total} → {n_after_track} "
             f"(track_len >= {min_track_len}) → {len(points)} "
             f"(z in [{z_lo:.1f}, {z_hi:.1f}], drop top/bot {z_outlier_percent:g}%)")
    return points


# ============================================================================
# Geometry helpers
# ============================================================================

def qvec_to_R(qvec):
    """COLMAP (qw,qx,qy,qz) → 3×3 world-to-cam rotation."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,   1-2*qx**2-2*qz**2,  2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,    1-2*qx**2-2*qy**2],
    ])


def qvec_to_opk(qvec):
    """COLMAP (qw,qx,qy,qz) → photogrammetric (omega, phi, kappa) in radians."""
    R = qvec_to_R(qvec)
    phi   = np.arcsin(-R[2, 0])
    omega = np.arctan2(R[2, 1], -R[2, 2])
    kappa = np.arctan2(R[1, 0], R[0, 0])
    return omega, phi, kappa


def opk_to_R(omega, phi, kappa):
    """Photogrammetric (omega, phi, kappa) → 3×3 world-to-cam rotation.

    Matches the C++ GetR() convention:
      R = Rz(kappa) · Ry(phi) · Rx(omega) · Rx(π)
    so that R(0,0,0) = diag(1, -1, -1).
    """
    co, so = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi),   np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)
    return np.array([
        [ ck*cp,           co*sk + ck*so*sp,  sk*so - ck*co*sp],
        [ cp*sk,           sk*so*sp - ck*co, -ck*so - co*sk*sp],
        [-sp,              cp*so,            -co*cp            ],
    ])


def get_K_dist(cam):
    """Return (K 3×3 float64, dist 4-vec) from a COLMAP camera dict."""
    m, p = cam["model"], cam["params"]
    if   m == "SIMPLE_PINHOLE":
        K = np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1.]])
        d = np.zeros(4)
    elif m == "PINHOLE":
        K = np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1.]])
        d = np.zeros(4)
    elif m == "SIMPLE_RADIAL":
        K = np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1.]])
        d = np.array([p[3],0.,0.,0.])
    elif m == "RADIAL":
        K = np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1.]])
        d = np.array([p[3],p[4],0.,0.])
    elif m == "OPENCV":
        K = np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1.]])
        d = np.array([p[4],p[5],p[6],p[7]])
    else:
        raise ValueError(f"Unsupported model: {m}")
    return K.astype(np.float64), d.astype(np.float64)


def cam_params_array(cam):
    """Return [fx,fy,cx,cy,k1,k2,p1,p2] matching compute_sensor_error_prop layout."""
    K, d = get_K_dist(cam)
    return np.array([K[0,0], K[1,1], K[0,2], K[1,2], d[0], d[1], d[2], d[3]])



# ============================================================================
# Co-visibility graph
# ============================================================================

def build_covis_graph(images, max_neighbors, points3D=None,
                      cameras=None, min_feature_pts=3,
                      min_bl_opt_angle=60.0,
                      optimal_angle_deg=12.0):
    """Build co-visibility graph with OpenMVS-style quality scoring.

    For each reference image, iterates its visible 3D points and scores every
    neighbor that also sees that point using:
      1. Triangulation angle weight (asymmetric Gaussian, peaked at optimal_angle)
      2. Scale weight (pixel footprint ratio, penalty beyond 1.6x)
      3. Spatial coverage (how well shared points spread across the reference image)
    Then filters by baseline-optical angle (for rectification feasibility).

    Reference: OpenMVS Scene::SelectNeighborViews
    """
    # Precompute camera centers, optical axes, and focal lengths
    cam_center = {}
    cam_opt = {}
    cam_focal = {}
    for iid, img in images.items():
        R_wc = opk_to_R(*img["opk"])
        C = -(R_wc.T @ np.array(img["tvec"]))
        cam_center[iid] = C
        cam_opt[iid] = R_wc.T @ np.array([0, 0, 1])
        if cameras is not None:
            cam = cameras[img["cam_id"]]
            cam_focal[iid] = cam["params"][0]

    # Build reverse map: pid → list of image ids
    pt_to_imgs = defaultdict(list)
    for iid, img in images.items():
        for pid in img["pt3d_ids"]:
            pt_to_imgs[pid].append(iid)

    use_scoring = (points3D is not None and len(points3D) > 0)

    # OpenMVS-style angle scoring parameters
    opt_angle = np.radians(optimal_angle_deg)
    sigma_small = -1.0 / (2.0 * (opt_angle * 0.38) ** 2)
    sigma_large = -1.0 / (2.0 * (opt_angle * 0.7) ** 2)

    # Baseline-optical angle filter (for rectification feasibility)
    def _bl_opt_ok(iid1, iid2):
        bl = cam_center[iid2] - cam_center[iid1]
        bl_len = np.linalg.norm(bl)
        if bl_len < 1e-6:
            return False
        bl_dir = bl / bl_len
        avg_opt = (cam_opt[iid1] + cam_opt[iid2]) / 2
        avg_opt /= np.linalg.norm(avg_opt)
        angle = np.degrees(np.arccos(
            np.clip(abs(np.dot(bl_dir, avg_opt)), 0, 1)))
        return angle >= min_bl_opt_angle

    COVERAGE_GRID = 4  # 4x4 grid for spatial coverage (OpenMVS uses ~16 bins)

    graph = {}
    n_skipped = 0

    for ref_iid, ref_img in images.items():
        ref_C = cam_center[ref_iid]
        ref_cam = cameras[ref_img["cam_id"]] if cameras else None
        ref_W = ref_cam["width"] if ref_cam else 1
        ref_H = ref_cam["height"] if ref_cam else 1
        pid_to_xy = ref_img.get("pid_to_xy", {})

        # Accumulate score per neighbor, and track which grid cells are covered
        nb_score = defaultdict(float)
        nb_count = defaultdict(int)
        nb_cells = defaultdict(set)  # neighbor → set of (gx, gy) grid cells

        for pid in ref_img["pt3d_ids"]:
            observers = pt_to_imgs.get(pid, [])
            # Track length filter: need >= 3 observers (matches read_points3D_bin)
            if len(observers) < 3:
                continue

            if use_scoring and pid in points3D:
                xyz = points3D[pid]
                V1 = ref_C - xyz
                d1 = np.linalg.norm(V1)
                if d1 < 1e-6:
                    continue

                for nb_iid in observers:
                    if nb_iid == ref_iid:
                        continue
                    V2 = cam_center[nb_iid] - xyz
                    d2 = np.linalg.norm(V2)
                    if d2 < 1e-6:
                        continue

                    # Triangulation angle weight
                    cos_a = np.clip(np.dot(V1, V2) / (d1 * d2), -1.0, 1.0)
                    angle = np.arccos(cos_a)
                    sigma = sigma_small if angle < opt_angle else sigma_large
                    w_angle = max(np.exp((angle - opt_angle) ** 2 * sigma), 0.1)

                    # Scale weight
                    w_scale = 1.0
                    if cam_focal:
                        fp1 = d1 / cam_focal.get(ref_iid, 1.0)
                        fp2 = d2 / cam_focal.get(nb_iid, 1.0)
                        ratio = fp1 / (fp2 + 1e-12)
                        if ratio > 1.6:
                            w_scale = (1.6 / ratio) ** 2
                        elif ratio < 1.0:
                            w_scale = ratio ** 2

                    nb_score[nb_iid] += w_angle * w_scale
                    nb_count[nb_iid] += 1

                    # Track spatial coverage
                    if pid in pid_to_xy:
                        px, py = pid_to_xy[pid]
                        gx = min(int(px / ref_W * COVERAGE_GRID), COVERAGE_GRID - 1)
                        gy = min(int(py / ref_H * COVERAGE_GRID), COVERAGE_GRID - 1)
                        nb_cells[nb_iid].add((gx, gy))
            else:
                # Fallback: simple count
                for nb_iid in observers:
                    if nb_iid == ref_iid:
                        continue
                    nb_score[nb_iid] += 1.0
                    nb_count[nb_iid] += 1

        # Apply spatial coverage multiplier (OpenMVS: area = covered_cells / total_cells)
        total_cells = COVERAGE_GRID * COVERAGE_GRID
        final_scores = {}
        final_coverage = {}
        for nb_iid, score in nb_score.items():
            if nb_count[nb_iid] < min_feature_pts:
                continue
            coverage = max(len(nb_cells.get(nb_iid, set())) / total_cells, 0.01)
            final_scores[nb_iid] = score * coverage
            final_coverage[nb_iid] = coverage

        # Sort by coverage first, then by angle×scale score to break ties
        nbrs = sorted(final_scores.items(),
                       key=lambda x: (-final_coverage[x[0]], -x[1]))
        filtered = []
        for n, s in nbrs:
            if _bl_opt_ok(ref_iid, n):
                filtered.append(n)
                if max_neighbors > 0 and len(filtered) >= max_neighbors:
                    break
            else:
                n_skipped += 1
        graph[ref_iid] = filtered

    if n_skipped:
        log.info(f"      (filtered {n_skipped} neighbor slots with "
              f"baseline-optical angle < {min_bl_opt_angle}°)")
    return graph


def build_sift_match_graph(images, image_dir, max_neighbors=10,
                           coarse_long_side=1024):
    """Build neighbor graph by exhaustive SIFT matching on coarse images.

    1. Downsample all images so the long side ≤ coarse_long_side.
    2. Extract SIFT features on each coarse image.
    3. GPU brute-force match all N*(N-1)/2 pairs, Lowe's ratio test.
    4. For each image, rank neighbors by match count, take top max_neighbors.

    Returns  graph : dict  iid → [neighbor_iid, …]
    """
    import torch
    from collections import defaultdict

    iid_list = sorted(images.keys())

    # --- 1. Load & downsample, extract SIFT ---
    sift = cv2.SIFT_create()
    desc_dict = {}  # iid → desc (np float32, N×128)

    log.info(f"      SIFT graph: extracting features on {len(iid_list)} images "
             f"(coarse_long_side={coarse_long_side}) ...")

    for iid in iid_list:
        name = images[iid]["name"]
        stem = Path(name).stem
        img = None
        for ext in [Path(name).suffix, ".jpg", ".JPG", ".png", ".PNG", ".tif", ".tiff"]:
            for p in [Path(image_dir)/name, Path(image_dir)/(stem+ext)]:
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        break
            if img is not None:
                break
        if img is None:
            log.warning(f"      SIFT graph: could not load {name}")
            continue

        # Downsample
        h, w = img.shape[:2]
        scale = min(1.0, coarse_long_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        kp, desc = sift.detectAndCompute(img, None)
        if desc is not None and len(kp) > 0:
            desc_dict[iid] = desc.astype(np.float32)

    log.info(f"      SIFT graph: features extracted for {len(desc_dict)}/{len(iid_list)} images")

    # --- 2. GPU brute-force matching for all pairs ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"      SIFT graph: matching on {device}")

    # Pre-load all descriptors to GPU as L2-normalized tensors
    desc_gpu = {}
    for iid, desc in desc_dict.items():
        t = torch.from_numpy(desc).to(device)
        # L2 normalize for cosine-distance matching via matmul
        t = t / (t.norm(dim=1, keepdim=True) + 1e-8)
        desc_gpu[iid] = t

    valid_iids = [iid for iid in iid_list if iid in desc_gpu]
    n_pairs = len(valid_iids) * (len(valid_iids) - 1) // 2
    log.info(f"      SIFT graph: {n_pairs} exhaustive pairs")

    match_counts = defaultdict(lambda: defaultdict(int))

    n_done = 0
    for i in range(len(valid_iids)):
        iid1 = valid_iids[i]
        d1 = desc_gpu[iid1]  # (N1, 128)

        for j in range(i + 1, len(valid_iids)):
            iid2 = valid_iids[j]
            d2 = desc_gpu[iid2]  # (N2, 128)

            # Cosine similarity → (N1, N2); higher = more similar
            sim = d1 @ d2.T

            # Top-2 matches per row for Lowe's ratio test
            top2, _ = sim.topk(2, dim=1)
            # Convert cosine similarity to distance: dist = 1 - sim
            # Ratio test: dist_best / dist_second < 0.7
            # Equivalent: (1 - sim_best) / (1 - sim_second) < 0.7
            # Equivalent: sim_best > 1 - 0.7 * (1 - sim_second)
            best = top2[:, 0]
            second = top2[:, 1]
            good = (1.0 - best) < 0.7 * (1.0 - second)
            count = good.sum().item()

            if count > 0:
                match_counts[iid1][iid2] = count
                match_counts[iid2][iid1] = count

            n_done += 1
            if n_done % 5000 == 0:
                log.info(f"      SIFT graph: matched {n_done}/{n_pairs} pairs ...")

    log.info(f"      SIFT graph: matching done ({n_done} pairs)")

    # Free GPU memory
    del desc_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3. Build graph: top-N by match count, with minimum threshold ---
    min_matches = 50
    graph = {}
    for iid in iid_list:
        nbs = match_counts.get(iid, {})
        ranked = sorted(nbs.items(), key=lambda x: -x[1])
        graph[iid] = [n for n, c in ranked[:max_neighbors] if c >= min_matches]

    return graph


# ============================================================================
# Stereo matching
# ============================================================================

def load_image(image_dir, name, gray=True):
    stem = Path(name).stem
    for ext in [Path(name).suffix, ".jpg", ".JPG", ".png", ".PNG", ".tif", ".tiff"]:
        for p in [Path(image_dir)/name, Path(image_dir)/(stem+ext)]:
            if p.exists():
                flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
                img = cv2.imread(str(p), flag)
                if img is not None:
                    return img
    raise FileNotFoundError(f"{name} not found in {image_dir}")


def estimate_disp_range_from_sparse(img1_pts, img2_pts, points3D,
                                    R_rect, C1, C2, K_rect1, K_rect2,
                                    margin=0.1):
    """Estimate disparity search range from shared sparse 3D points.

    Projects shared sparse points into both rectified cameras and computes
    the actual disparity (u2_rect - u1_rect) directly, matching the _run_sgbm
    output convention. No sign assumptions needed.

    Returns (disp_center, disp_half_range) or None if not enough points.
    """
    shared = img1_pts & img2_pts
    if not shared or points3D is None:
        return None

    xyz_list = [points3D[pid] for pid in shared if pid in points3D]
    if len(xyz_list) < 10:
        return None

    xyz = np.array(xyz_list)  # (N, 3)

    # Project into rectified camera 1
    t1 = -R_rect @ C1
    pc1 = (R_rect @ xyz.T + t1.reshape(3, 1)).T  # (N, 3)

    # Project into rectified camera 2
    t2 = -R_rect @ C2
    pc2 = (R_rect @ xyz.T + t2.reshape(3, 1)).T  # (N, 3)

    # Keep only points in front of both cameras
    valid = (pc1[:, 2] > 0) & (pc2[:, 2] > 0)
    if valid.sum() < 10:
        return None

    # Rectified u-coordinates
    u1 = K_rect1[0, 0] * pc1[valid, 0] / pc1[valid, 2] + K_rect1[0, 2]
    u2 = K_rect2[0, 0] * pc2[valid, 0] / pc2[valid, 2] + K_rect2[0, 2]

    # Disparity in _run_sgbm convention: d = -(OpenCV_disp) = -(u1 - u2) = u2 - u1
    disparities = u2 - u1

    # Robust range using percentiles + margin
    d_lo = np.percentile(disparities, 2)
    d_hi = np.percentile(disparities, 98)
    d_spread = d_hi - d_lo
    d_lo -= d_spread * margin
    d_hi += d_spread * margin

    disp_center = (d_lo + d_hi) / 2.0
    disp_half_range = (d_hi - d_lo) / 2.0
    disp_half_range = max(disp_half_range, 32.0)  # minimum for safety

    return disp_center, disp_half_range


def compute_energy_proxy(img1_rect, img2_rect, disp_map, block_size=9):
    """
    SAD at the selected disparity as a proxy for SGM energy.
    disp_map uses ErrorProp convention: disp = u2 - u1.
    Returns uint16 energy (high = uncertain match).
    """
    h, w = img1_rect.shape[:2]
    i1 = img1_rect.astype(np.float32)
    i2 = img2_rect.astype(np.float32)
    dv = np.where(np.isnan(disp_map), 0., disp_map).astype(np.float32)
    map_x = np.arange(w, dtype=np.float32)[None,:] + dv
    map_y = np.arange(h, dtype=np.float32)[:,None] * np.ones((1, w), dtype=np.float32)
    i2w = cv2.remap(i2, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    diff = np.abs(i1 - np.nan_to_num(i2w, nan=255.))
    ef   = cv2.boxFilter(diff, -1, (block_size, block_size), normalize=True)
    ef[np.isnan(disp_map)] = ef.max() if ef.max() > 0 else 1.
    mx = ef.max()
    if mx > 0:
        ef = ef / mx * 65000.
    return ef.astype(np.uint16)


def _run_sgbm(img1, img2, cv_min_d, num_d, block_size, P1, P2,
              lr_threshold, speckle_size):
    """Run one SGBM forward+backward pass. Returns (d1, d2) in ErrorProp convention."""
    sgbm_fwd = cv2.StereoSGBM_create(
        minDisparity=cv_min_d, numDisparities=num_d, blockSize=block_size,
        P1=P1, P2=P2, disp12MaxDiff=lr_threshold, uniquenessRatio=10,
        speckleWindowSize=speckle_size, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    raw1 = sgbm_fwd.compute(img1, img2)
    del sgbm_fwd  # free SGBM internal buffers immediately

    # Reverse pass: true SGBM disparity has opposite sign to forward.
    # Forward searches [cv_min_d, cv_min_d + num_d).
    # Reverse needs  [-(cv_min_d + num_d), -cv_min_d).
    rev_min_d = -(cv_min_d + num_d)
    sgbm_rev = cv2.StereoSGBM_create(
        minDisparity=rev_min_d, numDisparities=num_d, blockSize=block_size,
        P1=P1, P2=P2, disp12MaxDiff=lr_threshold, uniquenessRatio=10,
        speckleWindowSize=speckle_size, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    raw2 = sgbm_rev.compute(img2, img1)
    del sgbm_rev  # free SGBM internal buffers immediately

    bad1 = raw1 == (cv_min_d - 1) * 16
    bad2 = raw2 == (rev_min_d - 1) * 16
    d1 = -(raw1.astype(np.float32) / 16.)
    d2 = -(raw2.astype(np.float32) / 16.)
    d1[bad1] = np.nan
    d2[bad2] = np.nan
    del bad1, bad2

    # Filter search-range boundary hits (± 3 integer disparities):
    # disparity pinned near the search window limits is unreliable,
    # often caused by SGM cost aggregation paths traversing the black
    # rectification border.  The 3-px margin catches the SGM
    # aggregation tail beyond the exact boundary value.
    _margin16 = 3 * 16
    fwd_max16 = (cv_min_d + num_d - 1) * 16
    fwd_min16 = cv_min_d * 16
    d1[(raw1 >= fwd_max16 - _margin16) | (raw1 <= fwd_min16 + _margin16)] = np.nan

    rev_max16 = (rev_min_d + num_d - 1) * 16
    rev_min16 = rev_min_d * 16
    d2[(raw2 >= rev_max16 - _margin16) | (raw2 <= rev_min16 + _margin16)] = np.nan
    del raw1, raw2  # free raw int16 disparity maps

    # ── External left-right consistency check ──
    # For each valid d1 pixel (v, u), the match in img2 is at (v, u + d1).
    # d2 at that location should satisfy d1[v,u] + d2[v, u+d1] ≈ 0.
    # Reject pixels where the round-trip error exceeds lr_threshold.
    h, w = d1.shape
    def _lr_check(d_fwd, d_rev, thresh):
        valid = ~np.isnan(d_fwd)
        if not valid.any():
            return
        vv, uu = np.where(valid)
        mu = np.round(uu + d_fwd[vv, uu]).astype(np.intp)
        in_bounds = (mu >= 0) & (mu < w)
        # Out-of-bounds matches are invalid
        d_fwd[vv[~in_bounds], uu[~in_bounds]] = np.nan
        vv, uu, mu = vv[in_bounds], uu[in_bounds], mu[in_bounds]
        d_rev_at_match = d_rev[vv, mu]
        # Round-trip: d_fwd + d_rev should be ~0
        rt_err = np.abs(d_fwd[vv, uu] + d_rev_at_match)
        bad = np.isnan(d_rev_at_match) | (rt_err > thresh)
        d_fwd[vv[bad], uu[bad]] = np.nan

    _lr_check(d1, d2, lr_threshold)
    _lr_check(d2, d1, lr_threshold)

    return d1, d2


def run_sgm_pair(img1_gray, img2_gray, disp_range=128, block_size=9,
                  speckle_size=120, lr_threshold=1, coarse_long_side=700,
                  sparse_disp_hint=None):
    """
    Coarse-to-fine SGBM using a 3-level pyramid to auto-discover the
    disparity range, then refine at full resolution.

    If sparse_disp_hint=(center, half_range) is provided (from sparse 3D
    points), the coarse level searches around the hint center with a range
    derived from max(hint_range, disp_range). This makes the coarse level
    more robust for wide-baseline pairs while still allowing SGM to correct
    inaccuracies in the sparse estimate.

    Returns (disp1, disp2) in ErrorProp convention: disp = u2 - u1.
    """
    import time as _time
    P1 = 8  * block_size**2
    P2 = 32 * block_size**2
    h, w = img1_gray.shape[:2]

    # ── Build image pyramid (coarse → fine, excluding full res) ──
    # The loop discovers disparity range; full-res pass runs separately after.
    long_side = max(h, w)
    scales = []
    s = min(1.0, coarse_long_side / long_side)
    while s < 1.0:
        scales.append(s)
        s *= 2.0

    cv_min_d = 0
    num_d = None
    median_ep = None

    # Use sparse hint to seed the coarse level if available
    if sparse_disp_hint is not None:
        hint_center, hint_half = sparse_disp_hint
        median_ep = hint_center  # full-res disparity center

    for lvl, scale in enumerate(scales):
        sh = max(1, int(h * scale))
        sw = max(1, int(w * scale))
        _t0 = _time.time()

        if scale < 1.0:
            img1_s = cv2.resize(img1_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            img2_s = cv2.resize(img2_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            img1_s, img2_s = img1_gray, img2_gray

        if lvl == 0:
            bs = max(5, block_size // 2 | 1)
            lP1, lP2 = 8 * bs**2, 32 * bs**2

            if median_ep is not None:
                # Sparse hint available: search around the hint center
                # Use wider range to allow SGM to correct the hint
                hint_half_scaled = (sparse_disp_hint[1] if sparse_disp_hint
                                    else disp_range / 2)
                lvl_range = max(int(hint_half_scaled * scale * 2.5), disp_range)
                lvl_num_d = lvl_range - (lvl_range % 16)
                lvl_num_d = max(lvl_num_d, 16)
                ep_center = median_ep * scale
                ep_min = ep_center - lvl_num_d // 2
                cv_min_d_coarse = int(-(ep_min + lvl_num_d))
                cv_min_d_coarse = cv_min_d_coarse - (cv_min_d_coarse % 16)

                d_fwd, _ = _run_sgbm(img1_s, img2_s, cv_min_d_coarse,
                                     lvl_num_d, bs, lP1, lP2, lr_threshold, 0)
                del img1_s, img2_s

                v_fwd = d_fwd[~np.isnan(d_fwd)]
                if v_fwd.size > 500:
                    median_ep = float(np.median(v_fwd)) / scale
                # else keep the sparse hint as-is
                del d_fwd, v_fwd
            else:
                # No hint: search wide range, both directions (original behavior)
                lvl_num_d = min(sw, 256) - (min(sw, 256) % 16)
                if lvl_num_d < 16:
                    lvl_num_d = 16

                d_fwd, _ = _run_sgbm(img1_s, img2_s, 0, lvl_num_d, bs,
                                     lP1, lP2, lr_threshold, 0)
                d_rev, _ = _run_sgbm(img2_s, img1_s, 0, lvl_num_d, bs,
                                     lP1, lP2, lr_threshold, 0)
                del img1_s, img2_s

                v_fwd = d_fwd[~np.isnan(d_fwd)]
                v_rev = d_rev[~np.isnan(d_rev)]

                if v_fwd.size >= v_rev.size and v_fwd.size > 500:
                    median_ep = float(np.median(v_fwd)) / scale
                elif v_rev.size > 500:
                    median_ep = -float(np.median(v_rev)) / scale
                elif v_fwd.size > 0:
                    median_ep = float(np.median(v_fwd)) / scale
                else:
                    median_ep = None
                del d_fwd, d_rev, v_fwd, v_rev

        else:
            # Finer levels: refine around previous estimate
            if median_ep is not None:
                lvl_range = disp_range if scale == 1.0 else disp_range // 2
                lvl_num_d = lvl_range - (lvl_range % 16)
                # median_ep is in full-res pixels; scale to this level
                ep_center = median_ep * scale
                ep_min = ep_center - lvl_num_d // 2
                cv_min_d = int(-(ep_min + lvl_num_d))
                cv_min_d = cv_min_d - (cv_min_d % 16)
            else:
                # No estimate yet: search wide
                lvl_num_d = min(sw, 512) - (min(sw, 512) % 16)
                cv_min_d = 0

            bs = block_size if scale == 1.0 else max(5, block_size // 2 | 1)
            lP1, lP2 = 8 * bs**2, 32 * bs**2
            sp = speckle_size if scale == 1.0 else 0

            d1_lvl, _ = _run_sgbm(img1_s, img2_s, cv_min_d, lvl_num_d, bs, lP1, lP2, lr_threshold, sp)
            del img1_s, img2_s

            v_lvl = d1_lvl[~np.isnan(d1_lvl)]
            if v_lvl.size > 500:
                median_ep = float(np.median(v_lvl)) / scale
            num_d = lvl_num_d
            del d1_lvl, v_lvl


    # ── Final pass at full resolution ──
    # Use the larger of default disp_range or sparse-derived range
    final_range = disp_range
    if sparse_disp_hint is not None:
        sparse_range = int(sparse_disp_hint[1] * 2.5)  # 2.5x margin
        final_range = max(final_range, sparse_range)
    if median_ep is not None:
        num_d = final_range - (final_range % 16)
        num_d = max(num_d, 16)
        ep_min = median_ep - num_d // 2
        cv_min_d = int(-(ep_min + num_d))
        cv_min_d = cv_min_d - (cv_min_d % 16)
    else:
        num_d = final_range - (final_range % 16)
        num_d = max(num_d, 16)
        cv_min_d = 0

    _t0 = _time.time()
    d1, d2 = _run_sgbm(img1_gray, img2_gray,
                        cv_min_d, num_d, block_size,
                        P1, P2, lr_threshold, speckle_size)
    return d1, d2



def rectify_pair(K1, R1_wc, C1, K2, R2_wc, C2, W1, H1, W2=None, H2=None):
    """
    Stereo-rectify two cameras using the same algorithm as rectify.py:
      - R_rect with x-axis aligned to baseline, using average optical axis
      - Common image size from max of per-image bounding box ranges
      - Per-image cx centering, shared cy (average of both warped y-centers)
      - Homographies H = K_rect @ R_rect @ R_orig^T @ K_orig^{-1}

    Returns:
        rect_params: dict with R1r, R2r, K_rect1, K_rect2, Tx,
                     H1_ori2epi, H2_ori2epi, map1x, map1y, map2x, map2y,
                     Wr, Hr  (output rectified image dimensions)
    """
    if W2 is None: W2 = W1
    if H2 is None: H2 = H1

    t1 = -R1_wc @ C1
    t2 = -R2_wc @ C2

    # Shared rectifying rotation (from rectify.py)
    R_rect = compute_rectifying_rotation(R1_wc, t1, R2_wc, t2)

    # Per-camera rectification rotations
    R1r = R_rect @ R1_wc.T
    R2r = R_rect @ R2_wc.T

    fx1, fy1 = K1[0, 0], K1[1, 1]
    fx2, fy2 = K2[0, 0], K2[1, 1]

    # Convert to 0-indexed pixel convention (COLMAP stores cx/cy with pixel
    # center at 0.5; subtract 0.5 to match C++ pipeline convention).
    K1_0 = K1.copy(); K1_0[0, 2] -= 0.5; K1_0[1, 2] -= 0.5
    K2_0 = K2.copy(); K2_0[0, 2] -= 0.5; K2_0[1, 2] -= 0.5
    K1_0inv = np.linalg.inv(K1_0)
    K2_0inv = np.linalg.inv(K2_0)

    # Raw homographies (before centering adjustment)
    H1_raw = K1_0 @ R_rect @ R1_wc.T @ K1_0inv
    H2_raw = K2_0 @ R_rect @ R2_wc.T @ K2_0inv

    # Warp original image corners to find rectified bounding boxes
    corners1 = np.array([
        [0, W1 - 1, 0, W1 - 1],
        [0, 0, H1 - 1, H1 - 1],
        [1, 1, 1, 1]], dtype=float)
    corners2 = np.array([
        [0, W2 - 1, 0, W2 - 1],
        [0, 0, H2 - 1, H2 - 1],
        [1, 1, 1, 1]], dtype=float)

    def _warp_pts(H, pts):
        w = H @ pts
        return w[:2] / w[2:3]

    c1 = _warp_pts(H1_raw, corners1)
    c2 = _warp_pts(H2_raw, corners2)

    # Per-image bounding box ranges
    x_range_1 = c1[0].max() - c1[0].min()
    x_range_2 = c2[0].max() - c2[0].min()
    y_range_1 = c1[1].max() - c1[1].min()
    y_range_2 = c2[1].max() - c2[1].min()

    # Common rectified image size = ceiling of max range
    Wr = int(np.ceil(max(x_range_1, x_range_2)))
    Hr = int(np.ceil(max(y_range_1, y_range_2)))

    # Guard against degenerate pairs where rectification blows up
    max_dim = max(W1, H1, W2 or W1, H2 or H1) * 4
    if Wr > max_dim or Hr > max_dim or Wr >= 32767 or Hr >= 32767:
        log.warning(f"        rectify: canvas too large ({Wr}x{Hr}, max_dim={max_dim})")
        return None

    # Principal points: center each image's warped bounding box at the
    # rectified image center.  cx is per-image; cy is shared (average).
    x_center_1 = (c1[0].min() + c1[0].max()) / 2.0
    x_center_2 = (c2[0].min() + c2[0].max()) / 2.0
    y_center_1 = (c1[1].min() + c1[1].max()) / 2.0
    y_center_2 = (c2[1].min() + c2[1].max()) / 2.0

    rect_cx = (Wr - 1) / 2.0
    rect_cy = (Hr - 1) / 2.0

    cx0_1, cy0_1 = K1_0[0, 2], K1_0[1, 2]
    cx0_2, cy0_2 = K2_0[0, 2], K2_0[1, 2]

    cx1_rect = cx0_1 + (rect_cx - x_center_1)
    cx2_rect = cx0_2 + (rect_cx - x_center_2)
    y_center_avg = (y_center_1 + y_center_2) / 2.0
    cy_rect = cy0_1 + (rect_cy - y_center_avg)

    K_rect1 = np.array([[fx1, 0, cx1_rect],
                         [0, fy1, cy_rect],
                         [0,  0,  1]], dtype=np.float64)
    K_rect2 = np.array([[fx2, 0, cx2_rect],
                         [0, fy2, cy_rect],
                         [0,  0,  1]], dtype=np.float64)

    # Baseline Tx (signed; positive when C2 is along the rectified x-axis)
    Tx = R_rect[0] @ (C2 - C1)

    # Final homographies (0-indexed pixel convention)
    H1_ori2epi = K_rect1 @ R_rect @ R1_wc.T @ K1_0inv
    H2_ori2epi = K_rect2 @ R_rect @ R2_wc.T @ K2_0inv

    # Build remap tables from inverse homographies
    H1_epi2ori = np.linalg.inv(H1_ori2epi)
    H2_epi2ori = np.linalg.inv(H2_ori2epi)

    def _build_remap(H_epi2ori, wr, hr):
        uu, vv = np.meshgrid(
            np.arange(wr, dtype=np.float64),
            np.arange(hr, dtype=np.float64))
        pts = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3).T
        src = H_epi2ori @ pts
        src /= src[2:3]
        mx = src[0].reshape(hr, wr).astype(np.float32)
        my = src[1].reshape(hr, wr).astype(np.float32)
        return mx, my

    map1x, map1y = _build_remap(H1_epi2ori, Wr, Hr)
    map2x, map2y = _build_remap(H2_epi2ori, Wr, Hr)

    return dict(
        R1r=R1r, R2r=R2r,
        K_rect1=K_rect1, K_rect2=K_rect2,
        Tx=Tx,
        H1_ori2epi=H1_ori2epi, H2_ori2epi=H2_ori2epi,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        Wr=Wr, Hr=Hr, rotated=False,
        # Effective world-to-cam rotations used for rectification
        R1_wc_eff=R1_wc, R2_wc_eff=R2_wc,
    )


# ============================================================================
# Depth unprojection and grid fusion
# ============================================================================

def unproject_to_grid_orig(disp_rect, K_self, R_self_wc, C_self,
                           K_nb, R_nb_wc, C_nb,
                           H_self_ori2epi, H_nb_ori2epi, W, H,
                           min_angle_deg=3.0):
    """
    Triangulate from rectified disparity using original camera poses.

    1. For each valid pixel (ur, vr) in rectified master, the match in the
       rectified neighbor is at (ur + disp, vr).
    2. Map both back to original image coordinates via inverse homographies.
    3. Triangulate using original K, R, C.
    4. Filter by triangulation angle to remove poorly-constrained points.

    Returns (gX, gY, gZ) each float32 H×W, offset = X_world - C_self,
    stored in original-image pixel grid.
    """
    h_r, w_r = disp_rect.shape

    # Valid mask
    valid = ~np.isnan(disp_rect)
    vr_idx, ur_idx = np.where(valid)
    ur = ur_idx.astype(np.float64)
    vr = vr_idx.astype(np.float64)
    dr = disp_rect[valid].astype(np.float64)

    # Matching pixel in rectified neighbor: u2 = u1 + disp (ErrorProp: disp = u2 - u1)
    ur_nb = ur + dr

    # Map rectified → original via inverse homographies
    H_self_epi2ori = np.linalg.inv(H_self_ori2epi)
    H_nb_epi2ori   = np.linalg.inv(H_nb_ori2epi)

    ones = np.ones_like(ur)
    p1_rect = np.stack([ur, vr, ones], axis=0)        # 3 x N
    p2_rect = np.stack([ur_nb, vr, ones], axis=0)     # 3 x N

    p1_orig = H_self_epi2ori @ p1_rect
    p1_orig /= p1_orig[2:3]
    p2_orig = H_nb_epi2ori @ p2_rect
    p2_orig /= p2_orig[2:3]

    # Projection matrices using 0-indexed convention (matching homographies)
    K1_0 = K_self.copy(); K1_0[0,2] -= 0.5; K1_0[1,2] -= 0.5
    K2_0 = K_nb.copy();   K2_0[0,2] -= 0.5; K2_0[1,2] -= 0.5

    t_self = -R_self_wc @ C_self
    t_nb   = -R_nb_wc   @ C_nb

    P1 = K1_0 @ np.hstack([R_self_wc, t_self.reshape(3,1)])
    P2 = K2_0 @ np.hstack([R_nb_wc,   t_nb.reshape(3,1)])

    # Triangulate
    X_hom = cv2.triangulatePoints(P1, P2,
                                   p1_orig[:2], p2_orig[:2])  # 4 x N
    X_world = (X_hom[:3] / X_hom[3:4]).T                      # N x 3

    # Filter by triangulation angle — reject poorly-constrained points
    if min_angle_deg > 0:
        ray1 = X_world - C_self.reshape(1, 3)
        ray2 = X_world - C_nb.reshape(1, 3)
        cos_angle = (np.sum(ray1 * ray2, axis=1) /
                     (np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1) + 1e-12))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        good = angle_deg >= min_angle_deg
        X_world[~good] = np.nan

    # Filter by ray uncertainty (Qin's Tol_RAY_Uncertainty)
    # Pixel-space uncertainty for 1-pixel disparity error: σ_pixel ≈ Z / B
    # Reject points where this exceeds threshold (depth > tol * baseline)
    tol_ray = 20.0  # pixels
    B = np.linalg.norm(C_nb - C_self)
    depth_from_master = np.sum(
        (X_world - C_self.reshape(1, 3)) * R_self_wc[2, :].reshape(1, 3), axis=1
    )
    sigma_pix = np.abs(depth_from_master) / (B + 1e-12)
    bad_unc = (sigma_pix > tol_ray) | (depth_from_master <= 0)
    X_world[bad_unc] = np.nan

    # Store as offsets from C_self
    offsets = X_world - C_self.reshape(1, 3)

    # Fill rectified-space grids
    gX_r = np.full((h_r, w_r), np.nan, np.float32)
    gY_r = np.full((h_r, w_r), np.nan, np.float32)
    gZ_r = np.full((h_r, w_r), np.nan, np.float32)
    gX_r[vr_idx, ur_idx] = offsets[:, 0].astype(np.float32)
    gY_r[vr_idx, ur_idx] = offsets[:, 1].astype(np.float32)
    gZ_r[vr_idx, ur_idx] = offsets[:, 2].astype(np.float32)

    # Neighbor pixel correspondence in original image space (rectified grid)
    corr_x_r = np.full((h_r, w_r), np.nan, np.float32)
    corr_y_r = np.full((h_r, w_r), np.nan, np.float32)
    corr_x_r[vr_idx, ur_idx] = p2_orig[0].astype(np.float32)
    corr_y_r[vr_idx, ur_idx] = p2_orig[1].astype(np.float32)

    # Remap from rectified → original image space
    uo = np.arange(W, dtype=np.float32)
    vo = np.arange(H, dtype=np.float32)
    uog, vog = np.meshgrid(uo, vo)
    ones_o = np.ones_like(uog)
    pts = np.stack([uog, vog, ones_o], axis=-1).reshape(-1, 3).T
    pr  = H_self_ori2epi @ pts
    pr /= pr[2:3]
    mx = pr[0].reshape(H, W).astype(np.float32)
    my = pr[1].reshape(H, W).astype(np.float32)

    # Use INTER_NEAREST for XYZ grids to avoid interpolating between
    # foreground and background 3D points at depth edges, which creates
    # phantom "flying pixels" along the ray direction.
    def warp_nn(ch):
        return cv2.remap(ch, mx, my, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    def warp_lin(ch):
        return cv2.remap(ch, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return warp_nn(gX_r), warp_nn(gY_r), warp_nn(gZ_r), warp_lin(corr_x_r), warp_lin(corr_y_r)


def fuse_grids(depth_list):
    """Median-fuse a list of (gX,gY,gZ) arrays. Returns (gX,gY,gZ,ray_num)."""
    if not depth_list:
        return None
    Xs = np.stack([r[0] for r in depth_list])
    Ys = np.stack([r[1] for r in depth_list])
    Zs = np.stack([r[2] for r in depth_list])
    valid  = ~np.isnan(Xs)
    rn     = valid.sum(0).astype(np.uint16)
    with np.errstate(all="ignore"):
        gX = np.nanmedian(Xs, 0).astype(np.float32)
        gY = np.nanmedian(Ys, 0).astype(np.float32)
        gZ = np.nanmedian(Zs, 0).astype(np.float32)
    gX[rn==0] = np.nan
    gY[rn==0] = np.nan
    gZ[rn==0] = np.nan
    return gX, gY, gZ, rn


def fuse_point_clouds(pts_list, rgb_list, rn_list,
                      xyz_cov_list=None, voxel_size=0.05):
    """
    Voxel-based fusion of multiple overlapping point clouds using
    precision-weighted (information filter) combination.

    Points falling in the same voxel are fused:
      - position: precision-weighted mean  x = Σ_f (Σ₁⁻¹x₁ + Σ₂⁻¹x₂ + …)
        (falls back to median for points without covariance)
      - RGB: mean of contributing colors
      - ray_num: sum of contributing ray counts
      - covariance: Σ_f⁻¹ = Σ₁⁻¹ + Σ₂⁻¹ + …

    Parameters
    ----------
    pts_list     : list of (N_i, 3) arrays — world coordinates
    rgb_list     : list of (N_i, 3) uint8 arrays — per-point color
    rn_list      : list of (N_i,) uint16 arrays — per-point ray count
    xyz_cov_list : list of (N_i, 3, 3) arrays (or None) — per-point covariance.
                   Must align 1-to-1 with pts_list entries.
    voxel_size   : edge length of cubic voxels (in world units)

    Returns  (fused_pts, fused_rgb, fused_rn, fused_cov_or_None)
    """
    all_pts = np.concatenate(pts_list, axis=0)
    all_rgb = np.concatenate(rgb_list, axis=0)
    all_rn  = np.concatenate(rn_list, axis=0)
    has_cov = xyz_cov_list is not None and len(xyz_cov_list) > 0
    if has_cov:
        all_cov = np.concatenate(xyz_cov_list, axis=0)

    n = len(all_pts)
    if n == 0:
        empty3 = np.empty((0, 3), np.float64)
        return empty3, np.empty((0, 3), np.uint8), np.empty(0, np.uint16), None

    # Assign each point to a voxel
    voxel_idx = np.floor(all_pts / voxel_size).astype(np.int64)
    dt = np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])
    keys = np.empty(n, dtype=dt)
    keys['x'] = voxel_idx[:, 0]
    keys['y'] = voxel_idx[:, 1]
    keys['z'] = voxel_idx[:, 2]

    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_voxels = len(counts)

    fused_pts = np.empty((n_voxels, 3), np.float64)
    fused_rgb = np.empty((n_voxels, 3), np.float64)
    fused_rn  = np.empty(n_voxels, np.uint32)
    fused_cov = np.empty((n_voxels, 3, 3), np.float64) if has_cov else None

    # Sort by voxel for grouped access
    order = np.argsort(inverse)
    starts = np.concatenate([[0], np.cumsum(counts[:-1])])

    # Single-point voxels (fast path)
    single = counts == 1
    if single.any():
        s_idx = np.where(single)[0]
        orig = order[starts[s_idx]]
        fused_pts[s_idx] = all_pts[orig]
        fused_rgb[s_idx] = all_rgb[orig].astype(np.float64)
        fused_rn[s_idx]  = all_rn[orig]
        if has_cov:
            fused_cov[s_idx] = all_cov[orig]

    # Multi-point voxels — precision-weighted fusion
    multi = np.where(~single)[0]
    for vi in multi:
        members = order[starts[vi]:starts[vi] + counts[vi]]
        fused_rgb[vi] = np.mean(all_rgb[members].astype(np.float64), axis=0)
        fused_rn[vi]  = np.sum(all_rn[members])

        if has_cov:
            # Information filter: accumulate precision and info vector
            prec_sum = np.zeros((3, 3), np.float64)
            info_sum = np.zeros(3, np.float64)
            n_good = 0
            for m in members:
                c = all_cov[m]
                if np.any(np.isnan(c)):
                    continue
                try:
                    p = np.linalg.inv(c)
                except np.linalg.LinAlgError:
                    continue
                prec_sum += p
                info_sum += p @ all_pts[m]
                n_good += 1
            if n_good >= 1:
                try:
                    fused_cov[vi] = np.linalg.inv(prec_sum)
                    fused_pts[vi] = fused_cov[vi] @ info_sum
                except np.linalg.LinAlgError:
                    fused_pts[vi] = np.median(all_pts[members], axis=0)
                    fused_cov[vi] = np.nan
            else:
                fused_pts[vi] = np.median(all_pts[members], axis=0)
                fused_cov[vi] = np.nan
        else:
            fused_pts[vi] = np.median(all_pts[members], axis=0)

    fused_rgb = np.clip(fused_rgb, 0, 255).astype(np.uint8)
    fused_rn  = fused_rn.astype(np.uint16)

    log.info(f"    Voxel fusion: {n} input points -> {n_voxels} fused points "
          f"(voxel_size={voxel_size:.3f}, "
          f"{n - n_voxels} duplicates removed)")

    return fused_pts, fused_rgb, fused_rn, fused_cov


def geometric_consistency_filter(
        gX, gY, gZ, C_master, R_master_wc, cam_p_master,
        neighbor_info, max_reproj=2.0, max_depth_rel=0.01,
        min_consistent=3):
    """
    Cross-view geometric consistency filter (COLMAP-style).

    For each valid fused 3D point in the master image, reproject it into
    each neighbor camera and compare:
      1. Reprojection error < max_reproj pixels
      2. Relative depth difference < max_depth_rel

    Points that are not confirmed by at least `min_consistent` neighbor
    views are invalidated (set to NaN).

    Parameters
    ----------
    gX, gY, gZ : float32 H×W  — fused offset grids (X_world = gX + C_master)
    C_master : (3,) — master camera center
    R_master_wc : (3,3) — master world-to-cam rotation
    cam_p_master : (8,) — master [fx,fy,cx,cy,k1,k2,p1,p2]
    neighbor_info : list of dicts, each with:
        'gX', 'gY', 'gZ' : float32 Hn×Wn  — neighbor fused offset grids
        'C'  : (3,)   — neighbor camera center
        'R_wc' : (3,3) — neighbor world-to-cam rotation
        'cam_p' : (8,) — neighbor [fx,fy,cx,cy,k1,k2,p1,p2]
        'W', 'H' : int — neighbor image dimensions
    max_reproj : float — max reprojection error in pixels (default 2.0)
    max_depth_rel : float — max relative depth difference (default 0.01 = 1%)
    min_consistent : int — minimum consistent neighbors to keep a point

    Returns
    -------
    consistent_count : uint16 H×W — number of consistent neighbors per pixel
    """
    Hm, Wm = gX.shape
    valid = ~np.isnan(gX)
    vv, uu = np.where(valid)
    if len(vv) == 0:
        return np.zeros((Hm, Wm), dtype=np.uint16)

    # World coordinates of fused points
    Xw = gX[vv, uu] + C_master[0]
    Yw = gY[vv, uu] + C_master[1]
    Zw = gZ[vv, uu] + C_master[2]
    pts_world = np.stack([Xw, Yw, Zw], axis=1)  # (N, 3)

    # Master axis-angle for _project_opencv_batch
    aa_master = cv2.Rodrigues(R_master_wc)[0].flatten()

    consistent = np.zeros(len(vv), dtype=np.int32)

    for nb in neighbor_info:
        nb_C = nb['C']
        nb_R = nb['R_wc']
        nb_cam_p = nb['cam_p']
        nb_W, nb_H = nb['W'], nb['H']
        nb_gX, nb_gY, nb_gZ = nb['gX'], nb['gY'], nb['gZ']

        # Project master points into neighbor camera (with distortion)
        nb_aa = cv2.Rodrigues(nb_R)[0].flatten()
        u_proj, v_proj = _project_opencv_batch(pts_world, nb_aa, nb_C, nb_cam_p)

        # Camera-space Z for depth comparison
        D = pts_world - nb_C.reshape(1, 3)  # (N, 3)
        Zc = D @ nb_R[2, :]                 # (N,)

        # Behind camera + bounds check
        in_bounds = ((Zc > 0) &
                     (u_proj >= 0) & (u_proj < nb_W - 1) &
                     (v_proj >= 0) & (v_proj < nb_H - 1))
        if not in_bounds.any():
            continue

        # Look up neighbor depth at projected pixel (nearest neighbor)
        ui = np.round(u_proj[in_bounds]).astype(np.intp)
        vi = np.round(v_proj[in_bounds]).astype(np.intp)

        nb_valid = ~np.isnan(nb_gX[vi, ui])

        # Depth along neighbor viewing ray: master point vs neighbor point
        depth_master_in_nb = Zc[in_bounds]
        # Neighbor point depth (offset grid → camera Z)
        nb_offset = np.stack([nb_gX[vi, ui], nb_gY[vi, ui], nb_gZ[vi, ui]], axis=1)
        depth_nb = nb_offset @ nb_R[2, :]  # (M,)

        # Relative depth difference
        depth_diff_rel = np.abs(depth_master_in_nb - depth_nb) / (np.abs(depth_nb) + 1e-12)

        # Reprojection check: project neighbor 3D point back into master
        nb_pts_world = nb_offset + nb_C.reshape(1, 3)  # (M, 3)
        u_back, v_back = _project_opencv_batch(nb_pts_world, aa_master,
                                                C_master, cam_p_master)

        # Original master pixel coordinates
        uu_orig = uu[in_bounds].astype(np.float64)
        vv_orig = vv[in_bounds].astype(np.float64)
        reproj_err = np.sqrt((u_back - uu_orig)**2 + (v_back - vv_orig)**2)

        # A neighbor confirms a point if:
        # - neighbor has valid depth at that pixel
        # - relative depth difference is small
        # - reprojection error is small
        is_consistent = (nb_valid &
                         (depth_diff_rel < max_depth_rel) &
                         (reproj_err < max_reproj))

        idx = np.where(in_bounds)[0]
        consistent[idx[is_consistent]] += 1

    # Write result grid
    count_grid = np.zeros((Hm, Wm), dtype=np.uint16)
    count_grid[vv, uu] = consistent.astype(np.uint16)

    # Invalidate points that don't have enough consistent neighbors
    fail = count_grid < min_consistent
    gX[fail] = np.nan
    gY[fail] = np.nan
    gZ[fail] = np.nan

    return count_grid


def _save_las(path, pts, rgb=None, ray_num=None):
    """Save Nx3 points to a LAS file with optional RGB and ray_num."""
    import laspy
    las = laspy.LasData(laspy.LasHeader(point_format=2, version="1.2"))
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    if rgb is not None:
        las.red   = rgb[:, 0].astype(np.uint16) * 256
        las.green = rgb[:, 1].astype(np.uint16) * 256
        las.blue  = rgb[:, 2].astype(np.uint16) * 256
    if ray_num is not None:
        las.intensity = ray_num.astype(np.uint16)
    las.write(str(path))


# ============================================================================
# Measurement error prediction
# ============================================================================


def _project_opencv_batch(X_world, aa, c, cam_p):
    """
    Vectorized projection of Nx3 world points through COLMAP OPENCV model.
    Returns (u, v) each length-N arrays, or None entries replaced by NaN.
    """
    fx, fy, cx, cy, k1, k2, p1, p2 = cam_p
    d = X_world - c.reshape(1, 3)  # (N, 3)
    theta2 = aa @ aa
    if theta2 > np.finfo(float).eps:
        theta = np.sqrt(theta2)
        axis = aa / theta
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        X_cam = (cos_t * d
                 + (1.0 - cos_t) * np.outer(d @ axis, axis)
                 + sin_t * np.cross(axis, d))
    else:
        X_cam = d + np.cross(aa, d)

    Zc = X_cam[:, 2]
    xp = X_cam[:, 0] / Zc
    yp = X_cam[:, 1] / Zc
    r2 = xp * xp + yp * yp
    r4 = r2 * r2
    rad = 1.0 + k1 * r2 + k2 * r4
    u = fx * (xp * rad + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx
    v = fy * (yp * rad + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp) + cy
    return u, v


def predict_measurement_error(
        disp_rect,          # float32 H×W, ErrorProp convention
        energy_rect,        # uint16 H×W, SGM energy proxy
        num_ray_epi,        # uint16 H×W (numray in rectified space)
        reproj_epi,         # float32 H×W×4 reprojection error map
        H_ori2epi,          # 3×3 homography original→rectified
        W, H,               # original image dimensions
        multi_ray=6,
        num_std=1,
        num_intervals=8,
):
    """
    Calibrate an energy→measurement-error curve from multi-ray pixels, then
    predict per-pixel measurement error in original image coordinates.

    Returns (me_x, me_y) each float32 H×W.
    """
    hr, wr = disp_rect.shape

    # ---- build calibration mask ----
    # Use pixels where: reproj is valid, self-reproj < 1 px, numray > multi_ray
    base_mask = (
        ~np.isnan(reproj_epi[..., 0]) &
        (np.abs(reproj_epi[..., 2]) < 1.0)
    )
    min_calib_pixels = 100         # need enough samples to fill ≥3 bins of 5
    min_ray = 3                     # floor: never fall below ray>3 (i.e. require ≥4 rays)
    effective_ray = multi_ray
    mask = base_mask & (num_ray_epi > effective_ray)
    while mask.sum() < min_calib_pixels and effective_ray > min_ray:
        effective_ray -= 1
        mask = base_mask & (num_ray_epi > effective_ray)
    if effective_ray < multi_ray and mask.sum() >= min_calib_pixels:
        log.warning(f"ray>{multi_ray} had <{min_calib_pixels} pixels; fell back to ray>{effective_ray} "
                    f"({mask.sum()} pixels)")

    xs = energy_rect[mask].astype(np.float64)
    ys = reproj_epi[mask, 0].astype(np.float64)   # err2_x (signed, matching C++)

    if len(xs) == 0:
        n_reproj = (~np.isnan(reproj_epi[..., 0])).sum()
        n_selfok = (np.abs(reproj_epi[..., 2]) < 1.0).sum()
        n_ray    = (num_ray_epi > effective_ray).sum()
        log.warning(f"No calibration samples (reproj valid:{n_reproj}, "
              f"self<1px:{n_selfok}, ray>{effective_ray}:{n_ray}); "
              "returning NaN measurement error (pair skipped).")
        return np.full((H, W), np.nan, np.float32), np.full((H, W), np.nan, np.float32), np.full((hr, wr), np.nan, np.float32)

    # ---- fit energy→error curve ----
    X_curve, Y_curve = [], []
    ni = num_intervals
    while ni >= 3:
        X_curve.clear(); Y_curve.clear()
        lo = xs.min()
        sorted_xs = np.sort(xs)
        mid_997 = min(sorted_xs[int(0.997 * len(sorted_xs))], 8000.)
        # C++ linspace: ni evenly-spaced edges from lo, last edge = max
        step = (mid_997 - lo) / (ni - 1)
        edges = np.array([lo + i * step for i in range(ni)] + [xs.max()])
        for k in range(ni):
            sel = (xs >= edges[k]) & (xs < edges[k+1])
            if sel.sum() < 5:
                continue
            X_curve.append(xs[sel].mean())
            Y_curve.append(num_std * ys[sel].std())
        if len(X_curve) >= int(ni * 0.75):
            break
        ni -= 2

    if not X_curve:
        log.warning("Calibration failed; returning NaN measurement error (pair skipped).")
        return np.full((H, W), np.nan, np.float32), np.full((H, W), np.nan, np.float32), np.full((hr, wr), np.nan, np.float32)

    # ---- predict per-pixel error in rectified space (vectorized) ----
    me_epi = np.full((hr, wr), np.nan, np.float32)
    valid_disp = ~np.isnan(disp_rect)
    if valid_disp.any():
        ene_vals = energy_rect[valid_disp].astype(np.float64)
        X_arr = np.array(X_curve)
        Y_arr = np.array(Y_curve)
        pred_vals = np.interp(ene_vals, X_arr, Y_arr).astype(np.float32)
        actual_vals = reproj_epi[..., 0][valid_disp]  # signed, matching C++
        actual_vals = np.where(np.isnan(actual_vals), 0., actual_vals)
        me_epi[valid_disp] = np.maximum(pred_vals, np.abs(actual_vals))

    # ---- warp to original image space ----
    # me_epi is a disparity-axis (x-direction) error in rectified coords.
    # Convert to (me_x, me_y) in original image: displace the rectified point
    # by the error amount, transform both through epi→ori, take the difference.
    H_epi2ori = np.linalg.inv(H_ori2epi)

    uo, vo = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    ones = np.ones_like(uo)
    pts  = np.stack([uo, vo, ones], axis=-1).reshape(-1, 3).T   # (3, N)
    pr   = H_ori2epi @ pts
    pr  /= pr[2:3]
    mx   = pr[0].reshape(H, W).astype(np.float32)
    my   = pr[1].reshape(H, W).astype(np.float32)

    me_sampled = cv2.remap(me_epi, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    # Transform (epi_x, epi_y) and (epi_x + me, epi_y) back to original space
    orig_x = np.full((H, W), np.nan, np.float32)
    orig_y = np.full((H, W), np.nan, np.float32)
    valid  = ~np.isnan(me_sampled)
    if valid.any():
        px  = mx[valid]
        py  = my[valid]
        me  = me_sampled[valid]

        def transform(x_arr, y_arr):
            pts2 = np.stack([x_arr, y_arr, np.ones_like(x_arr)], 0)
            res  = H_epi2ori @ pts2
            res /= res[2:3]
            return res[0], res[1]

        ox0, oy0 = transform(px, py)
        ox1, oy1 = transform(px + me, py)
        orig_x[valid] = np.abs(ox1 - ox0).astype(np.float32)
        orig_y[valid] = np.abs(oy1 - oy0).astype(np.float32)

    return orig_x, orig_y, me_epi


# ============================================================================
# Error propagation (Gauss-Markov)
# ============================================================================

def _jac_point_batch(X_batch, aa, c, cam_p):
    """
    Batched ∂(u,v)/∂(X,Y,Z) for N points. Returns (N, 2, 3).
    X_batch: (N, 3), aa/c: (3,), cam_p: (8,).
    """
    fx, fy = cam_p[0], cam_p[1]
    D = X_batch - c.reshape(1, 3)  # (N, 3)
    theta2 = aa @ aa

    if theta2 > np.finfo(float).eps:
        theta = np.sqrt(theta2)
        n = aa / theta
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        one_cos = 1.0 - cos_t
        # R = cos*I + (1-cos)*n⊗n + sin*[n]×
        n_skew = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R = cos_t * np.eye(3) + one_cos * np.outer(n, n) + sin_t * n_skew
    else:
        n_skew = np.array([[0, -aa[2], aa[1]], [aa[2], 0, -aa[0]], [-aa[1], aa[0], 0]])
        R = np.eye(3) + n_skew

    Xc = (R @ D.T).T  # (N, 3)
    Zc = Xc[:, 2]
    xp = Xc[:, 0] / Zc
    yp = Xc[:, 1] / Zc

    k1, k2, p1, p2 = cam_p[4:]
    r2 = xp * xp + yp * yp
    drad = k1 + 2.0 * k2 * r2
    rad = 1.0 + k1 * r2 + k2 * r2 * r2

    # J_dist (N, 2, 2) — ∂(xd,yd)/∂(xp,yp)
    dxd_dxp = rad + 2.0 * xp * xp * drad + 2.0 * p1 * yp + 6.0 * p2 * xp
    dxd_dyp = 2.0 * xp * yp * drad + 2.0 * p1 * xp + 2.0 * p2 * yp
    dyd_dyp = rad + 2.0 * yp * yp * drad + 6.0 * p1 * yp + 2.0 * p2 * xp

    # J_persp (N, 2, 3) — ∂(xp,yp)/∂Xc
    inv_Zc = 1.0 / Zc
    J_persp = np.zeros((len(Xc), 2, 3))
    J_persp[:, 0, 0] = inv_Zc
    J_persp[:, 0, 2] = -xp * inv_Zc
    J_persp[:, 1, 1] = inv_Zc
    J_persp[:, 1, 2] = -yp * inv_Zc

    # J_f @ J_dist @ J_persp @ R, all batched
    # J_f is diagonal: [[fx, 0], [0, fy]]
    # J_dist is (N, 2, 2), J_persp is (N, 2, 3), R is (3, 3)
    # result: (N, 2, 3)
    # First: J_dist @ J_persp → (N, 2, 3)
    JdJp = np.zeros((len(Xc), 2, 3))
    JdJp[:, 0, :] = dxd_dxp[:, None] * J_persp[:, 0, :] + dxd_dyp[:, None] * J_persp[:, 1, :]
    JdJp[:, 1, :] = dxd_dyp[:, None] * J_persp[:, 0, :] + dyd_dyp[:, None] * J_persp[:, 1, :]

    # Apply J_f (scale rows)
    JdJp[:, 0, :] *= fx
    JdJp[:, 1, :] *= fy

    # Apply R: (N, 2, 3) @ (3, 3) → (N, 2, 3)
    return JdJp @ R



# ============================================================================
# Main pipeline
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",          required=True)
    ap.add_argument("--images",         required=True)
    ap.add_argument("--out",            required=True)
    ap.add_argument("--max_neighbors",  type=int, default=10,
                    help="Max co-visible neighbors per image; 0=all")
    ap.add_argument("--optimal_angle", type=float, default=10.0,
                    help="Co-visibility: peak triangulation angle in degrees (default 10, OpenMVS)")
    ap.add_argument("--min_angle",     type=float, default=1.5,
                    help="Triangulation: reject points below this angle in degrees (default 3)")
    ap.add_argument("--disp_range",     type=int, default=128)
    ap.add_argument("--block_size",     type=int, default=9)
    ap.add_argument("--speckle_size",   type=int, default=120)
    ap.add_argument("--lr_threshold",   type=int, default=1)
    ap.add_argument("--coarse_long_side", type=int, default=700,
                    help="Long-side resolution for coarse disparity estimation")
    ap.add_argument("--multi_rays",     type=int, default=6)
    ap.add_argument("--num_std",        type=int, default=1)
    ap.add_argument("--sample_interval",type=int, default=3,
                    help="Sample every N pixels for propagation")
    ap.add_argument("--min_rays",      type=int, default=3,
                    help="Minimum ray count to keep a fused pixel (default 3)")
    ap.add_argument("--voxel_size",    type=float, default=0.05,
                    help="Voxel edge length for fusing overlapping point clouds")
    ap.add_argument("--geom_max_reproj", type=float, default=2.0,
                    help="Geometric consistency: max reprojection error (px)")
    ap.add_argument("--geom_max_depth_rel", type=float, default=0.01,
                    help="Geometric consistency: max relative depth diff (0.01=1%%)")
    ap.add_argument("--geom_min_consistent", type=int, default=3,
                    help="Geometric consistency: min consistent neighbors")
    ap.add_argument("--workers",       type=int, default=8,
                    help="Max parallel SGM workers (reduce for large images)")
    ap.add_argument("--scale",         type=float, default=1.0,
                    help="Downsample factor for images (e.g. 0.5 = half res)")
    ap.add_argument("--resume_tmp",    type=str, default=None,
                    help="Path to existing mvs_tmp_* dir to skip steps 2-3")
    ap.add_argument("--sift_graph",   action="store_true",
                    help="Use SIFT matching on coarse images to select neighbors "
                         "(instead of co-visibility scoring)")
    args = ap.parse_args()

    scene_dir = Path(args.scene)
    image_dir = Path(args.images)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    si = args.sample_interval

    # ------------------------------------------------------------------
    # Set up logging to console + file
    # ------------------------------------------------------------------
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File handler (DEBUG) – timestamped log in output dir
    log_file = out_dir / f"mvs_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    log.info("Log file: %s", log_file)
    log.info("Args: %s", vars(args))

    # ------------------------------------------------------------------
    # 1. Load COLMAP model
    # ------------------------------------------------------------------
    log.info("[1/5] Loading COLMAP model ...")
    cameras = read_cameras_bin(scene_dir / "cameras.bin")
    images  = read_images_bin(scene_dir  / "images.bin")
    points3D_path = scene_dir / "points3D.bin"
    points3D = read_points3D_bin(points3D_path) if points3D_path.exists() else {}
    log.info(f"      {len(cameras)} cameras, {len(images)} images, {len(points3D)} points3D")

    # Downsample: scale camera intrinsics and dimensions once
    if args.scale < 1.0:
        log.info(f"      Applying --scale {args.scale} to all cameras")
        for cam in cameras.values():
            cam["width"]  = int(cam["width"]  * args.scale)
            cam["height"] = int(cam["height"] * args.scale)
            p = cam["params"]
            if cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                # params: f, cx, cy, [k1, k2]
                p[0] *= args.scale   # f
                p[1] *= args.scale   # cx
                p[2] *= args.scale   # cy
            elif cam["model"] in ("PINHOLE", "OPENCV"):
                # params: fx, fy, cx, cy, [k1, k2, p1, p2]
                p[0] *= args.scale   # fx
                p[1] *= args.scale   # fy
                p[2] *= args.scale   # cx
                p[3] *= args.scale   # cy

        # Write scaled sparse model for downstream partition_data.py
        _MODEL_NAME_TO_ID = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}
        scaled_sparse = out_dir / "sparse_scaled"
        scaled_sparse.mkdir(parents=True, exist_ok=True)
        with open(scaled_sparse / "cameras.bin", "wb") as f:
            f.write(struct.pack("<Q", len(cameras)))
            for cid, cam in cameras.items():
                mid = _MODEL_NAME_TO_ID[cam["model"]]
                f.write(struct.pack("<I", cid))
                f.write(struct.pack("<I", mid))
                f.write(struct.pack("<Q", cam["width"]))
                f.write(struct.pack("<Q", cam["height"]))
                f.write(struct.pack(f"<{len(cam['params'])}d", *cam["params"]))
        import shutil as _shutil
        _shutil.copy2(scene_dir / "images.bin",   scaled_sparse / "images.bin")
        _shutil.copy2(scene_dir / "points3D.bin", scaled_sparse / "points3D.bin")
        log.info(f"      Wrote scaled sparse model → {scaled_sparse}")

    sorted_imgs = sorted(images.items())   # stable ordering

    import tempfile, shutil

    # depth_files[stem] = list of temp .npz paths containing (gX, gY, gZ)
    depth_files = defaultdict(list)
    # pair_files[stem][stem2] = temp .npz path for pair data
    pair_files = defaultdict(dict)
    # iid_for_stem[stem] = iid  (for looking up image id from stem)
    iid_for_stem = {}

    if args.resume_tmp:
        # ----------------------------------------------------------
        # Resume: rebuild depth_files / pair_files from existing dir
        # ----------------------------------------------------------
        tmp_dir = Path(args.resume_tmp)
        log.info(f"[2/5] Skipped (resuming from {tmp_dir})")
        log.info(f"[3/5] Skipped (resuming from {tmp_dir})")
        # Build set of known stems for filename parsing
        all_stems = set()
        for iid, img in sorted_imgs:
            stem = Path(img["name"]).stem
            iid_for_stem[stem] = iid
            all_stems.add(stem)

        for f in sorted(tmp_dir.glob("depth_*.npz")):
            # depth_{stem1}--{stem2}_{1|2}.npz
            name = f.name[len("depth_"):]          # strip prefix
            name = name.rsplit("_", 1)[0]           # strip _1.npz / _2.npz
            if "--" not in name:
                continue
            s1, s2 = name.split("--", 1)
            if s1 in all_stems and s2 in all_stems:
                depth_files[s1].append(str(f))
        for f in sorted(tmp_dir.glob("pair_*.npz")):
            # pair_{stem1}--{stem2}.npz
            name = f.stem[len("pair_"):]            # strip prefix and .npz
            if "--" not in name:
                continue
            s1, s2 = name.split("--", 1)
            if s1 in all_stems and s2 in all_stems:
                pair_files[s1][s2] = str(f)
        log.info(f"      {sum(len(v) for v in depth_files.values())} depth grids, "
              f"{sum(len(v) for v in pair_files.values())} pair files on disk")
    else:
        # ----------------------------------------------------------
        # 2. Co-visibility graph
        # ----------------------------------------------------------
        log.info("[2/5] Building co-visibility graph ...")
        if args.sift_graph:
            # Union of covisibility (top N) and SIFT matching (top 7)
            graph_covis = build_covis_graph(images, args.max_neighbors,
                                            points3D=points3D, cameras=cameras,
                                            optimal_angle_deg=args.optimal_angle)
            graph_sift = build_sift_match_graph(images, image_dir,
                                                max_neighbors=args.max_neighbors,
                                                coarse_long_side=1024)
            graph = {}
            for iid in images:
                covis_nbs = graph_covis.get(iid, [])
                sift_nbs = graph_sift.get(iid, [])
                # Union: covis first, then sift (deduped)
                seen = set()
                merged = []
                for n in covis_nbs + sift_nbs:
                    if n not in seen:
                        seen.add(n)
                        merged.append(n)
                graph[iid] = merged
            n_nbs = [len(v) for v in graph.values()]
            log.info(f"      Union graph: {min(n_nbs)}-{max(n_nbs)} neighbors per image "
                     f"(mean {sum(n_nbs)/len(n_nbs):.1f})")
        else:
            graph = build_covis_graph(images, args.max_neighbors,
                                      points3D=points3D, cameras=cameras,
                                      optimal_angle_deg=args.optimal_angle)

        # ----------------------------------------------------------
        # 3. Stereo processing: SGM → depth grids  (spilled to disk)
        # ----------------------------------------------------------
        log.info("[3/5] Running SGM stereo for all pairs ...")

        tmp_dir = Path(tempfile.mkdtemp(prefix="mvs_tmp_", dir=out_dir))
        log.info(f"      Temp dir: {tmp_dir}")

        done_pairs = set()
        pair_jobs = []   # collect all pair info for parallel processing
        for iid, img in sorted_imgs:
            stem1 = Path(img["name"]).stem
            iid_for_stem[stem1] = iid
            cam  = cameras[img["cam_id"]]
            K1, dist1 = get_K_dist(cam)
            R1_wc = opk_to_R(*img["opk"])
            C1    = -(R1_wc.T @ np.array(img["tvec"]))
            W, H  = cam["width"], cam["height"]

            for nb_iid in graph.get(iid, []):
                if nb_iid not in images:
                    continue
                key = (min(iid, nb_iid), max(iid, nb_iid))
                if key in done_pairs:
                    continue
                done_pairs.add(key)

                nb_img    = images[nb_iid]
                stem2     = Path(nb_img["name"]).stem
                cam2      = cameras[nb_img["cam_id"]]
                K2, dist2 = get_K_dist(cam2)
                R2_wc     = opk_to_R(*nb_img["opk"])
                C2        = -(R2_wc.T @ np.array(nb_img["tvec"]))
                W2, H2 = cam2["width"], cam2["height"]

                pair_jobs.append(dict(
                    iid=iid, nb_iid=nb_iid,
                    name1=img["name"], name2=nb_img["name"],
                    stem1=stem1, stem2=stem2,
                    K1=K1, K2=K2, dist1=dist1, dist2=dist2,
                    R1_wc=R1_wc, R2_wc=R2_wc, C1=C1, C2=C2,
                    W=W, H=H, W2=W2, H2=H2,
                    pt3d_ids1=img["pt3d_ids"],
                    pt3d_ids2=nb_img["pt3d_ids"],
                ))

        log.info(f"      {len(pair_jobs)} pairs to process")

        def _process_one_pair(job):
            """Process a single stereo pair: rectify, SGM, triangulate, save."""
            stem1, stem2 = job["stem1"], job["stem2"]
            K1, K2 = job["K1"], job["K2"]
            dist1, dist2 = job["dist1"], job["dist2"]
            R1_wc, R2_wc = job["R1_wc"], job["R2_wc"]
            C1, C2 = job["C1"], job["C2"]
            W, H, W2, H2 = job["W"], job["H"], job["W2"], job["H2"]

            try:
                img1_gray = load_image(image_dir, job["name1"], gray=True)
                img2_gray = load_image(image_dir, job["name2"], gray=True)
            except FileNotFoundError as e:
                log.warning(f"        skip: image not found – {e}")
                return None

            K1_pair, K2_pair = K1.copy(), K2.copy()
            if img1_gray.shape[:2] != (H, W):
                img1_gray = cv2.resize(img1_gray, (W, H))
            if img2_gray.shape[:2] != (H2, W2):
                img2_gray = cv2.resize(img2_gray, (W2, H2))

            if np.any(dist1 != 0):
                img1_gray = cv2.undistort(img1_gray, K1_pair, dist1)
            if np.any(dist2 != 0):
                img2_gray = cv2.undistort(img2_gray, K2_pair, dist2)

            rp = rectify_pair(K1_pair, R1_wc, C1, K2_pair, R2_wc, C2,
                              W, H, W2, H2)
            if rp is None:
                log.warning(f"        skip: rectify_pair failed (degenerate geometry or canvas too large)")
                return None

            Rr_wc = rp["R1r"] @ rp["R1_wc_eff"]

            rect1 = cv2.remap(img1_gray, rp["map1x"], rp["map1y"], cv2.INTER_LINEAR)
            rect2 = cv2.remap(img2_gray, rp["map2x"], rp["map2y"], cv2.INTER_LINEAR)
            del img1_gray, img2_gray

            _ones1 = np.full((H, W), 255, dtype=np.uint8)
            _ones2 = np.full((H2, W2), 255, dtype=np.uint8)
            valid_rect1 = cv2.remap(_ones1, rp["map1x"], rp["map1y"],
                                    cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid_rect2 = cv2.remap(_ones2, rp["map2x"], rp["map2y"],
                                    cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            _half = args.block_size // 2
            _kern = np.ones((2 * _half + 1, 2 * _half + 1), np.uint8)
            valid_rect1 = cv2.erode(valid_rect1, _kern)
            valid_rect2 = cv2.erode(valid_rect2, _kern)

            R_rect = rp["R1r"] @ rp["R1_wc_eff"]
            sparse_hint = estimate_disp_range_from_sparse(
                job["pt3d_ids1"], job["pt3d_ids2"], points3D,
                R_rect, C1, C2, rp["K_rect1"], rp["K_rect2"])

            disp1, disp2 = run_sgm_pair(rect1, rect2,
                                        args.disp_range, args.block_size,
                                        args.speckle_size, args.lr_threshold,
                                        args.coarse_long_side,
                                        sparse_disp_hint=sparse_hint)

            def _mask_border(disp, v_master, v_neighbor):
                disp[v_master == 0] = np.nan
                valid = ~np.isnan(disp)
                if not valid.any():
                    return
                vv, uu = np.where(valid)
                mu = np.round(uu + disp[vv, uu]).astype(np.intp)
                oob = (mu < 0) | (mu >= v_neighbor.shape[1])
                ok = ~oob
                nb_border = np.ones(len(vv), dtype=bool)
                nb_border[ok] = v_neighbor[vv[ok], mu[ok]] == 0
                disp[vv[oob | nb_border], uu[oob | nb_border]] = np.nan

            _mask_border(disp1, valid_rect1, valid_rect2)
            _mask_border(disp2, valid_rect2, valid_rect1)

            n_valid1 = int(np.count_nonzero(~np.isnan(disp1)))
            n_valid2 = int(np.count_nonzero(~np.isnan(disp2)))
            if n_valid1 == 0 or n_valid2 == 0:
                log.warning(f"        skip: zero valid disparity pixels after masking (n1={n_valid1}, n2={n_valid2})")
                return None

            energy1 = compute_energy_proxy(rect1, rect2, disp1, args.block_size)
            energy2 = compute_energy_proxy(rect2, rect1, disp2, args.block_size)
            del rect1, rect2

            gX1, gY1, gZ1, _, _ = unproject_to_grid_orig(
                disp1, K1_pair, R1_wc, C1, K2_pair, R2_wc, C2,
                rp["H1_ori2epi"], rp["H2_ori2epi"], W, H,
                min_angle_deg=args.min_angle)
            gX2, gY2, gZ2, _, _ = unproject_to_grid_orig(
                disp2, K2_pair, R2_wc, C2, K1_pair, R1_wc, C1,
                rp["H2_ori2epi"], rp["H1_ori2epi"], W2, H2,
                min_angle_deg=args.min_angle)

            # Save to disk
            df1 = str(tmp_dir / f"depth_{stem1}--{stem2}_1.npz")
            np.savez_compressed(df1, gX=gX1, gY=gY1, gZ=gZ1)
            df2 = str(tmp_dir / f"depth_{stem2}--{stem1}_2.npz")
            np.savez_compressed(df2, gX=gX2, gY=gY2, gZ=gZ2)

            pf1 = str(tmp_dir / f"pair_{stem1}--{stem2}.npz")
            np.savez_compressed(pf1,
                disp_rect=disp1, energy_rect=energy1,
                R_self_wc=Rr_wc, R_nb_wc=Rr_wc,
                H_ori2epi=rp["H1_ori2epi"],
                K_rect_self=rp["K_rect1"], K_rect_nb=rp["K_rect2"],
                C_self=C1, C_nb=C2, iid=job["iid"], nb_iid=job["nb_iid"])

            pf2 = str(tmp_dir / f"pair_{stem2}--{stem1}.npz")
            np.savez_compressed(pf2,
                disp_rect=disp2, energy_rect=energy2,
                R_self_wc=Rr_wc, R_nb_wc=Rr_wc,
                H_ori2epi=rp["H2_ori2epi"],
                K_rect_self=rp["K_rect2"], K_rect_nb=rp["K_rect1"],
                C_self=C2, C_nb=C1, iid=job["nb_iid"], nb_iid=job["iid"])

            return dict(stem1=stem1, stem2=stem2,
                        df1=df1, df2=df2, pf1=pf1, pf2=pf2,
                        Wr=rp["Wr"], Hr=rp["Hr"])

        # Run pairs in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # SGM is CPU-bound and releases the GIL in OpenCV, so threads work.
        # Limit workers to avoid memory pressure (~2GB per pair).
        n_workers = max(1, min(args.workers, len(pair_jobs)))
        log.info(f"      Processing {len(pair_jobs)} pairs with {n_workers} workers")

        pair_idx = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_one_pair, job): job
                       for job in pair_jobs}
            for future in as_completed(futures):
                pair_idx += 1
                result = future.result()
                job = futures[future]
                s1, s2 = job["stem1"], job["stem2"]
                if result is None:
                    log.warning(f"      Pair {pair_idx}: {s1} <-> {s2}  [SKIP]")
                    continue
                depth_files[s1].append(result["df1"])
                depth_files[s2].append(result["df2"])
                pair_files[s1][s2] = result["pf1"]
                pair_files[s2][s1] = result["pf2"]
                log.info(f"      Pair {pair_idx}/{len(pair_jobs)}: {s1} <-> {s2}  "
                      f"rect {result['Wr']}x{result['Hr']}")

        log.info(f"      {sum(len(v) for v in depth_files.values())} depth grids, "
              f"{sum(len(v) for v in pair_files.values())} pair files on disk")

    # Precompute camera poses (shared across all per-image npz saves)
    cam_xyz = np.array([
        -(opk_to_R(*im["opk"]).T @ np.array(im["tvec"]))
        for _, im in sorted_imgs
    ], dtype=np.float64)
    cam_R = np.array([
        opk_to_R(*im["opk"])
        for _, im in sorted_imgs
    ], dtype=np.float64)

    # ------------------------------------------------------------------
    # 4. Per-image: fuse depths, predict ME, propagate covariance
    # ------------------------------------------------------------------
    log.info("[4/5] Per-image fusion, measurement error, and error propagation ...")
    log.info(f"  master ME: fixed 0.25 px")

    all_las_pts = []
    all_las_rgb = []
    all_las_rn  = []
    all_xyz_list = []
    all_cov_list = []
    n_written_total = 0

    # Cache fused depth grids for geometric consistency
    # stem → (gX, gY, gZ, ray_num)
    fused_grid_cache = {}

    def _get_fused_grid(stem):
        """Load and fuse depth grids for a given image stem, with caching."""
        if stem in fused_grid_cache:
            return fused_grid_cache[stem]
        if stem not in depth_files or not depth_files[stem]:
            return None
        dl = []
        for df_path in depth_files[stem]:
            d = np.load(df_path)
            dl.append((d["gX"], d["gY"], d["gZ"]))
        result = fuse_grids(dl)
        if result is not None:
            fused_grid_cache[stem] = result
        return result

    for iid, img in sorted_imgs:
        stem1 = Path(img["name"]).stem
        if stem1 not in depth_files or not depth_files[stem1]:
            continue

        cam   = cameras[img["cam_id"]]
        W, H  = cam["width"], cam["height"]
        R1_wc = opk_to_R(*img["opk"])
        C1    = -(R1_wc.T @ np.array(img["tvec"]))
        cp1   = cam_params_array(cam)

        # --- Load and fuse depth grids from disk ---
        fused_result = _get_fused_grid(stem1)
        if fused_result is None:
            log.warning(f"[SKIP] No depth for {stem1}")
            continue
        # Copy so geometric consistency filter can modify in-place
        gX, gY, gZ, ray_num = (fused_result[0].copy(), fused_result[1].copy(),
                                fused_result[2].copy(), fused_result[3])

        n_before = (~np.isnan(gX)).sum()

        # --- Geometric consistency filter ---
        nb_stems = list(pair_files.get(stem1, {}).keys())
        if nb_stems and args.geom_min_consistent > 0:
            neighbor_info = []
            for nb_stem in nb_stems:
                nb_iid = iid_for_stem.get(nb_stem)
                if nb_iid is None:
                    continue
                nb_img = images[nb_iid]
                nb_cam = cameras[nb_img["cam_id"]]
                nb_R_wc = opk_to_R(*nb_img["opk"])
                nb_C = -(nb_R_wc.T @ np.array(nb_img["tvec"]))
                nb_W, nb_H = nb_cam["width"], nb_cam["height"]

                # Get neighbor fused grid
                nb_fused = _get_fused_grid(nb_stem)
                if nb_fused is None:
                    continue
                nb_gX, nb_gY, nb_gZ = nb_fused[0], nb_fused[1], nb_fused[2]

                neighbor_info.append(dict(
                    gX=nb_gX, gY=nb_gY, gZ=nb_gZ,
                    C=nb_C, R_wc=nb_R_wc, cam_p=cam_params_array(nb_cam),
                    W=nb_W, H=nb_H,
                ))

            if neighbor_info:
                geom_count = geometric_consistency_filter(
                    gX, gY, gZ, C1, R1_wc, cp1,
                    neighbor_info,
                    max_reproj=args.geom_max_reproj,
                    max_depth_rel=args.geom_max_depth_rel,
                    min_consistent=args.geom_min_consistent,
                )
                n_after = (~np.isnan(gX)).sum()
                pos = geom_count[geom_count > 0]
                med_str = f"{np.median(pos):.0f}" if len(pos) > 0 else "n/a"
                log.info(f"  Geom consistency {stem1}: {n_before} -> {n_after} "
                      f"({n_before - n_after} removed, "
                      f"{len(neighbor_info)} neighbors checked, "
                      f"median consistent={med_str})")
            del neighbor_info

        log.info(f"  Fused {stem1}: ray_num distribution: "
              f"1={np.sum(ray_num==1)}, 2={np.sum(ray_num==2)}, "
              f"3+={np.sum(ray_num>=3)}")

        # --- Per-pair measurement error (load pair data from disk) ---
        me_maps_local = {}  # nb_iid → (me_x, me_y)
        for stem2, pf_path in pair_files.get(stem1, {}).items():
            pd = np.load(pf_path)
            disp        = pd["disp_rect"]
            energy      = pd["energy_rect"]
            R_self_wc   = pd["R_self_wc"]
            R_nb_wc     = pd["R_nb_wc"]
            H_ori2epi   = pd["H_ori2epi"]
            K_rect_self = pd["K_rect_self"]
            K_rect_nb   = pd["K_rect_nb"]
            C_self      = pd["C_self"]
            C_nb        = pd["C_nb"]
            nb_iid      = int(pd["nb_iid"])

            hr, wr = disp.shape

            H_epi2ori = np.linalg.inv(H_ori2epi)
            mr_x = np.arange(wr, dtype=np.float32)[None,:]*np.ones((hr,1),np.float32)
            mr_y = np.arange(hr, dtype=np.float32)[:,None]*np.ones((1,wr),np.float32)
            ones = np.ones((hr, wr), np.float32)
            pts_h = np.stack([mr_x, mr_y, ones], -1).reshape(-1,3).T
            po   = H_epi2ori @ pts_h
            po  /= po[2:3]
            ox   = po[0].reshape(hr, wr).astype(np.float32)
            oy   = po[1].reshape(hr, wr).astype(np.float32)

            def orig_to_epi_remap(src):
                return cv2.remap(src, ox, oy, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

            gX_epi  = orig_to_epi_remap(gX)
            gY_epi  = orig_to_epi_remap(gY)
            gZ_epi  = orig_to_epi_remap(gZ)
            rn_epi  = cv2.remap(ray_num.astype(np.float32), ox, oy, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.uint16)

            reproj_epi = np.full((hr, wr, 4), np.nan, np.float32)
            valid_mask = ~np.isnan(disp) & ~np.isnan(gX_epi)
            if valid_mask.any():
                X_world = np.stack([
                    gX_epi[valid_mask] + C_self[0],
                    gY_epi[valid_mask] + C_self[1],
                    gZ_epi[valid_mask] + C_self[2],
                ], axis=1)

                aa_self = cv2.Rodrigues(R_self_wc)[0].flatten()
                aa_nb   = cv2.Rodrigues(R_nb_wc)[0].flatten()
                cam_p_self = np.array([K_rect_self[0,0], K_rect_self[1,1],
                                       K_rect_self[0,2], K_rect_self[1,2],
                                       0., 0., 0., 0.])
                cam_p_nb = np.array([K_rect_nb[0,0], K_rect_nb[1,1],
                                     K_rect_nb[0,2], K_rect_nb[1,2],
                                     0., 0., 0., 0.])

                vv_idx, uu_idx = np.where(valid_mask)
                u_nb, v_nb = _project_opencv_batch(X_world, aa_nb, C_nb, cam_p_nb)
                reproj_epi[vv_idx, uu_idx, 0] = ((disp[vv_idx, uu_idx] + uu_idx) - u_nb).astype(np.float32)
                reproj_epi[vv_idx, uu_idx, 1] = (vv_idx - v_nb).astype(np.float32)

                u_self, v_self = _project_opencv_batch(X_world, aa_self, C_self, cam_p_self)
                reproj_epi[vv_idx, uu_idx, 2] = (uu_idx - u_self).astype(np.float32)
                reproj_epi[vv_idx, uu_idx, 3] = (vv_idx - v_self).astype(np.float32)

            me_x, me_y, _ = predict_measurement_error(
                disp, energy, rn_epi, reproj_epi, H_ori2epi, W, H,
                multi_ray=args.multi_rays, num_std=args.num_std,
            )
            me_maps_local[nb_iid] = (me_x, me_y)
            del pd

        # --- Error propagation (uses fused grid + ME maps) ---
        xyz_list = []
        cov_list = []
        n_written = 0

        # Collect sampled pixel coordinates
        vv_all = np.arange(0, H, si)
        uu_all = np.arange(0, W, si)
        vv_grid, uu_grid = np.meshgrid(vv_all, uu_all, indexing='ij')
        vv_flat = vv_grid.ravel()
        uu_flat = uu_grid.ravel()

        # Filter to valid (non-NaN) depth pixels with enough rays
        valid_mask = ~np.isnan(gX[vv_flat, uu_flat]) & (ray_num[vv_flat, uu_flat] >= args.min_rays)
        vv_valid = vv_flat[valid_mask]
        uu_valid = uu_flat[valid_mask]
        n_valid = len(vv_valid)

        if n_valid > 0:
            X_world_all = np.stack([
                gX[vv_valid, uu_valid] + C1[0],
                gY[vv_valid, uu_valid] + C1[1],
                gZ[vv_valid, uu_valid] + C1[2],
            ], axis=1)  # (N, 3)

            # Precompute per-camera constants
            nb_list = list(me_maps_local.keys())
            cam_data = {}  # iid -> (aa, C, cam_p)
            # Master camera
            img_m = images[iid]
            cam_m = cameras[img_m["cam_id"]]
            aa_m = aa_from_qvec(np.array(img_m["qvec"]))
            C_m = -(opk_to_R(*img_m["opk"]).T @ np.array(img_m["tvec"]))
            cp_m = cam_params_array(cam_m)
            cam_data[iid] = (aa_m, C_m, cp_m)
            for nb_iid in nb_list:
                img_nb = images[nb_iid]
                cam_nb = cameras[img_nb["cam_id"]]
                cam_data[nb_iid] = (
                    aa_from_qvec(np.array(img_nb["qvec"])),
                    -(opk_to_R(*img_nb["opk"]).T @ np.array(img_nb["tvec"])),
                    cam_params_array(cam_nb),
                )

            # Batch compute Jacobians for master: (N, 2, 3)
            J_master = _jac_point_batch(X_world_all, *cam_data[iid])

            # Batch compute Jacobians for each neighbor: (N, 2, 3)
            nb_jacs = {}
            nb_me = {}
            for nb_iid in nb_list:
                me_x_map, me_y_map = me_maps_local[nb_iid]
                mx_vals = me_x_map[vv_valid, uu_valid]
                my_vals = me_y_map[vv_valid, uu_valid]
                nb_me[nb_iid] = (mx_vals, my_vals)
                nb_jacs[nb_iid] = _jac_point_batch(X_world_all, *cam_data[nb_iid])

            # For each pixel, accumulate A = prior_inv*I + sum_obs(J^T W^{-1} J)
            # Master contribution: fixed 0.25 px measurement error
            # (master pixel is the query, not a match — its location is known
            # to sub-pixel precision; depth uncertainty is carried by neighbors)
            master_me = 0.25
            master_w_inv = 1.0 / (master_me**2)
            master_w_inv_x = np.full(n_valid, master_w_inv)
            master_w_inv_y = np.full(n_valid, master_w_inv)
            # J^T @ diag(wx, wy) @ J for master
            JtJ_master = (
                master_w_inv_x[:, None, None] * J_master[:, :1, :].transpose(0, 2, 1) @ J_master[:, :1, :] +
                master_w_inv_y[:, None, None] * J_master[:, 1:, :].transpose(0, 2, 1) @ J_master[:, 1:, :]
            )

            A_batch = np.tile(np.eye(3) * PRIOR_INV, (n_valid, 1, 1)) + JtJ_master

            # Track which pixels have at least 2 observations
            obs_count = np.ones(n_valid, dtype=np.int32)  # master always counts

            for nb_iid in nb_list:
                mx_vals, my_vals = nb_me[nb_iid]
                nb_valid = ~np.isnan(mx_vals) & ~np.isnan(my_vals)
                obs_count[nb_valid] += 1

                if nb_valid.any():
                    J_nb = nb_jacs[nb_iid]  # (N, 2, 3)
                    # Combine matching uncertainty with pixel precision floor
                    # via RSS (independent error sources add in variance)
                    mx_clamped = np.sqrt(mx_vals**2 + master_me**2)
                    my_clamped = np.sqrt(my_vals**2 + master_me**2)
                    # W_inv for this neighbor: diag(1/(mx^2 + 1e-6), 1/(my^2 + 1e-6))
                    w_inv_x = np.where(nb_valid, 1.0 / (mx_clamped**2 + 1e-6), 0.0)
                    w_inv_y = np.where(nb_valid, 1.0 / (my_clamped**2 + 1e-6), 0.0)
                    # J^T @ diag(wx, wy) @ J  for each pixel
                    # = wx * J[0,:]^T @ J[0,:] + wy * J[1,:]^T @ J[1,:]
                    JtWJ = (w_inv_x[:, None, None] *
                            np.einsum('ni,nj->nij', J_nb[:, 0, :], J_nb[:, 0, :]) +
                            w_inv_y[:, None, None] *
                            np.einsum('ni,nj->nij', J_nb[:, 1, :], J_nb[:, 1, :]))
                    A_batch += JtWJ

            # Diagnostic: measurement error and weight statistics
            for nb_iid in nb_list:
                mx_vals, my_vals = nb_me[nb_iid]
                v = ~np.isnan(mx_vals)
                if v.any():
                    nb_stem = Path(images[nb_iid]["name"]).stem
                    log.info(f"    nb {nb_stem}: me_x median={np.median(mx_vals[v]):.3f} "
                          f"me_y median={np.median(my_vals[v]):.3f} "
                          f"w_inv_x median={np.median(1.0/(mx_vals[v]**2+1e-6)):.1f} "
                          f"w_inv_y median={np.median(1.0/(my_vals[v]**2+1e-6)):.1f}")
            log.info(f"  master_w_inv_x median={np.median(master_w_inv_x):.1f} "
                     f"master_w_inv_y median={np.median(master_w_inv_y):.1f}")
            log.info(f"  Propagation {stem1}: {n_valid} depth pixels, "
                  f"{len(nb_list)} neighbors, "
                  f"obs_count: 1={np.sum(obs_count==1)}, 2={np.sum(obs_count==2)}, "
                  f"3+={np.sum(obs_count>=3)}")

            # Only invert where we have >= 2 observations
            enough_obs = obs_count >= 2
            if enough_obs.any():
                A_good = A_batch[enough_obs]
                try:
                    cov_batch = np.linalg.inv(A_good)  # (M, 3, 3)
                except np.linalg.LinAlgError:
                    # Fallback: per-pixel inversion
                    cov_batch = np.zeros_like(A_good)
                    for k in range(len(A_good)):
                        try:
                            cov_batch[k] = np.linalg.inv(A_good[k])
                        except np.linalg.LinAlgError:
                            cov_batch[k] = np.nan

                X_good = X_world_all[enough_obs]

                # Diagnostic: eigenvalue ratio (needle-ness)
                eigs = np.linalg.eigvalsh(cov_batch)  # (M, 3), sorted ascending
                ratios = eigs[:, 2] / (eigs[:, 0] + 1e-30)
                log.info(f"  Cov eigenvalue ratios (max/min): "
                      f"median={np.median(ratios):.1f}, "
                      f"p95={np.percentile(ratios, 95):.1f}")
                log.info(f"  Cov eigenvalues (median): "
                      f"[{np.median(eigs[:,0]):.2e}, {np.median(eigs[:,1]):.2e}, {np.median(eigs[:,2]):.2e}]")

                # Diagnose negative diagonals
                neg_diag = ((cov_batch[:, 0, 0] < 0) |
                            (cov_batch[:, 1, 1] < 0) |
                            (cov_batch[:, 2, 2] < 0))
                if neg_diag.any():
                    idx = np.where(neg_diag)[0][0]
                    log.debug(f"[DIAG] {neg_diag.sum()}/{len(cov_batch)} points have negative cov diagonals")
                    log.debug(f"[DIAG] Example A (normal matrix):\n{A_good[idx]}")
                    log.debug(f"[DIAG] Example cov (A^-1):\n{cov_batch[idx]}")
                    log.debug(f"[DIAG] A condition number: {np.linalg.cond(A_good[idx]):.2e}")
                    log.debug(f"[DIAG] A eigenvalues: {np.linalg.eigvalsh(A_good[idx])}")
                    log.debug(f"[DIAG] obs_count for this pixel: {obs_count[enough_obs][idx]}")
                    log.debug(f"[DIAG] X_world: {X_good[idx]}")

                # Filter out NaN entries
                valid_cov = ~np.isnan(cov_batch[:, 0, 0])
                X_good = X_good[valid_cov]
                cov_batch = cov_batch[valid_cov]

                # Track pixel coords for RGB extraction
                vv_cov = vv_valid[enough_obs][valid_cov]
                uu_cov = uu_valid[enough_obs][valid_cov]

                xyz_list = list(X_good)
                cov_list = list(cov_batch)
                n_written = len(xyz_list)

        # ------------------------------------------------------------------
        # Save per-image .npz and .las (only points with valid covariance)
        # ------------------------------------------------------------------
        if xyz_list:
            xyz_arr = np.array(xyz_list, dtype=np.float64)
            cov_arr = np.array(cov_list, dtype=np.float64)
            sigma = np.sqrt(np.sum(np.diagonal(cov_arr, axis1=1, axis2=2), axis=1))

            npz_path = out_dir / f"{stem1}_cov.npz"
            np.savez_compressed(npz_path,
                                xyz=xyz_arr, gt_cov=cov_arr, sigma=sigma,
                                cam_xyz=cam_xyz, cam_R=cam_R)
            n_written_total += n_written
            log.info(f"  Saved {n_written} anchor points → {npz_path}")

            # Save per-image LAS (only covariance-valid points)
            try:
                img_color = load_image(image_dir, img["name"], gray=False)
                if img_color.shape[:2] != (H, W):
                    img_color = cv2.resize(img_color, (W, H))
                img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                rgb = img_rgb[vv_cov, uu_cov]
            except Exception:
                rgb = None

            rn = ray_num[vv_cov, uu_cov]
            las_path = out_dir / f"{stem1}_fused.las"
            _save_las(las_path, xyz_arr, rgb=rgb, ray_num=rn)
            log.info(f"  Saved {len(xyz_arr)} points -> {las_path}")

            all_las_pts.append(xyz_arr)
            all_las_rgb.append(rgb if rgb is not None
                               else np.zeros((len(xyz_arr), 3), dtype=np.uint8))
            all_las_rn.append(rn)

            # Accumulate for merged npz
            all_xyz_list.append(xyz_arr)
            all_cov_list.append(cov_arr)


    # ------------------------------------------------------------------
    # 5. Fuse all images into combined outputs
    # ------------------------------------------------------------------
    log.info("[5/5] Fusing all images ...")

    # Fused LAS (voxel fusion of all per-image point clouds)
    if all_las_pts:
        fused_pts, fused_rgb, fused_rn, _ = fuse_point_clouds(
            all_las_pts, all_las_rgb, all_las_rn,
            voxel_size=args.voxel_size)
        fused_las_path = out_dir / "fused_all.las"
        _save_las(fused_las_path, fused_pts, rgb=fused_rgb, ray_num=fused_rn)
        log.info(f"  Saved {fused_pts.shape[0]} fused points → {fused_las_path}")

    # Fused NPZ (voxel fusion with precision-weighted covariance)
    if all_xyz_list:
        # Build dummy rgb/rn for the covariance points (needed by fuse_point_clouds)
        cov_rgb_list = [np.zeros((len(x), 3), np.uint8) for x in all_xyz_list]
        cov_rn_list  = [np.ones(len(x), np.uint16) for x in all_xyz_list]

        fused_xyz, _, _, fused_cov = fuse_point_clouds(
            all_xyz_list, cov_rgb_list, cov_rn_list,
            xyz_cov_list=all_cov_list,
            voxel_size=args.voxel_size)

        # Filter out failed fusions (NaN covariance)
        valid = ~np.any(np.isnan(fused_cov.reshape(-1, 9)), axis=1)
        fused_xyz = fused_xyz[valid]
        fused_cov = fused_cov[valid]

        fused_sigma = np.sqrt(np.sum(
            np.diagonal(fused_cov, axis1=1, axis2=2), axis=1))

        fused_npz_path = out_dir / "fused_all_cov.npz"
        np.savez_compressed(fused_npz_path,
                            xyz=fused_xyz, gt_cov=fused_cov,
                            sigma=fused_sigma,
                            cam_xyz=cam_xyz, cam_R=cam_R)
        log.info(f"  Saved {len(fused_xyz)} fused anchor points → {fused_npz_path}")

    log.info(f"\nDone.  {n_written_total} total anchor points across all images.")


if __name__ == "__main__":
    main()
