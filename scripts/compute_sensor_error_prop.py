#!/usr/bin/env python3
"""
Sensor model error propagation for a single COLMAP block.

Computes per-3D-point covariance matrices by propagating camera-parameter
uncertainty through the reprojection Jacobian (Gauss–Markoff law):

    Cov(X) = (prior_inv·I  +  Bx^T · W · Bx)^{-1}

where
    Bx  = ∂(u,v)/∂X          (2N×3) Jacobian w.r.t. ground point
    W   = (A · Cs · A^T)^{-1}  inverse of propagated camera-param covariance
    A   = ∂(u,v)/∂θ           (2N×14N) Jacobian w.r.t. camera params
    Cs  = camera-param covariance (full Schur complement from BA normal equations)

Camera-param covariance matches ErrorProp::ComputeCovarianceCameraParams (USfM NBUP):
  - Builds the full H_cc and H_cp blocks of the BA normal equations
  - Computes 7-DOF SfM gauge nullspace H (Algorithm::computeJacobianNullspace)
  - Extended Schur complement: MsH=[M H;H^T 0] → Zs=(n_c+7)×(n_c+7)
  - Variance factor (mirrors USfM InputCovariance_VarianceFactor):
      vf = SSR / (2*n_obs - n_params),  n_params = n_imgs*6 + n_cams*ncp + n_pts*3
  - Sigma_cam = vf * inv(Zs)[:n_c, :n_c]

Jacobians (jac_cam_params, jac_point) mirror ErrorProp::ComputeJacobianCameraParams /
ComputeJacobianGroundPoint via the same Rodrigues chain rule.  The C++ uses
auto-expanded CSE for speed; this implementation uses factored chain-rule which
is mathematically identical.

Camera parameter layout (matches ErrorProp.hpp):
  image params (per image):  aa[3] + c[3]          (6 params)
  camera params (per model): OPENCV: fx fy cx cy k1 k2 p1 p2 (8)
                             FULL_OPENCV: fx fy cx cy k1 k2 p1 p2 k3 (9)

Usage:
    python compute_sensor_error_prop.py --block data/Dataset-1/block_0000
    python compute_sensor_error_prop.py --block data/Dataset-1/block_0000 --out_json out.json
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_model, qvec2rotmat

# ── constants ────────────────────────────────────────────────────────────────
NUM_IMG_PARAMS = 6    # aa[3] + c[3]
PRIOR_INV      = 1e-6

# Camera model definitions: model_name -> num_cam_params
CAMERA_MODELS = {
    "OPENCV":      8,   # fx fy cx cy k1 k2 p1 p2
    "FULL_OPENCV": 9,   # fx fy cx cy k1 k2 p1 p2 k3  (k4-k6 ignored)
}


def _detect_cam_model(cameras):
    """Detect camera model and return model name string."""
    models = set(cam.model for cam in cameras.values())
    if len(models) != 1:
        raise ValueError(f"Mixed camera models not supported: {models}")
    model = models.pop()
    if model not in CAMERA_MODELS:
        raise ValueError(f"Unsupported camera model: {model}. "
                         f"Supported: {list(CAMERA_MODELS.keys())}")
    return model


def _get_cam_params(camera, num_cam_params):
    """Extract camera params, taking first num_cam_params from COLMAP params."""
    return camera.params[:num_cam_params].copy()


# ── geometry helpers ─────────────────────────────────────────────────────────

def aa_from_qvec(qvec: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z] → angle-axis."""
    R     = qvec2rotmat(qvec)
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-9:
        return np.zeros(3)
    skew = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    return skew * (angle / (2.0 * np.sin(angle)))


def camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP convention: t = -R·c  →  c = -R^T·t."""
    return -qvec2rotmat(qvec).T @ tvec


# ── vectorized COLMAP OPENCV projection ─────────────────────────────────────

def project_opencv_batch(X, aa, c, cam_p, cam_model):
    """
    Batched projection.  All inputs (N, ...).
    X: (N,3), aa: (N,3), c: (N,3), cam_p: (N, num_cam_params).
    cam_model: "OPENCV" or "FULL_OPENCV".
    Returns (N, 2) pixel coords.
    """
    fx = cam_p[:, 0]; fy = cam_p[:, 1]; cx = cam_p[:, 2]; cy = cam_p[:, 3]
    k1 = cam_p[:, 4]; k2 = cam_p[:, 5]; p1 = cam_p[:, 6]; p2 = cam_p[:, 7]

    d = X - c                                      # (N, 3)
    theta2 = np.sum(aa * aa, axis=1)               # (N,)
    theta  = np.sqrt(np.maximum(theta2, 1e-30))    # (N,)
    axis   = aa / theta[:, None]                    # (N, 3)
    cos_t  = np.cos(theta)                          # (N,)
    sin_t  = np.sin(theta)                          # (N,)
    ad     = np.sum(axis * d, axis=1)               # (N,)
    axd    = np.cross(axis, d)                      # (N, 3)

    small = theta2 < np.finfo(float).eps
    Xc = cos_t[:, None] * d + (1.0 - cos_t[:, None]) * ad[:, None] * axis + sin_t[:, None] * axd
    if np.any(small):
        aaxd = np.cross(aa[small], d[small])
        Xc[small] = d[small] + aaxd

    xp = Xc[:, 0] / Xc[:, 2]
    yp = Xc[:, 1] / Xc[:, 2]
    r2  = xp * xp + yp * yp
    r4  = r2 * r2
    if cam_model == "FULL_OPENCV":
        k3  = cam_p[:, 8]
        r6  = r4 * r2
        rad = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    else:  # OPENCV
        rad = 1.0 + k1 * r2 + k2 * r4
    u = fx * (xp * rad + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx
    v = fy * (yp * rad + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp) + cy
    return np.stack([u, v], axis=1)


# ── vectorized Jacobians ────────────────────────────────────────────────────

def _rodrigues_jac_batch(aa, d):
    """
    Batched Rodrigues rotation + Jacobian.
    aa: (N,3), d: (N,3).
    Returns Xc: (N,3), R: (N,3,3), dp_daa: (N,3,3).
    """
    N = aa.shape[0]
    theta2 = np.sum(aa * aa, axis=1)                     # (N,)
    theta  = np.sqrt(np.maximum(theta2, 1e-30))           # (N,)
    n      = aa / theta[:, None]                          # (N, 3)
    cos_t  = np.cos(theta)                                # (N,)
    sin_t  = np.sin(theta)                                # (N,)
    one_cos = 1.0 - cos_t                                 # (N,)

    # R = cos*I + (1-cos)*n⊗n + sin*[n]×
    I3 = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    nn = n[:, :, None] * n[:, None, :]                    # (N, 3, 3)
    # skew(n): (N, 3, 3)
    skew_n = np.zeros((N, 3, 3))
    skew_n[:, 0, 1] = -n[:, 2]; skew_n[:, 0, 2] =  n[:, 1]
    skew_n[:, 1, 0] =  n[:, 2]; skew_n[:, 1, 2] = -n[:, 0]
    skew_n[:, 2, 0] = -n[:, 1]; skew_n[:, 2, 1] =  n[:, 0]

    R = (cos_t[:, None, None] * I3
         + one_cos[:, None, None] * nn
         + sin_t[:, None, None] * skew_n)                 # (N, 3, 3)
    Xc = np.einsum('nij,nj->ni', R, d)                    # (N, 3)

    # dp_daa = outer(dp_dtheta, n) + dp_dn @ dn_daa
    nd = np.sum(n * d, axis=1)                             # (N,)
    nxd = np.cross(n, d)                                   # (N, 3)
    dp_dtheta = (-sin_t[:, None] * d
                 + cos_t[:, None] * nxd
                 + sin_t[:, None] * nd[:, None] * n)       # (N, 3)

    # skew(d): (N, 3, 3)
    skew_d = np.zeros((N, 3, 3))
    skew_d[:, 0, 1] = -d[:, 2]; skew_d[:, 0, 2] =  d[:, 1]
    skew_d[:, 1, 0] =  d[:, 2]; skew_d[:, 1, 2] = -d[:, 0]
    skew_d[:, 2, 0] = -d[:, 1]; skew_d[:, 2, 1] =  d[:, 0]

    # dp_dn = -sin*[d]× + (1-cos)*(nd*I + n⊗d)
    nd_I = nd[:, None, None] * I3                          # (N, 3, 3)
    n_outer_d = n[:, :, None] * d[:, None, :]              # (N, 3, 3)
    dp_dn = (-sin_t[:, None, None] * skew_d
             + one_cos[:, None, None] * (nd_I + n_outer_d))  # (N, 3, 3)

    # dn_daa = (I - n⊗n) / theta
    dn_daa = (I3 - nn) / theta[:, None, None]              # (N, 3, 3)

    dp_daa = (dp_dtheta[:, :, None] * n[:, None, :]        # outer(dp_dtheta, n)
              + np.einsum('nij,njk->nik', dp_dn, dn_daa))  # (N, 3, 3)

    # Handle small angles: R ≈ I + [aa]×, dp_daa = -[d]×
    small = theta2 < np.finfo(float).eps
    if np.any(small):
        skew_aa = np.zeros((int(np.sum(small)), 3, 3))
        aa_s = aa[small]
        skew_aa[:, 0, 1] = -aa_s[:, 2]; skew_aa[:, 0, 2] =  aa_s[:, 1]
        skew_aa[:, 1, 0] =  aa_s[:, 2]; skew_aa[:, 1, 2] = -aa_s[:, 0]
        skew_aa[:, 2, 0] = -aa_s[:, 1]; skew_aa[:, 2, 1] =  aa_s[:, 0]
        R_small = I3[:int(np.sum(small))] + skew_aa
        R[small] = R_small
        Xc[small] = np.einsum('nij,nj->ni', R_small, d[small])
        dp_daa[small] = -skew_d[small]

    return Xc, R, dp_daa


def jac_batch(X, aa, c, cam_p, cam_model):
    """
    Batched Jacobians for all observations at once.
    X: (N,3), aa: (N,3), c: (N,3), cam_p: (N, num_cam_params).
    cam_model: "OPENCV" or "FULL_OPENCV".
    Returns:
        Jcam: (N, 2, 6+num_cam_params)
        Jpt:  (N, 2, 3)  — ∂(u,v)/∂(X,Y,Z)
        uv:   (N, 2)     — projected pixel coords
    """
    num_cam_params = CAMERA_MODELS[cam_model]
    N = X.shape[0]
    fx = cam_p[:, 0]; fy = cam_p[:, 1]
    k1 = cam_p[:, 4]; k2 = cam_p[:, 5]
    p1 = cam_p[:, 6]; p2 = cam_p[:, 7]

    d = X - c                                            # (N, 3)
    Xc, R, dp_daa = _rodrigues_jac_batch(aa, d)          # (N,3), (N,3,3), (N,3,3)

    Zc = Xc[:, 2]                                        # (N,)
    xp = Xc[:, 0] / Zc                                   # (N,)
    yp = Xc[:, 1] / Zc                                   # (N,)

    # ── Distortion + Jacobians ──
    r2  = xp * xp + yp * yp
    r4  = r2 * r2
    if cam_model == "FULL_OPENCV":
        k3   = cam_p[:, 8]
        r6   = r4 * r2
        rad  = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        drad = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
    else:  # OPENCV
        rad  = 1.0 + k1 * r2 + k2 * r4
        drad = k1 + 2.0 * k2 * r2

    xd = xp * rad + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)
    yd = yp * rad + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp

    # J_dist (N, 2, 2)
    dxd_dxp = rad + 2.0 * xp * xp * drad + 2.0 * p1 * yp + 6.0 * p2 * xp
    dxd_dyp = 2.0 * xp * yp * drad + 2.0 * p1 * xp + 2.0 * p2 * yp
    dyd_dyp = rad + 2.0 * yp * yp * drad + 6.0 * p1 * yp + 2.0 * p2 * xp
    J_dist = np.zeros((N, 2, 2))
    J_dist[:, 0, 0] = dxd_dxp
    J_dist[:, 0, 1] = dxd_dyp
    J_dist[:, 1, 0] = dxd_dyp  # symmetric off-diagonal
    J_dist[:, 1, 1] = dyd_dyp

    # J_dist_params: (N, 2, n_dist) — distortion param Jacobians
    if cam_model == "FULL_OPENCV":
        n_dist = 5  # k1, k2, p1, p2, k3
        J_dist_params = np.zeros((N, 2, n_dist))
        J_dist_params[:, 0, 4] = xp * r6
        J_dist_params[:, 1, 4] = yp * r6
    else:  # OPENCV
        n_dist = 4  # k1, k2, p1, p2
        J_dist_params = np.zeros((N, 2, n_dist))
    J_dist_params[:, 0, 0] = xp * r2
    J_dist_params[:, 0, 1] = xp * r4
    J_dist_params[:, 0, 2] = 2.0 * xp * yp
    J_dist_params[:, 0, 3] = r2 + 2.0 * xp * xp
    J_dist_params[:, 1, 0] = yp * r2
    J_dist_params[:, 1, 1] = yp * r4
    J_dist_params[:, 1, 2] = r2 + 2.0 * yp * yp
    J_dist_params[:, 1, 3] = 2.0 * xp * yp

    # J_f (N, 2, 2) diagonal
    J_f = np.zeros((N, 2, 2))
    J_f[:, 0, 0] = fx
    J_f[:, 1, 1] = fy

    # J_persp (N, 2, 3)
    inv_Zc = 1.0 / Zc
    J_persp = np.zeros((N, 2, 3))
    J_persp[:, 0, 0] = inv_Zc
    J_persp[:, 0, 2] = -xp * inv_Zc
    J_persp[:, 1, 1] = inv_Zc
    J_persp[:, 1, 2] = -yp * inv_Zc

    # J_base = J_f @ J_dist @ J_persp  (N, 2, 3)
    J_fd = np.einsum('nij,njk->nik', J_f, J_dist)        # (N, 2, 2)
    J_base = np.einsum('nij,njk->nik', J_fd, J_persp)    # (N, 2, 3)

    # ── Jcam (N, 2, 6+num_cam_params) ──
    n_jcam = NUM_IMG_PARAMS + num_cam_params
    Jcam = np.zeros((N, 2, n_jcam))
    # J_aa = J_base @ dp_daa  (N, 2, 3)
    Jcam[:, :, :3] = np.einsum('nij,njk->nik', J_base, dp_daa)
    # J_c = J_base @ (-R)  (N, 2, 3)
    Jcam[:, :, 3:6] = np.einsum('nij,njk->nik', J_base, -R)
    # J_intr: (N, 2, 4) w.r.t. fx, fy, cx, cy
    Jcam[:, 0, 6] = xd
    Jcam[:, 1, 7] = yd
    Jcam[:, 0, 8] = 1.0
    Jcam[:, 1, 9] = 1.0
    # J_kp = J_f @ J_dist_params  (N, 2, n_dist)
    Jcam[:, :, 10:10+n_dist] = np.einsum('nij,njk->nik', J_f, J_dist_params)

    # ── Jpt = J_base @ R  (N, 2, 3) ──
    Jpt = np.einsum('nij,njk->nik', J_base, R)

    # ── projected coords ──
    u = fx * xd + cam_p[:, 2]
    v = fy * yd + cam_p[:, 3]
    uv = np.stack([u, v], axis=1)

    return Jcam, Jpt, uv


# ── SfM gauge nullspace (mirrors Algorithm::computeJacobianNullspace) ─────────

def _compute_jacobian_nullspace(images, points3D, obs_index,
                                pt_ids_sorted, img_ids, cam_ids,
                                img_params, cam_params,
                                jac_cache, num_cam_params):
    """
    Compute the 7-column SfM gauge nullspace H of the full BA Jacobian.
    Uses precomputed Jacobians from jac_cache.
    """
    n_pts  = len(pt_ids_sorted)
    n_imgs = len(img_ids)
    n_cams = len(cam_ids)
    pts_off  = 0
    imgs_off = n_pts * 3
    cams_off = n_pts * 3 + n_imgs * 6
    n_total  = cams_off + n_cams * num_cam_params

    pt_rank  = {pid: i for i, pid in enumerate(pt_ids_sorted)}
    img_rank = {iid: i for i, iid in enumerate(img_ids)}
    cam_rank = {cid: j for j, cid in enumerate(cam_ids)}

    H = np.zeros((n_total, 7))

    # ── Fill known rows: points ──
    for pt_id in pt_ids_sorted:
        X = points3D[pt_id].xyz
        b = pts_off + pt_rank[pt_id] * 3
        H[b:b+3, :] = [
            [1, 0, 0,  0,    -X[2],  X[1], X[0]],
            [0, 1, 0,  X[2],  0,    -X[0], X[1]],
            [0, 0, 1, -X[1],  X[0],  0,    X[2]],
        ]

    # ── Fill known rows: camera centres ──
    for img_id in img_ids:
        _, C = img_params[img_id]
        b = imgs_off + img_rank[img_id] * 6 + 3
        H[b:b+3, :] = [
            [1, 0, 0,  0,    -C[2],  C[1], C[0]],
            [0, 1, 0,  C[2],  0,    -C[0], C[1]],
            [0, 0, 1, -C[1],  C[0],  0,    C[2]],
        ]

    # ── Solve for aa rows via J·H = 0 (sparse — no dense rows) ──
    # Each row of the full BA Jacobian is mostly zeros; the only nonzero
    # blocks are Jpt (3 entries at the point offset), Jcam_img (6 at the
    # image offset), and Jcam_intrinsics (ncp at the camera offset).
    # The dot product  row @ H[:, 3:6]  decomposes into contributions from
    # those sparse blocks.  Camera-intrinsic rows of H are zero, so only
    # the point and image-centre blocks contribute:
    #   dot = Jpt_row @ H_pt_rot  +  Jcam_c_row @ H_c_rot
    # where H_pt_rot = skew(X) and H_c_rot = skew(C).
    # This avoids allocating any length-n_total arrays.

    Jdiag   = np.zeros((3 * n_imgs, 3 * n_imgs))
    Jrows_H = np.zeros((3 * n_imgs, 3))

    for img_id in img_ids:
        ri      = img_rank[img_id]
        obs     = obs_index[img_id]

        # Precompute H rotation block for this image's centre
        _, C = img_params[img_id]
        H_c_rot = np.array([[ 0,   -C[2],  C[1]],
                            [ C[2],  0,   -C[0]],
                            [-C[1],  C[0],  0   ]])

        # For each observation, compute the sparse dot product and
        # cache just what's needed for row selection (Jpt, Jcam_aa, dot).
        obs_data = []   # list of (Jpt_row, Jcam_aa_row, dot_3vec)
        for local_k, (pt_id, _) in enumerate(obs):
            cache_idx = jac_cache['img_obs_to_flat'][(img_id, local_k)]
            Jcam_k = jac_cache['Jcam'][cache_idx]   # (2, 6+ncp)
            Jpt_k  = jac_cache['Jpt'][cache_idx]     # (2, 3)

            X = points3D[pt_id].xyz
            H_pt_rot = np.array([[ 0,    -X[2],  X[1]],
                                 [ X[2],  0,    -X[0]],
                                 [-X[1],  X[0],  0   ]])

            for ell in range(2):
                Jpt_row  = Jpt_k[ell]
                Jcam_aa  = Jcam_k[ell, :3]
                Jcam_c   = Jcam_k[ell, 3:6]
                dot = Jpt_row @ H_pt_rot + Jcam_c @ H_c_rot
                obs_data.append((Jpt_row, Jcam_aa, dot))

        # Select 3 rows matching C++ heuristic
        first_Jpt = obs_data[0][0]

        r2_idx = None
        k  = 2
        while k < len(obs_data):
            cand_Jpt = obs_data[k][0]
            if any(abs(first_Jpt[j] - cand_Jpt[j]) > abs(first_Jpt[j] * 0.005)
                   for j in range(3)):
                r2_idx = k
                break
            k += 2
        if r2_idx is None:
            r2_idx = 2 if len(obs_data) > 2 else 1

        base = ri * 3
        for k_out, sel_idx in enumerate([0, 1, r2_idx]):
            _, Jcam_aa, dot = obs_data[sel_idx]
            Jdiag[base + k_out, ri*3:ri*3+3] = Jcam_aa
            Jrows_H[base + k_out]            = dot

    invJdiag = np.zeros_like(Jdiag)
    for ri in range(n_imgs):
        b = ri * 3
        invJdiag[b:b+3, b:b+3] = np.linalg.inv(Jdiag[b:b+3, b:b+3])

    H_rot = -(invJdiag @ Jrows_H)

    for img_id in img_ids:
        ri = img_rank[img_id]
        b  = imgs_off + ri * 6
        H[b:b+3, 3:6] = H_rot[ri*3:ri*3+3, :]

    return H


# ── BA covariance via full Schur complement (mirrors USfM NBUP) ───────────────

def compute_camera_covariance_schur(cameras, images, points3D, obs_index,
                                    jac_cache, cam_model):
    """
    Camera-parameter covariance via the full Schur complement of the BA normal
    equations, using precomputed Jacobians from jac_cache.
    """
    num_cam_params = CAMERA_MODELS[cam_model]
    img_ids = sorted(images.keys())
    cam_ids = sorted(cameras.keys())
    pt_ids  = sorted(points3D.keys())

    img_offset = {iid: i * NUM_IMG_PARAMS for i, iid in enumerate(img_ids)}
    cam_offset = {cid: len(img_ids) * NUM_IMG_PARAMS + j * num_cam_params
                  for j, cid in enumerate(cam_ids)}

    img_params = {iid: (aa_from_qvec(images[iid].qvec),
                        camera_center(images[iid].qvec, images[iid].tvec))
                  for iid in img_ids}
    cam_params = {cid: cameras[cid].params[:num_cam_params].copy() for cid in cam_ids}

    n_c = len(img_ids) * NUM_IMG_PARAMS + len(cam_ids) * num_cam_params

    H_cc = np.zeros((n_c, n_c))
    H_cp = {}
    H_pp = {}

    # Use cached Jacobians + precomputed residuals
    all_Jcam = jac_cache['Jcam']      # (N_obs, 2, 15)
    all_Jpt  = jac_cache['Jpt']       # (N_obs, 2, 3)
    all_uv   = jac_cache['uv']        # (N_obs, 2)
    all_xy   = jac_cache['xy_obs']    # (N_obs, 2)

    total_sq = 0.0
    n_obs    = 0

    for img_id, obs_list in obs_index.items():
        cam_id = images[img_id].camera_id
        ic     = img_offset[img_id]
        cc_    = cam_offset[cam_id]
        n_obs_img = len(obs_list)

        # Gather flat indices for this image's observations
        flat_indices = np.array([jac_cache['img_obs_to_flat'][(img_id, k)]
                                 for k in range(n_obs_img)])
        pt_ids_obs = [obs_list[k][0] for k in range(n_obs_img)]

        Ji_all  = all_Jcam[flat_indices, :, :NUM_IMG_PARAMS]   # (n, 2, 6)
        Jk_all  = all_Jcam[flat_indices, :, NUM_IMG_PARAMS:]   # (n, 2, ncp)
        Jpt_all = all_Jpt[flat_indices]                         # (n, 2, 3)

        # H_cc: vectorized sums over all obs in this image
        H_cc[ic:ic+NUM_IMG_PARAMS, ic:ic+NUM_IMG_PARAMS] += \
            np.einsum('nij,nik->jk', Ji_all, Ji_all)
        H_cc[cc_:cc_+num_cam_params, cc_:cc_+num_cam_params] += \
            np.einsum('nij,nik->jk', Jk_all, Jk_all)
        blk_sum = np.einsum('nij,nik->jk', Ji_all, Jk_all)
        H_cc[ic:ic+NUM_IMG_PARAMS, cc_:cc_+num_cam_params] += blk_sum
        H_cc[cc_:cc_+num_cam_params, ic:ic+NUM_IMG_PARAMS] += blk_sum.T

        # Precompute per-obs products for H_cp / H_pp
        JiTJpt_all  = np.einsum('nij,nik->njk', Ji_all, Jpt_all)   # (n, 6, 3)
        JkTJpt_all  = np.einsum('nij,nik->njk', Jk_all, Jpt_all)   # (n, ncp, 3)
        JptTJpt_all = np.einsum('nij,nik->njk', Jpt_all, Jpt_all)  # (n, 3, 3)

        for k in range(n_obs_img):
            pt_id = pt_ids_obs[k]
            if pt_id not in H_cp:
                H_cp[pt_id] = np.zeros((n_c, 3))
                H_pp[pt_id] = np.zeros((3, 3))
            H_cp[pt_id][ic:ic+NUM_IMG_PARAMS, :] += JiTJpt_all[k]
            H_cp[pt_id][cc_:cc_+num_cam_params, :] += JkTJpt_all[k]
            H_pp[pt_id] += JptTJpt_all[k]

        # Residuals: vectorized
        res = np.array([obs_list[k][1] for k in range(n_obs_img)]) - all_uv[flat_indices]
        total_sq += float(np.sum(res * res))
        n_obs += n_obs_img

    n_params = (len(img_ids) * NUM_IMG_PARAMS
                + len(cam_ids) * num_cam_params
                + len(points3D) * 3)
    dof    = max(1, 2 * n_obs - n_params)
    sigma2 = total_sq / dof

    # Robust variance: recompute excluding observations with residual > threshold.
    # Large residuals come from points projecting near/behind the camera plane
    # (Zc ≈ 0), which are data-quality issues, not model errors.
    RESID_THRESH = 50.0  # px
    all_res = all_xy - all_uv                         # (N_total, 2)
    res_sq  = np.sum(all_res ** 2, axis=1)            # (N_total,)
    inlier  = res_sq < RESID_THRESH ** 2
    n_inlier = int(np.sum(inlier))
    if n_inlier > n_params:
        total_sq_robust = float(np.sum(res_sq[inlier]))
        dof_robust = max(1, 2 * n_inlier - n_params)
        sigma2 = total_sq_robust / dof_robust
        dof    = dof_robust
        n_outlier = len(res_sq) - n_inlier
        print(f"    Filtered {n_outlier} outlier observations (residual > {RESID_THRESH} px)")
    print(f"    sigma_reprojection = {np.sqrt(sigma2):.4f} px  (sigma2 = {sigma2:.4e}, dof = {dof})")

    # ── Gauge nullspace ──
    num_cam_params = CAMERA_MODELS[cam_model]
    H_null = _compute_jacobian_nullspace(
        images, points3D, obs_index,
        pt_ids, img_ids, cam_ids, img_params, cam_params,
        jac_cache, num_cam_params)

    imgs_off_null = len(pt_ids) * 3
    Hc     = H_null[imgs_off_null:, :]
    H_pts  = H_null[:imgs_off_null, :]
    pt_rank = {pid: i for i, pid in enumerate(pt_ids)}

    n_ext = n_c + 7
    Zs = np.zeros((n_ext, n_ext))
    Zs[:n_c, :n_c] = H_cc
    Zs[:n_c, n_c:] = Hc
    Zs[n_c:, :n_c] = Hc.T

    # Vectorized Schur complement: batch all point contributions
    valid_pts = [pid for pid in pt_ids if pid in H_pp]
    CHUNK = 10000
    eye3_reg = np.eye(3) * 1e-10
    for start in range(0, len(valid_pts), CHUNK):
        chunk = valid_pts[start:start+CHUNK]
        n_chunk = len(chunk)
        Hpp_batch = np.empty((n_chunk, 3, 3))
        Bs_batch  = np.empty((n_chunk, 3, n_ext))
        for i, pt_id in enumerate(chunk):
            Hpp_batch[i] = H_pp[pt_id] + eye3_reg
            r = pt_rank[pt_id]
            Bs_batch[i, :, :n_c] = H_cp[pt_id].T
            Bs_batch[i, :, n_c:] = H_pts[r*3:r*3+3, :]
        # Batched solve: (n,3,3) \ (n,3,n_ext) → (n,3,n_ext)
        inv_Hpp_Bs = np.linalg.solve(Hpp_batch, Bs_batch)
        # Sum all rank-3 updates at once
        Zs -= np.einsum('pji,pjk->ik', Bs_batch, inv_Hpp_Bs)

    iZs       = np.linalg.inv(Zs)
    Sigma_cam = sigma2 * iZs[:n_c, :n_c]

    return Sigma_cam, img_offset, cam_offset, img_params, cam_params


# ── per-point covariance ─────────────────────────────────────────────────────

def cov_point_sensor(X, obs_list, Sigma_cam, img_offset, cam_offset,
                     img_params, cam_params, images, jac_cache, pt_id,
                     cam_model):
    """
    Mirrors ErrorProp::ComputeCovariancePoint_SensorModel.
    Uses precomputed Jacobians from jac_cache.
    """
    num_cam_params = CAMERA_MODELS[cam_model]
    N     = len(obs_list)
    n_col = N * (NUM_IMG_PARAMS + num_cam_params)

    Bx = np.zeros((2 * N, 3))
    A  = np.zeros((2 * N, n_col))
    Cs = np.zeros((n_col, n_col))

    obs_info = []
    for k, (img_id, _) in enumerate(obs_list):
        cam_id   = images[img_id].camera_id
        ic       = img_offset[img_id]
        cc_      = cam_offset[cam_id]

        # Look up from pt_obs cache
        cache_idx = jac_cache['pt_obs_to_flat'].get((pt_id, img_id))
        if cache_idx is not None:
            Jcam = jac_cache['Jcam'][cache_idx]
            Jpt  = jac_cache['Jpt'][cache_idx]
        else:
            # Fallback: compute (should not happen if cache is built correctly)
            aa, c = img_params[img_id]
            cp    = cam_params[cam_id]
            Jcam  = _jac_cam_params_single(X, aa, c, cp, cam_model)
            Jpt   = _jac_point_single(X, aa, c, cp, cam_model)

        row = k * 2
        n_per = NUM_IMG_PARAMS + num_cam_params
        col = k * n_per
        Bx[row:row+2, :]          = Jpt
        A[row:row+2, col:col+n_per] = Jcam
        obs_info.append((ic, cc_))

    n_per = NUM_IMG_PARAMS + num_cam_params
    for k1, (ic1, cc1) in enumerate(obs_info):
        c1 = k1 * n_per
        for k2, (ic2, cc2) in enumerate(obs_info):
            c2 = k2 * n_per
            Cs[c1:c1+NUM_IMG_PARAMS, c2:c2+NUM_IMG_PARAMS] = \
                Sigma_cam[ic1:ic1+NUM_IMG_PARAMS, ic2:ic2+NUM_IMG_PARAMS]
            Cs[c1+NUM_IMG_PARAMS:c1+n_per, c2+NUM_IMG_PARAMS:c2+n_per] = \
                Sigma_cam[cc1:cc1+num_cam_params, cc2:cc2+num_cam_params]
            Cs[c1:c1+NUM_IMG_PARAMS, c2+NUM_IMG_PARAMS:c2+n_per] = \
                Sigma_cam[ic1:ic1+NUM_IMG_PARAMS, cc2:cc2+num_cam_params]
            Cs[c1+NUM_IMG_PARAMS:c1+n_per, c2:c2+NUM_IMG_PARAMS] = \
                Sigma_cam[cc1:cc1+num_cam_params, ic2:ic2+NUM_IMG_PARAMS]

    ACA = A @ Cs @ A.T
    W   = np.linalg.inv(ACA)
    Cov = np.linalg.inv(PRIOR_INV * np.eye(3) + Bx.T @ W @ Bx)
    return Cov


# ── scalar fallbacks for edge cases ─────────────────────────────────────────

def _skew(v):
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

def _jac_cam_params_single(X, aa, c, cam_p, cam_model):
    fx, fy = cam_p[0], cam_p[1]
    k1, k2, p1, p2 = cam_p[4], cam_p[5], cam_p[6], cam_p[7]
    d = X - c
    theta2 = aa @ aa
    if theta2 > np.finfo(float).eps:
        theta = np.sqrt(theta2)
        n = aa / theta
        cos_t = np.cos(theta); sin_t = np.sin(theta); one_cos = 1.0 - cos_t
        R = cos_t * np.eye(3) + one_cos * np.outer(n, n) + sin_t * _skew(n)
        Xc = R @ d
        nd = float(n @ d); nxd = np.cross(n, d)
        dp_dtheta = -sin_t * d + cos_t * nxd + sin_t * nd * n
        dp_dn = -sin_t * _skew(d) + one_cos * (nd * np.eye(3) + np.outer(n, d))
        dn_daa = (np.eye(3) - np.outer(n, n)) / theta
        dp_daa = np.outer(dp_dtheta, n) + dp_dn @ dn_daa
    else:
        R = np.eye(3) + _skew(aa); Xc = R @ d; dp_daa = -_skew(d)
    Zc = Xc[2]; xp = Xc[0]/Zc; yp = Xc[1]/Zc
    r2 = xp*xp + yp*yp; r4 = r2*r2
    if cam_model == "FULL_OPENCV":
        k3 = cam_p[8]; r6 = r4*r2
        rad = 1.0 + k1*r2 + k2*r4 + k3*r6; drad = k1 + 2.0*k2*r2 + 3.0*k3*r4
    else:  # OPENCV
        rad = 1.0 + k1*r2 + k2*r4; drad = k1 + 2.0*k2*r2
    xd = xp*rad + 2.0*p1*xp*yp + p2*(r2+2.0*xp*xp)
    yd = yp*rad + p1*(r2+2.0*yp*yp) + 2.0*p2*xp*yp
    J_dist = np.array([[rad+2*xp*xp*drad+2*p1*yp+6*p2*xp, 2*xp*yp*drad+2*p1*xp+2*p2*yp],
                       [2*xp*yp*drad+2*p1*xp+2*p2*yp, rad+2*yp*yp*drad+6*p1*yp+2*p2*xp]])
    J_f = np.array([[fx,0],[0,fy]]); inv_Zc = 1.0/Zc
    J_persp = np.array([[inv_Zc,0,-xp*inv_Zc],[0,inv_Zc,-yp*inv_Zc]])
    J_base = J_f @ J_dist @ J_persp
    if cam_model == "FULL_OPENCV":
        J_dist_params = np.array([[xp*r2,xp*r4,2*xp*yp,r2+2*xp*xp,xp*r6],
                                  [yp*r2,yp*r4,r2+2*yp*yp,2*xp*yp,yp*r6]])
    else:  # OPENCV
        J_dist_params = np.array([[xp*r2,xp*r4,2*xp*yp,r2+2*xp*xp],
                                  [yp*r2,yp*r4,r2+2*yp*yp,2*xp*yp]])
    return np.hstack([J_base @ dp_daa, J_base @ (-R),
                      np.array([[xd,0,1,0],[0,yd,0,1]]), J_f @ J_dist_params])

def _jac_point_single(X, aa, c, cam_p, cam_model):
    fx, fy = cam_p[0], cam_p[1]
    k1, k2, p1, p2 = cam_p[4], cam_p[5], cam_p[6], cam_p[7]
    d = X - c
    theta2 = aa @ aa
    if theta2 > np.finfo(float).eps:
        theta = np.sqrt(theta2); n = aa/theta
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        R = cos_t*np.eye(3)+(1-cos_t)*np.outer(n,n)+sin_t*_skew(n)
        Xc = R @ d
    else:
        R = np.eye(3)+_skew(aa); Xc = R @ d
    Zc=Xc[2]; xp=Xc[0]/Zc; yp=Xc[1]/Zc
    r2=xp*xp+yp*yp; r4=r2*r2
    if cam_model == "FULL_OPENCV":
        k3=cam_p[8]; r6=r4*r2
        rad=1+k1*r2+k2*r4+k3*r6; drad=k1+2*k2*r2+3*k3*r4
    else:  # OPENCV
        rad=1+k1*r2+k2*r4; drad=k1+2*k2*r2
    J_dist = np.array([[rad+2*xp*xp*drad+2*p1*yp+6*p2*xp,2*xp*yp*drad+2*p1*xp+2*p2*yp],
                       [2*xp*yp*drad+2*p1*xp+2*p2*yp,rad+2*yp*yp*drad+6*p1*yp+2*p2*xp]])
    J_f=np.array([[fx,0],[0,fy]]); inv_Zc=1.0/Zc
    J_persp=np.array([[inv_Zc,0,-xp*inv_Zc],[0,inv_Zc,-yp*inv_Zc]])
    return J_f @ J_dist @ J_persp @ R


# ── precompute all Jacobians ─────────────────────────────────────────────────

def precompute_jacobians(images, cameras, points3D, obs_index, pt_obs,
                         cam_model):
    """
    Compute all Jacobians in one vectorized batch.
    Returns jac_cache dict with:
      Jcam:  (N_obs, 2, NUM_IMG_PARAMS + num_cam_params)
      Jpt:   (N_obs, 2, 3)
      uv:    (N_obs, 2)
      xy_obs: (N_obs, 2)
      img_obs_to_flat: {(img_id, local_k) -> flat_idx}
      pt_obs_to_flat:  {(pt_id, img_id) -> flat_idx}
    """
    num_cam_params = CAMERA_MODELS[cam_model]
    # Precompute per-image params
    img_params = {}
    for iid in images:
        img_params[iid] = (aa_from_qvec(images[iid].qvec),
                           camera_center(images[iid].qvec, images[iid].tvec))
    cam_params = {cid: cameras[cid].params[:num_cam_params].copy() for cid in cameras}

    # Build flat observation arrays
    flat_X    = []
    flat_aa   = []
    flat_c    = []
    flat_cp   = []
    flat_xy   = []
    img_obs_to_flat = {}
    pt_obs_to_flat  = {}
    flat_pt_ids = []
    flat_img_ids = []

    flat_idx = 0
    for img_id, obs_list in obs_index.items():
        aa, c   = img_params[img_id]
        cam_id  = images[img_id].camera_id
        cp      = cam_params[cam_id]
        for local_k, (pt_id, xy) in enumerate(obs_list):
            flat_X.append(points3D[pt_id].xyz)
            flat_aa.append(aa)
            flat_c.append(c)
            flat_cp.append(cp)
            flat_xy.append(xy)
            img_obs_to_flat[(img_id, local_k)] = flat_idx
            pt_obs_to_flat[(pt_id, img_id)] = flat_idx
            flat_pt_ids.append(pt_id)
            flat_img_ids.append(img_id)
            flat_idx += 1

    N = flat_idx
    print(f"  Precomputing {N} Jacobians (vectorized) …")

    flat_X  = np.array(flat_X)       # (N, 3)
    flat_aa = np.array(flat_aa)      # (N, 3)
    flat_c  = np.array(flat_c)       # (N, 3)
    flat_cp = np.array(flat_cp)      # (N, 9)
    flat_xy = np.array(flat_xy)      # (N, 2)

    Jcam, Jpt, uv = jac_batch(flat_X, flat_aa, flat_c, flat_cp, cam_model)

    # ── Filter out bad observations at the root ──
    # Points projecting behind the camera (Zc ≈ 0) produce garbage Jacobians
    # and huge residuals. Remove them from obs_index and pt_obs so all
    # downstream code (Schur, nullspace, per-point cov) never sees them.
    RESID_THRESH = 50.0  # px
    res_norms = np.linalg.norm(flat_xy - uv, axis=1)
    bad_mask = res_norms > RESID_THRESH
    n_bad = int(np.sum(bad_mask))
    if n_bad > 0:
        print(f"  Filtering {n_bad} bad observations (residual > {RESID_THRESH} px)")
        bad_set = set(np.where(bad_mask)[0].tolist())

        # Rebuild obs_index: remove bad observations
        new_obs_index = {}
        new_img_obs_to_flat = {}
        for img_id, obs_list in obs_index.items():
            new_list = []
            for local_k, (pt_id, xy) in enumerate(obs_list):
                flat_idx_k = img_obs_to_flat.get((img_id, local_k))
                if flat_idx_k is not None and flat_idx_k not in bad_set:
                    new_local_k = len(new_list)
                    new_list.append((pt_id, xy))
                    new_img_obs_to_flat[(img_id, new_local_k)] = flat_idx_k
            if new_list:
                new_obs_index[img_id] = new_list
        obs_index.clear()
        obs_index.update(new_obs_index)
        img_obs_to_flat = new_img_obs_to_flat

        # Remove images that lost all observations
        dropped_imgs = set(images.keys()) - set(obs_index.keys())
        for iid in dropped_imgs:
            del images[iid]
        if dropped_imgs:
            print(f"    Dropped {len(dropped_imgs)} images with no remaining observations")

        # Rebuild pt_obs: remove bad observations
        new_pt_obs = {}
        for pt_id, obs_list in pt_obs.items():
            new_list = []
            for img_id, xy in obs_list:
                flat_idx_k = pt_obs_to_flat.get((pt_id, img_id))
                if flat_idx_k is not None and flat_idx_k not in bad_set:
                    new_list.append((img_id, xy))
            if new_list:
                new_pt_obs[pt_id] = new_list
        pt_obs.clear()
        pt_obs.update(new_pt_obs)

    return {
        'Jcam': Jcam,
        'Jpt':  Jpt,
        'uv':   uv,
        'xy_obs': flat_xy,
        'img_obs_to_flat': img_obs_to_flat,
        'pt_obs_to_flat':  pt_obs_to_flat,
        'flat_pt_ids': flat_pt_ids,
        'flat_img_ids': flat_img_ids,
    }


# ── process one block ────────────────────────────────────────────────────────

def process_block(block_dir: Path, min_rays: int = 2):
    sparse_dir = block_dir / "sparse"
    # Auto-detect text vs binary format
    if (sparse_dir / "cameras.txt").exists():
        ext = ".txt"
    elif (sparse_dir / "cameras.bin").exists():
        ext = ".bin"
    else:
        raise FileNotFoundError(f"No cameras.txt or cameras.bin in {sparse_dir}")
    cameras, images, points3D = read_model(str(sparse_dir), ext=ext)

    # Detect camera model
    cam_model = _detect_cam_model(cameras)
    num_cam_params = CAMERA_MODELS[cam_model]
    print(f"  Camera model: {cam_model} ({num_cam_params} params)")

    # build observation index
    obs_index = {iid: [] for iid in images}
    for img_id, img in images.items():
        valid = img.point3D_ids > 0
        for pt_id, xy in zip(img.point3D_ids[valid], img.xys[valid]):
            obs_index[img_id].append((int(pt_id), xy))

    pt_obs = {}
    for img_id, obs_list in obs_index.items():
        for pt_id, xy in obs_list:
            pt_obs.setdefault(pt_id, []).append((img_id, xy))

    # Deduplicate
    n_dedup = 0
    for pt_id in pt_obs:
        seen = {}
        unique = []
        for img_id, xy in pt_obs[pt_id]:
            if img_id not in seen:
                seen[img_id] = True
                unique.append((img_id, xy))
        if len(unique) < len(pt_obs[pt_id]):
            n_dedup += 1
            pt_obs[pt_id] = unique
    if n_dedup:
        print(f"  Deduplicated observations for {n_dedup} points")

    print(f"  {len(images)} images, {len(cameras)} cameras, {len(points3D)} points")

    # ── Precompute all Jacobians once ──
    jac_cache = precompute_jacobians(images, cameras, points3D, obs_index, pt_obs,
                                     cam_model)

    print("  Computing BA camera covariance (Schur complement) …")
    Sigma_cam, img_offset, cam_offset, img_params, cam_params = \
        compute_camera_covariance_schur(cameras, images, points3D, obs_index,
                                        jac_cache, cam_model)

    print("  Propagating uncertainty to each 3D point …")
    xyz_parts, cov_parts = [], []
    n_skip = 0

    # Group points by ray count for batched processing
    ray_groups = {}
    for pt_id, obs in pt_obs.items():
        nr = len(obs)
        if nr < min_rays:
            n_skip += 1
            continue
        ray_groups.setdefault(nr, []).append((pt_id, obs))

    num_cam_params = CAMERA_MODELS[cam_model]
    n_per = NUM_IMG_PARAMS + num_cam_params

    for n_rays, group in sorted(ray_groups.items()):
        M = len(group)
        n_col  = n_rays * n_per
        n_rows = 2 * n_rays

        # Pre-gather offsets and cache indices
        cache_idx_arr = np.empty((M, n_rays), dtype=np.intp)
        ic_arr = np.empty((M, n_rays), dtype=np.intp)
        cc_arr = np.empty((M, n_rays), dtype=np.intp)
        xyz_arr = np.empty((M, 3))

        for m, (pt_id, obs_list) in enumerate(group):
            xyz_arr[m] = points3D[pt_id].xyz
            for k, (img_id, _) in enumerate(obs_list):
                cache_idx_arr[m, k] = jac_cache['pt_obs_to_flat'][(pt_id, img_id)]
                ic_arr[m, k] = img_offset[img_id]
                cc_arr[m, k] = cam_offset[images[img_id].camera_id]

        # Gather Jacobians: (M, n_rays, 2, *)
        Jcam_g = jac_cache['Jcam'][cache_idx_arr]   # (M, n_rays, 2, n_per)
        Jpt_g  = jac_cache['Jpt'][cache_idx_arr]     # (M, n_rays, 2, 3)

        # Process in chunks to limit memory (~500 MB cap for Cs)
        CHUNK = max(1, min(20000, 500_000_000 // (n_col * n_col * 8)))

        for start in range(0, M, CHUNK):
            end = min(start + CHUNK, M)
            mc = end - start

            # Bx: (mc, 2*n_rays, 3)
            Bx = Jpt_g[start:end].reshape(mc, n_rows, 3)

            # A: (mc, 2*n_rays, n_col) — block-diagonal
            A = np.zeros((mc, n_rows, n_col))
            for k in range(n_rays):
                A[:, 2*k:2*k+2, k*n_per:(k+1)*n_per] = Jcam_g[start:end, k]

            # Sigma_cam index: (mc, n_col) — row/col indices into Sigma_cam
            idx = np.empty((mc, n_col), dtype=np.intp)
            for k in range(n_rays):
                col = k * n_per
                idx[:, col:col+NUM_IMG_PARAMS] = (
                    ic_arr[start:end, k:k+1] + np.arange(NUM_IMG_PARAMS))
                idx[:, col+NUM_IMG_PARAMS:col+n_per] = (
                    cc_arr[start:end, k:k+1] + np.arange(num_cam_params))

            # Extract Cs: (mc, n_col, n_col)
            Cs = Sigma_cam[idx[:, :, None], idx[:, None, :]]

            # ACA = A @ Cs @ A^T: (mc, n_rows, n_rows)
            ACs = np.einsum('mij,mjk->mik', A, Cs)
            ACA = np.einsum('mij,mkj->mik', ACs, A)

            # W = inv(ACA), BtWBx = Bx^T @ W @ Bx + prior*I
            try:
                W = np.linalg.inv(ACA)
                WBx = np.einsum('mij,mjk->mik', W, Bx)
                BtWBx = np.einsum('mji,mjk->mik', Bx, WBx)
                BtWBx += PRIOR_INV * np.eye(3)
                Cov_batch = np.linalg.inv(BtWBx)

                valid = np.all(np.isfinite(
                    Cov_batch.reshape(mc, -1)), axis=1)
                xyz_parts.append(xyz_arr[start:end][valid])
                cov_parts.append(Cov_batch[valid])
                n_skip += int(mc - np.sum(valid))
            except np.linalg.LinAlgError:
                # Fallback: scalar per-point
                for m in range(mc):
                    try:
                        w = np.linalg.inv(ACA[m])
                        btw = Bx[m].T @ w @ Bx[m] + PRIOR_INV * np.eye(3)
                        c_ = np.linalg.inv(btw)
                        if np.all(np.isfinite(c_)):
                            xyz_parts.append(xyz_arr[start + m:start + m + 1])
                            cov_parts.append(c_[np.newaxis])
                        else:
                            n_skip += 1
                    except np.linalg.LinAlgError:
                        n_skip += 1

    n_computed = sum(len(x) for x in xyz_parts)
    print(f"  {n_computed} points computed, {n_skip} skipped")
    xyz = np.concatenate(xyz_parts, axis=0).astype(np.float64) if xyz_parts else np.empty((0, 3))
    cov = np.concatenate(cov_parts, axis=0).astype(np.float64) if cov_parts else np.empty((0, 3, 3))

    sorted_iids = sorted(images.keys())
    cam_xyz = np.array([img_params[iid][1]            for iid in sorted_iids], dtype=np.float64)
    cam_R   = np.array([qvec2rotmat(images[iid].qvec) for iid in sorted_iids], dtype=np.float64)
    return xyz, cov, cam_xyz, cam_R


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", required=True,
                        help="Path to block directory (contains sparse/)")
    parser.add_argument("--out_npz", default=None,
                        help="Save xyz+cov to .npz (default: <block>/sensor_cov.npz)")
    parser.add_argument("--out_json", default=None,
                        help="Also export JSON for the browser visualiser")
    parser.add_argument("--out_txt", default=None,
                        help="Also export C++-compatible txt: X Y Z Cov00..Cov22 (17 sig figs)")
    parser.add_argument("--min_rays", type=int, default=2)
    args = parser.parse_args()

    block_dir = Path(args.block)
    print(f"Block: {block_dir}")

    xyz, cov, cam_xyz, cam_R = process_block(block_dir, args.min_rays)

    npz_path = Path(args.out_npz) if args.out_npz else block_dir / "sensor_cov.npz"
    sigma = np.sqrt(np.sum(np.diagonal(cov, axis1=1, axis2=2), axis=1))
    np.savez_compressed(npz_path, xyz=xyz, gt_cov=cov, sigma=sigma,
                        cam_xyz=cam_xyz, cam_R=cam_R)
    print(f"  {len(cam_xyz)} camera poses saved")
    print(f"Saved: {npz_path}")

    if args.out_txt:
        txt_path = Path(args.out_txt)
        with open(txt_path, "w") as f:
            for i in range(len(xyz)):
                row = [*xyz[i]]
                for r in range(3):
                    for c in range(3):
                        row.append(cov[i, r, c])
                f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
        print(f"Saved: {txt_path}")

    if args.out_json:
        records = []
        for i in range(len(xyz)):
            vals, vecs = np.linalg.eigh(cov[i])
            vals = np.maximum(vals, 0.0)
            records.append({
                "xyz":    xyz[i].tolist(),
                "cov":    cov[i].tolist(),
                "eigval": vals.tolist(),
                "eigvec": vecs.T.tolist(),
            })
        json_path = Path(args.out_json)
        with open(json_path, "w") as f:
            json.dump(records, f, separators=(",", ":"))
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
