"""
Epipolar rectification from COLMAP poses.

Implements the same rectification as the C++ pipeline:
1. Compute R_rect with x-axis aligned to baseline, using average optical axis
2. Determine common image size from max of per-image bounding box ranges
3. Center each warped image, share cy between images
4. Build homographies H = K_rect @ R_rect @ R_orig^T @ K_orig^{-1}
"""

import numpy as np
from scipy.spatial.transform import Rotation
import struct


# ============================================================
# COLMAP I/O helpers
# ============================================================

def qvec2rotmat(qvec):
    """COLMAP quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    return Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()


def read_cameras_binary(path):
    cameras = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack('<I', f.read(4))[0]
            model = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12}[model]
            params = struct.unpack('<' + 'd' * num_params, f.read(8 * num_params))
            cameras[cam_id] = {
                'model': model, 'w': w, 'h': h, 'params': params
            }
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack('<I', f.read(4))[0]
            qvec = struct.unpack('<dddd', f.read(32))
            tvec = struct.unpack('<ddd', f.read(24))
            cam_id = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch
            num_pts = struct.unpack('<Q', f.read(8))[0]
            f.read(num_pts * 24)
            images[img_id] = {
                'name': name.decode(), 'cam_id': cam_id,
                'qvec': qvec, 'tvec': tvec
            }
    return images


# ============================================================
# Core rectification
# ============================================================

def compute_rectifying_rotation(R_master, t_master, R_neighbor, t_neighbor):
    """
    Compute the shared rectifying rotation R_rect for a stereo pair.

    R_rect is constructed so that:
      - Row 0 (new x-axis) = baseline direction (C_neighbor - C_master)
      - Row 2 (new z-axis) = projection of average optical axis onto the
        plane perpendicular to the baseline
      - Row 1 = cross(row2, row0) to complete a right-handed frame

    Parameters
    ----------
    R_master, R_neighbor : (3,3) arrays
        COLMAP world-to-camera rotation matrices.
    t_master, t_neighbor : (3,) arrays
        COLMAP translation vectors (t = -R @ C).

    Returns
    -------
    R_rect : (3,3) array
        New world-to-camera rotation shared by both rectified images.
    """
    # Camera centers in world coordinates
    C1 = -R_master.T @ t_master
    C2 = -R_neighbor.T @ t_neighbor
    baseline = C2 - C1

    # New x-axis = baseline direction
    e1 = baseline / np.linalg.norm(baseline)

    # Average optical axis of both cameras (third row = z-axis in world)
    z_avg = (R_master[2, :] + R_neighbor[2, :]) / 2.0
    z_avg /= np.linalg.norm(z_avg)

    # New y-axis = perpendicular to both baseline and average z
    e2 = np.cross(z_avg, e1)
    e2 /= np.linalg.norm(e2)

    # New z-axis = completes the right-handed frame
    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3)

    return np.array([e1, e2, e3])


def rectify_stereo_pair(K_orig, R_master, t_master, R_neighbor, t_neighbor,
                        img_w, img_h):
    """
    Compute the full epipolar rectification for a stereo pair.

    All output homographies and intrinsics use 0-indexed pixel coordinates
    (pixel center at integer coords), matching the C++ pipeline convention.
    COLMAP's K_orig (pixel center at 0.5) is converted internally.

    To project a 3D point into rectified coordinates:
        K0 = K_orig.copy(); K0[0,2] -= 0.5; K0[1,2] -= 0.5
        p = K0 @ (R @ P + t)          # 0-indexed original pixel
        r = H_ori2epi @ [p[0]/p[2], p[1]/p[2], 1]  # rectified pixel

    Parameters
    ----------
    K_orig : (3,3) array
        Original camera intrinsic matrix (COLMAP convention, pixel center
        at 0.5). Shared by both images.
    R_master, R_neighbor : (3,3) arrays
        COLMAP world-to-camera rotation matrices.
    t_master, t_neighbor : (3,) arrays
        COLMAP translation vectors (t = -R @ C).
    img_w, img_h : int
        Original image dimensions (pixels).

    Returns
    -------
    dict with keys:
        R_rect      : (3,3)  shared rectifying rotation
        K1_rect     : (3,3)  rectified intrinsics for master  (0-indexed)
        K2_rect     : (3,3)  rectified intrinsics for neighbor (0-indexed)
        rect_w      : int    common rectified image width
        rect_h      : int    common rectified image height
        H1_ori2epi  : (3,3)  homography original -> rectified (master)
        H2_ori2epi  : (3,3)  homography original -> rectified (neighbor)
        H1_epi2ori  : (3,3)  homography rectified -> original (master)
        H2_epi2ori  : (3,3)  homography rectified -> original (neighbor)
    """
    R_rect = compute_rectifying_rotation(
        R_master, t_master, R_neighbor, t_neighbor
    )

    fx, fy = K_orig[0, 0], K_orig[1, 1]
    # Use 0-indexed pixel convention (pixel center at integer coords)
    # to match the C++ pipeline. COLMAP stores cx/cy with pixel center
    # at (0.5, 0.5); subtract 0.5 to convert.
    cx0 = K_orig[0, 2] - 0.5
    cy0 = K_orig[1, 2] - 0.5
    K0 = np.array([[fx, 0, cx0], [0, fy, cy0], [0, 0, 1]])
    K0_inv = np.linalg.inv(K0)

    # --- Raw homographies (using 0-indexed intrinsics) ---
    H1_raw = K0 @ R_rect @ R_master.T @ K0_inv
    H2_raw = K0 @ R_rect @ R_neighbor.T @ K0_inv

    # --- Warp original image corners to find rectified bounding boxes ---
    corners = np.array([
        [0, img_w - 1, 0, img_w - 1],
        [0, 0, img_h - 1, img_h - 1],
        [1, 1, 1, 1]
    ], dtype=float)

    def _warp_pts(H, pts):
        w = H @ pts
        return w[:2] / w[2:3]

    c1 = _warp_pts(H1_raw, corners)
    c2 = _warp_pts(H2_raw, corners)

    # Per-image bounding box ranges
    x_range_1 = c1[0].max() - c1[0].min()
    x_range_2 = c2[0].max() - c2[0].min()
    y_range_1 = c1[1].max() - c1[1].min()
    y_range_2 = c2[1].max() - c2[1].min()

    # Common rectified image size = ceiling of max range
    rect_w = int(np.ceil(max(x_range_1, x_range_2)))
    rect_h = int(np.ceil(max(y_range_1, y_range_2)))

    # --- Principal points ---
    # Center each image's warped bounding box at the rectified image center.
    # cx is per-image; cy is shared (average of both warped y-centers).
    x_center_1 = (c1[0].min() + c1[0].max()) / 2.0
    x_center_2 = (c2[0].min() + c2[0].max()) / 2.0
    y_center_1 = (c1[1].min() + c1[1].max()) / 2.0
    y_center_2 = (c2[1].min() + c2[1].max()) / 2.0

    rect_cx = (rect_w - 1) / 2.0   # target center of rectified image
    rect_cy = (rect_h - 1) / 2.0

    cx1_rect = cx0 + (rect_cx - x_center_1)
    cx2_rect = cx0 + (rect_cx - x_center_2)
    y_center_avg = (y_center_1 + y_center_2) / 2.0
    cy_rect = cy0 + (rect_cy - y_center_avg)

    K1_rect = np.array([[fx, 0, cx1_rect],
                         [0, fy, cy_rect],
                         [0, 0, 1]])
    K2_rect = np.array([[fx, 0, cx2_rect],
                         [0, fy, cy_rect],
                         [0, 0, 1]])

    # --- Final homographies ---
    H1_ori2epi = K1_rect @ R_rect @ R_master.T @ K0_inv
    H2_ori2epi = K2_rect @ R_rect @ R_neighbor.T @ K0_inv
    H1_epi2ori = np.linalg.inv(H1_ori2epi)
    H2_epi2ori = np.linalg.inv(H2_ori2epi)

    return {
        'R_rect': R_rect,
        'K1_rect': K1_rect,
        'K2_rect': K2_rect,
        'rect_w': rect_w,
        'rect_h': rect_h,
        'H1_ori2epi': H1_ori2epi,
        'H2_ori2epi': H2_ori2epi,
        'H1_epi2ori': H1_epi2ori,
        'H2_epi2ori': H2_epi2ori,
    }


# ============================================================
# Verification against C++ results
# ============================================================

def parse_epi_pose(path):
    with open(path) as f:
        lines = f.readlines()
    K = np.array([[float(x) for x in lines[i].split()] for i in range(1, 4)])
    params = lines[6].split()
    imgw, imgh = int(params[5]), int(params[6])
    R = np.array([[float(x) for x in lines[i].split()] for i in range(8, 11)])
    return K, R, imgw, imgh


def parse_transmatrix(path):
    with open(path) as f:
        lines = f.readlines()
    def p3(s):
        return np.array([[float(x) for x in lines[s + i].split()] for i in range(3)])
    return p3(1), p3(5), p3(9), p3(13)


def verify_all_pairs(sparse_dir, cpp_dir):
    """Compare our rectification against all C++ results."""
    import os

    cameras = read_cameras_binary(f'{sparse_dir}/cameras.bin')
    images = read_images_binary(f'{sparse_dir}/images.bin')
    cam = cameras[list(cameras.keys())[0]]
    fx, fy, cx, cy = cam['params'][:4]
    K_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    name2img = {im['name']: im for im in images.values()}

    pose1_files = sorted([
        f for f in os.listdir(cpp_dir) if f.endswith('_EPI_pose1.txt')
    ])

    # Auto-detect master stem from file names
    first_prefix = pose1_files[0].replace('_EPI_pose1.txt', '')
    master_stem = None
    for name in name2img:
        stem = name.replace('.jpg', '')
        if first_prefix.startswith(stem + '_'):
            master_stem = stem
            break
    if master_stem is None:
        raise RuntimeError("Could not detect master image stem from filenames")

    print(f"Dataset: {sparse_dir}")
    print(f"Camera: model={cam['model']}, {cam['w']}x{cam['h']}, "
          f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"Images: {len(images)}, Master: {master_stem}, Pairs: {len(pose1_files)}")

    header = (f"{'Pair':<20} {'R_err':>8} {'cx1_err':>8} {'cx2_err':>8} "
              f"{'cy_err':>8} {'W_err':>6} {'H_err':>6} "
              f"{'H1_err':>8} {'H2_err':>8}")
    print(f"\n{header}")
    print("-" * len(header))

    for pf in pose1_files:
        prefix = pf.replace('_EPI_pose1.txt', '')
        neighbor_stem = prefix[len(master_stem) + 1:]
        master_name = master_stem + '.jpg'
        neighbor_name = neighbor_stem + '.jpg'
        short = neighbor_name.split('_')[1] if '_' in neighbor_name \
            else neighbor_name[:15]

        # C++ reference
        K1_ref, R_ref, w_ref, h_ref = parse_epi_pose(
            os.path.join(cpp_dir, prefix + '_EPI_pose1.txt'))
        K2_ref, _, _, _ = parse_epi_pose(
            os.path.join(cpp_dir, prefix + '_EPI_pose2.txt'))
        _, _, H_o2e1_ref, H_o2e2_ref = parse_transmatrix(
            os.path.join(cpp_dir, prefix + '_Epi2Ori_Ori2Epi_TransMatrix.txt'))

        # COLMAP poses
        im_m = name2img[master_name]
        im_n = name2img[neighbor_name]
        R_m = qvec2rotmat(im_m['qvec'])
        t_m = np.array(im_m['tvec'])
        R_n = qvec2rotmat(im_n['qvec'])
        t_n = np.array(im_n['tvec'])

        # Our rectification
        res = rectify_stereo_pair(
            K_orig, R_m, t_m, R_n, t_n, cam['w'], cam['h']
        )

        # Comparisons
        R_err = np.max(np.abs(res['R_rect'] - R_ref))
        cx1_err = res['K1_rect'][0, 2] - K1_ref[0, 2]
        cx2_err = res['K2_rect'][0, 2] - K2_ref[0, 2]
        cy_err = res['K1_rect'][1, 2] - K1_ref[1, 2]
        W_err = res['rect_w'] - w_ref
        H_err = res['rect_h'] - h_ref

        def norm_H(H):
            return H / H[2, 2]

        H1_err = np.max(np.abs(
            norm_H(res['H1_ori2epi']) - norm_H(H_o2e1_ref)))
        H2_err = np.max(np.abs(
            norm_H(res['H2_ori2epi']) - norm_H(H_o2e2_ref)))

        print(f"{short:<20} {R_err:>8.1e} {cx1_err:>8.3f} {cx2_err:>8.3f} "
              f"{cy_err:>8.3f} {W_err:>6d} {H_err:>6d} "
              f"{H1_err:>8.4f} {H2_err:>8.4f}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        base = sys.argv[1]
        sparse_dir = f'{base}/sparse'
        cpp_dir = f'{base}/c++'
    else:
        sparse_dir = 'UseGeo/tmp/sparse'
        cpp_dir = 'UseGeo/tmp/c++'
    verify_all_pairs(sparse_dir, cpp_dir)
