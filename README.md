# UncertaintyQuantification

Per-point covariance estimation for aerial / UAV photogrammetric point clouds via stereo matching and Gauss–Markov error propagation.

This repository is the **Python re-implementation** of the MVS-with-uncertainty framework presented in:

> **Uncertainty Quantification Framework for Aerial and UAV Photogrammetry through Error Propagation**
> *ISPRS Journal of Photogrammetry and Remote Sensing* (accepted, 2026)
> Preprint: <https://arxiv.org/abs/2507.13486>
> *(this README will be updated with the official journal link once published)*

The original paper used a C++ stack — **MSP** (an in-house MVS package built on Semi-Global Matching) coupled with the bespoke uncertainty propagation framework. This repository replaces MSP with an **OpenCV SGBM**-based MVS implementation in pure Python, and re-implements the same uncertainty propagation framework on top of it. We therefore expect the numerical results produced here to be **close to, but not bit-exact with**, the figures reported in the paper.

The pipeline takes a COLMAP sparse reconstruction plus the source images and produces, for every kept pixel, a 3×3 world-frame covariance matrix carried alongside its 3D point.

---

## Pipeline overview

For each reference image and its co-visible neighbors:

1. **Read COLMAP sparse model** — `cameras.bin`, `images.bin`, `points3D.bin`.
2. **Select stereo pairs** from the co-visibility graph (or by SIFT matching with `--sift_graph`).
3. **Run OpenCV SGBM stereo** — disparity map + SGM energy proxy.
4. **Fuse depth maps** into per-image XYZ grids with multi-view geometric consistency.
5. **Predict per-pixel measurement error** (energy-calibrated reprojection σ).
6. **Propagate measurement error** to 3D-point covariance via Gauss–Markov:
   `Cov(X) = (prior_inv·I + Σ Jᵀ W J)⁻¹`
7. **Write outputs** — per-image `*_cov.npz` (xyz, 3×3 cov, σ), `*_fused.las`, and a fused `fused_all_cov.npz` / `fused_all.las`.

A separate script, `compute_sensor_error_prop.py`, implements the full bundle-adjustment-covariance variant (USfM-NBUP / Schur-complement form) used for the sensor-error baseline in the paper.

---

## Installation

```bash
git clone https://github.com/GDAOSU/UncertaintyQuantification.git
cd UncertaintyQuantification

# Option A: conda
conda create -n uncertainty python=3.10 -y
conda activate uncertainty
pip install -r requirements.txt

# Option B: venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required packages:

| Package | Notes |
|---------|-------|
| numpy, scipy | core math |
| opencv-python | SGBM stereo, image I/O |
| laspy | LAS point cloud output |
| torch | only used by `--sift_graph` (GPU brute-force SIFT matching) |

A CUDA-capable GPU is optional but recommended for `--sift_graph` on large datasets.

---

## Repo layout

```
UncertaintyQuantification/
├── scripts/
│   ├── run_mvs_pipeline.py            # main entrypoint (steps 1–7 above)
│   ├── compute_sensor_error_prop.py   # BA-covariance baseline
│   ├── rectify.py                     # stereo rectification helpers
│   └── colmap_io.py                   # COLMAP binary readers (Schönberger)
├── examples/
│   ├── Dortmund/sparse/               # COLMAP poses only — see "Examples"
│   └── UseGeo/Dataset-{1,2,3}/sparse/
├── requirements.txt
└── LICENSE                            # Apache 2.0
```

The `examples/` folders contain only the COLMAP `sparse/` triplet (`cameras.bin`, `images.bin`, `points3D.bin`). Source images must be downloaded separately — see below.

---

## Examples

### 1. Dortmund (oblique aerial, ISPRS / NeRFBK)

Download the Dortmund images from the FBK NeRFBK benchmark:

- https://github.com/3DOM-FBK/NeRFBK

Place the images in `examples/Dortmund/images/` so the layout is:

```
examples/Dortmund/
├── sparse/         # provided
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── images/         # you download
    ├── DJI_0001.JPG
    └── ...
```

Run:

```bash
python scripts/run_mvs_pipeline.py \
    --scene  examples/Dortmund/sparse \
    --images examples/Dortmund/images \
    --out    examples/Dortmund/out    \
    --scale  0.25  \
    --workers 8    \
    --sift_graph
```

### 2. UseGeo (FBK-3DOM aerial benchmark)

Download the UseGeo images from:

- https://github.com/3DOM-FBK/UseGeo

We ship sparse poses for `Dataset-1`, `Dataset-2`, and `Dataset-3`. Place each set of images in the matching folder:

```
examples/UseGeo/Dataset-1/
├── sparse/         # provided
└── images/         # you download
```

Run on Dataset-1 (full resolution):

```bash
python scripts/run_mvs_pipeline.py \
    --scene  examples/UseGeo/Dataset-1/sparse  \
    --images examples/UseGeo/Dataset-1/images  \
    --out    examples/UseGeo/Dataset-1/out     \
    --scale  1.0  \
    --workers 8   \
    --sift_graph
```

Repeat with `Dataset-2` / `Dataset-3` for the other two scenes.

---

## Outputs

In the `--out` directory:

| File | Contents |
|------|----------|
| `<image>_cov.npz` | per-image anchor points: `xyz` (N,3), `gt_cov` (N,3,3), `sigma` (N,), `cam_xyz`, `cam_R` |
| `<image>_fused.las` | per-image fused colored point cloud |
| `fused_all_cov.npz` | voxel-fused dataset-wide covariance point cloud |
| `fused_all.las` | voxel-fused colored point cloud (all images) |
| `mvs_pipeline_<timestamp>.log` | full DEBUG log for the run |

---

## Common flags

| Flag | Default | Effect |
|------|---------|--------|
| `--scale`  | 1.0 | downsample factor for images (e.g. 0.25 = quarter-res) |
| `--workers` | 8 | parallel SGM workers — reduce for large images / low RAM |
| `--max_neighbors` | 10 | stereo neighbors per reference image |
| `--sift_graph` | off | use SIFT matching instead of co-visibility for neighbor selection |
| `--voxel_size` | 0.05 | edge length for final voxel fusion (in scene units) |

See `python scripts/run_mvs_pipeline.py --help` for the full list.

---

## Citation

If you use this code, please cite:

```bibtex
@article{huang2026uncertainty,
  title   = {Uncertainty Quantification Framework for Aerial and UAV Photogrammetry through Error Propagation},
  author  = {Huang, Debao and others},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2026},
  note    = {Preprint: arXiv:2507.13486}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

`scripts/colmap_io.py` is derived from the COLMAP `read_write_model.py` script
(© ETH Zürich / UNC Chapel Hill, BSD 3-clause); see the header in that file.
