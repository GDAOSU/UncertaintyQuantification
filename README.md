# Uncertainty Quantification Framework for Aerial and UAV Photogrammetry through Error Propagation

Per-point covariance estimation for aerial / UAV photogrammetric point clouds. For every point in the dense reconstruction, the pipeline produces a 3×3 error covariance matrix that propagates uncertainty through the full two-step photogrammetry process — Structure-from-Motion / Bundle Adjustment (SfM/BA) and Multi-View Stereo (MVS / Dense Image Matching).

This repository is the **Python re-implementation** of the uncertainty quantification framework presented in:

> **Uncertainty Quantification Framework for Aerial and UAV Photogrammetry through Error Propagation**
> [Debao Huang](https://debaohuang.github.io/), [Rongjun Qin](https://u.osu.edu/qin.324/)
> *ISPRS Journal of Photogrammetry and Remote Sensing* (accepted, 2026)
> Preprint: <https://arxiv.org/abs/2507.13486> · PDF: [`paper/ISPRS_J_uncertainty.pdf`](paper/ISPRS_J_uncertainty.pdf)
> *(this README will be updated with the official journal link once published)*

The paper's experimental results were produced with the in-house C++ MVS package **MSP** (<https://u.osu.edu/qin.324/msp/>), which uses **Semi-Global Matching (SGM)** for dense matching. This repository ships an end-to-end Python pipeline that uses **OpenCV SGBM** for the dense matching step and re-implements the same uncertainty quantification framework on top of it. We therefore expect the numerical results produced here to be **close to, but not bit-exact with**, the figures reported in the paper.

The framework propagates uncertainty as

```
Σ_g = ( Σ_ε⁻¹ + B_Xᵀ ( A Σ_θ Aᵀ + Σ_disp )⁻¹ B_X )⁻¹
```

where Σ_θ is the camera-parameter covariance from BA, Σ_disp is the disparity-uncertainty covariance from MVS, and A, B_X are Jacobians of the projection function w.r.t. the camera parameters and the 3D point, respectively (paper Eq. 1).

---

## Pipeline overview

`scripts/run_mvs_pipeline.py` implements, for each reference image and a set of co-visible neighbors:

1. **Read COLMAP sparse model** — `cameras.bin`, `images.bin`, `points3D.bin`.
2. **Select stereo pairs** from the co-visibility graph (or by SIFT matching with `--sift_graph`).
3. **Pairwise dense matching** with **OpenCV SGBM** — disparity map + matching-cost proxy.
4. **Multi-view fusion** — fuse per-pair depth maps into per-reference XYZ grids with a multi-view geometric-consistency check; a 3D point is kept only if observed in ≥ 3 views.
5. **Disparity-uncertainty regression** — using **n-view points (n ≥ 6)** as pseudo-check points, regress per-pixel disparity uncertainty `u` from matching-cost cues (8 cost-σ bins, interpolated per-pixel), then refine with the reprojection residual.
6. **Propagation to 3D-point covariance** — propagate disparity uncertainty (Σ_disp) plus the SfM camera-parameter covariance (Σ_θ) to a 3×3 covariance per 3D point via Eq. 1 above.
7. **Write outputs** — per-image `*_cov.npz` (xyz, 3×3 cov, σ), `*_fused.las`, plus dataset-wide `fused_all_cov.npz` and `fused_all.las`.

A separate script, `scripts/compute_sensor_error_prop.py`, implements just the **SfM-stage** propagation (camera-parameter covariance → 3D-point covariance, i.e. Σ_disp = 0 in Eq. 1) using the **NBUP** formulation from the **USfM** framework — see <https://github.com/michalpolic/usfm.github.io>. This is the "BA-only" baseline used in the paper.

`scripts/visualize_sensor_cov.py` builds a self-contained Three.js HTML viewer that draws the 3σ covariance ellipsoid for any clicked point.

---

## Installation

```bash
git lfs install               # the example COLMAP poses are tracked in Git LFS
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

If you cloned without `git-lfs`, the `examples/**/*.bin` files will be plain-text pointer stubs — run `git lfs pull` after installing `git-lfs` to fetch them.

Required packages:

| Package | Notes |
|---------|-------|
| numpy, scipy | core math |
| opencv-python | SGBM stereo, image I/O |
| laspy | LAS point cloud output (and visualizer LAS overlay) |
| matplotlib | visualizer LAS-overlay polygon clipping |
| torch | only used by `--sift_graph` (GPU brute-force SIFT matching) |

A CUDA-capable GPU is optional but recommended for `--sift_graph` on large datasets.

---

## Repo layout

```
UncertaintyQuantification/
├── scripts/
│   ├── run_mvs_pipeline.py            # main entrypoint (steps 1–7 above)
│   ├── compute_sensor_error_prop.py   # SfM-only baseline (USfM/NBUP, Σ_disp=0)
│   ├── rectify.py                     # stereo rectification helpers
│   ├── colmap_io.py                   # COLMAP binary readers (Schönberger)
│   └── visualize_sensor_cov.py        # interactive Three.js viewer
├── examples/
│   ├── Dortmund/sparse/               # COLMAP poses only — see "Examples"
│   └── UseGeo/Dataset-{1,2,3}/sparse/
├── requirements.txt
└── LICENSE                            # Apache 2.0
```

The `examples/` folders contain only the COLMAP `sparse/` triplet (`cameras.bin`, `images.bin`, `points3D.bin`). Source images must be downloaded separately — see below.

---

## Examples

The example poses we ship were produced with COLMAP using the **original FBK images, with the original filenames preserved**. The shipped `images.bin` therefore references files such as `backward_001_009_145000282.tif` (Dortmund) or `2021-04-23_13-17-12_S2223314_DxO_res.jpg` (UseGeo). Do **not** rename images after download — the pipeline matches by filename.

### 1. Dortmund (oblique aerial — NeRFBK / FBK 3DOM)

The Dortmund scene is part of the **NeRFBK** benchmark. It is a 59-image subset (nadir + oblique) captured with the **IGI PentaCam** (5 camera heads), at native resolution 6132×8176. Download from:

- <https://github.com/3DOM-FBK/NeRFBK> (under the *Aerial — Dortmund* section)

Place all 59 images into `examples/Dortmund/images/`:

```
examples/Dortmund/
├── sparse/                       # provided (LFS)
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── images/                       # 59 .tif files you download
    ├── backward_001_009_145000282.tif
    ├── backward_001_010_145000281.tif
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

> **Note** — the Dortmund subset has no GT camera poses or GT depth maps in NeRFBK, so disparity-uncertainty evaluation is not feasible. Airborne LiDAR is provided and is used in the paper to evaluate point-cloud uncertainty only.

### 2. UseGeo (FBK-3DOM aerial benchmark)

UseGeo is a UAV-based multi-sensor dataset with three flights covering different urban / peri-urban areas (Dataset-1 / Dataset-2 / Dataset-3). Download from:

- <https://github.com/3DOM-FBK/UseGeo>

Use the **DxO color-corrected, resampled** RGB images — filenames end in `..._DxO_res.jpg`. Place them under each `Dataset-N/images/`:

```
examples/UseGeo/Dataset-1/
├── sparse/                       # provided (LFS)
└── images/                       # 224 .jpg files you download
    ├── 2021-04-23_13-17-12_S2223314_DxO_res.jpg
    └── ...
```

Image counts in our shipped poses:

| Dataset   | # images |
|-----------|----------|
| Dataset-1 |   224    |
| Dataset-2 |   327    |
| Dataset-3 |   277    |

Run on Dataset-1:

```bash
python scripts/run_mvs_pipeline.py \
    --scene  examples/UseGeo/Dataset-1/sparse  \
    --images examples/UseGeo/Dataset-1/images  \
    --out    examples/UseGeo/Dataset-1/out     \
    --scale  1.0  \
    --workers 8   \
    --sift_graph
```

Repeat for `Dataset-2` and `Dataset-3`.

### Verifying your image folder

Quick sanity check — Python reports any name in `images.bin` that is missing on disk:

```bash
python - <<'PY'
import sys; sys.path.insert(0, 'scripts')
from run_mvs_pipeline import read_images_bin
from pathlib import Path
sparse = Path('examples/UseGeo/Dataset-1/sparse/images.bin')
img_dir = Path('examples/UseGeo/Dataset-1/images')
names = {im['name'] for im in read_images_bin(sparse).values()}
missing = sorted(n for n in names if not (img_dir / n).exists())
print(f"{len(names)-len(missing)}/{len(names)} images present, {len(missing)} missing")
for n in missing[:5]:
    print("  missing:", n)
PY
```

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

The 3×3 `gt_cov` field stores Σ_g (paper Eq. 1). Per-point scalar uncertainty `sigma` is computed as `sqrt(trace(Σ_g))`.

---

## Visualizer

`scripts/visualize_sensor_cov.py` builds a self-contained Three.js HTML page that lets you click any point to draw its 3σ covariance ellipsoid:

```bash
python scripts/visualize_sensor_cov.py \
    --npz examples/UseGeo/Dataset-1/out/fused_all_cov.npz \
    --las examples/UseGeo/Dataset-1/out/fused_all.las \
    --max_points 200000
```

The script writes the HTML to a temp directory, starts a one-shot local HTTP server, and opens your browser. `--las` is optional — when given, the LAS point cloud is overlaid in its native RGB. `--max_points 0` disables subsampling.

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
  author  = {Huang, Debao and Qin, Rongjun},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2026},
  note    = {Preprint: arXiv:2507.13486}
}
```

If you use the BA-only baseline (`scripts/compute_sensor_error_prop.py`), please also acknowledge the **USfM** framework by Michal Polic — <https://github.com/michalpolic/usfm.github.io>.

---

## License

### Code

Source code in this repository is released under the **Apache License 2.0** — see [LICENSE](LICENSE).

`scripts/colmap_io.py` is derived from the COLMAP `read_write_model.py` script
(© ETH Zürich / UNC Chapel Hill, BSD 3-clause); see the header in that file.

The BA-covariance formulation in `scripts/compute_sensor_error_prop.py` reproduces the math of the **USfM / NBUP** framework by Michal Polic (CTU Prague). The original USfM repository at <https://github.com/michalpolic/usfm.github.io> does not declare a license.

### Example data

The COLMAP `sparse/` files we ship under `examples/` were produced from third-party imagery and are redistributed for **research / non-commercial** evaluation only:

| Source dataset | License | Upstream |
|----------------|---------|----------|
| Dortmund (NeRFBK) | CC-BY-NC-SA 4.0 | <https://github.com/3DOM-FBK/NeRFBK> |
| UseGeo Dataset-1/2/3 | CC-BY-NC-SA 4.0 | <https://github.com/3DOM-FBK/UseGeo> |

Both upstream datasets require attribution, prohibit commercial use, and require derivative works to use the same license. The Apache-2.0 license on this repository covers our **code** only — the example COLMAP poses inherit the upstream CC-BY-NC-SA 4.0 terms. Source images are **not** redistributed here; download them from the FBK repositories above.
