#!/usr/bin/env python3
"""
Generate a self-contained HTML visualiser for sensor-model covariance results.

Load the sensor_cov.npz produced by compute_sensor_error_prop.py, then embed
all data into a single HTML file with Three.js.

Click any point → draws the 3σ covariance ellipsoid at that location.

Usage:
    python visualize_sensor_cov.py --npz data/Dataset-1/block_0000/sensor_cov.npz \
                                   --out  viewer.html
"""

import argparse
import json
import numpy as np
from pathlib import Path


def make_html(xyz: np.ndarray, cov: np.ndarray, sigma: np.ndarray,
              cov2: np.ndarray = None, sigma2: np.ndarray = None,
              cov_label: str = "gt_cov", cov2_label: str = "pred_cov",
              cam_xyz: np.ndarray = None, cam_R: np.ndarray = None,
              aligned_distance: np.ndarray = None,
              bounded: np.ndarray = None, abs_error: np.ndarray = None,
              sq_error: np.ndarray = None,
              lidar_xyz: np.ndarray = None, lidar_rgb: np.ndarray = None,
              title: str = "Sensor Covariance Viewer") -> str:
    """Build a self-contained HTML string."""

    # centre coordinates so Three.js does not lose precision
    # Use median to be robust against outlier points
    centre = np.median(xyz, axis=0)
    xyz_c  = xyz - centre

    # eigen-decompose all covariance matrices (vectorised)
    vals_all, vecs_all = np.linalg.eigh(cov.astype(np.float32))  # (N,3), (N,3,3) ascending
    vals_all = np.maximum(vals_all, 1e-12)
    stds_all = np.sqrt(vals_all)                       # (N,3) semi-axes
    vecs_T   = np.swapaxes(vecs_all, 1, 2)            # (N,3,3) rows = eigenvectors

    has_both = cov2 is not None
    if has_both:
        vals2, vecs2 = np.linalg.eigh(cov2.astype(np.float32))
        vals2 = np.maximum(vals2, 1e-12)
        stds2 = np.sqrt(vals2)
        vecs2_T = np.swapaxes(vecs2, 1, 2)

    has_dist = aligned_distance is not None
    has_metrics = bounded is not None
    has_lidar = lidar_xyz is not None and lidar_rgb is not None

    records = []
    for i in range(len(xyz_c)):
        rec = {
            "p":   xyz_c[i].tolist(),
            "s":   stds_all[i].tolist(),
            "v":   vecs_T[i].tolist(),
            "sxy": float(sigma[i]),
        }
        if has_both:
            rec["s2"]   = stds2[i].tolist()
            rec["v2"]   = vecs2_T[i].tolist()
            rec["sxy2"] = float(sigma2[i])
        if has_dist:
            rec["ad"] = float(aligned_distance[i])
        if has_metrics:
            rec["bd"] = int(bounded[i])        # 1=bounded, 0=unbounded
            rec["ae"] = float(abs_error[i])    # |σ - d|
            rec["se"] = float(sq_error[i])     # (σ - d)²
        records.append(rec)

    data_js = json.dumps(records, separators=(",", ":"))

    # camera poses (same centering as point cloud)
    cam_records = []
    if cam_xyz is not None and cam_R is not None:
        for i in range(len(cam_xyz)):
            cam_records.append({
                "c": (cam_xyz[i] - centre).tolist(),  # centred camera centre
                "R": cam_R[i].tolist(),               # world-to-camera rotation
            })
    cam_js = json.dumps(cam_records, separators=(",", ":"))

    # LiDAR RGB layer (separate point cloud, same centering)
    lidar_records = []
    if has_lidar:
        lidar_c = lidar_xyz - centre
        for i in range(len(lidar_c)):
            lidar_records.append({
                "p": lidar_c[i].tolist(),
                "r": [int(lidar_rgb[i, 0]), int(lidar_rgb[i, 1]), int(lidar_rgb[i, 2])],
            })
    lidar_js = json.dumps(lidar_records, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#111; color:#eee; font-family:monospace; overflow:hidden; }}
  canvas {{ display:block; }}
  #info {{
    position:fixed; top:12px; left:12px; padding:10px 14px;
    background:rgba(0,0,0,.65); border-radius:6px; font-size:13px;
    pointer-events:none; line-height:1.6;
  }}
  #panel {{
    position:fixed; top:12px; right:12px; width:260px; padding:10px 14px;
    background:rgba(0,0,0,.75); border-radius:6px; font-size:12px; line-height:1.7;
    display:none;
  }}
  #panel h3 {{ font-size:13px; margin-bottom:6px; color:#7cf; }}
  .row {{ display:flex; justify-content:space-between; }}
  .lbl {{ color:#aaa; }}
  .val {{ color:#fff; }}
  #cscale {{
    position:fixed; bottom:12px; left:50%; transform:translateX(-50%);
    padding:8px 14px; background:rgba(0,0,0,.75); border-radius:6px;
    font-size:12px; font-family:monospace; text-align:center; min-width:280px;
  }}
  #csrangewrap {{ position:relative; height:22px; margin-bottom:4px; }}
  #csbar {{
    position:absolute; top:50%; transform:translateY(-50%);
    left:5px; right:5px; height:14px; border-radius:3px;
    background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
  }}
  #csrangewrap input[type=range] {{
    position:absolute; left:0; width:100%; top:50%; transform:translateY(-50%);
    -webkit-appearance:none; appearance:none;
    background:transparent; pointer-events:none;
    height:22px; margin:0; padding:0;
  }}
  #csrangewrap input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance:none; appearance:none; pointer-events:all;
    width:10px; height:22px; background:#fff; border:2px solid #888;
    border-radius:3px; cursor:ew-resize;
  }}
  #csrangewrap input[type=range]::-moz-range-thumb {{
    pointer-events:all; width:8px; height:18px;
    background:#fff; border:2px solid #888; border-radius:3px; cursor:ew-resize;
  }}
  #cslabels {{ display:flex; justify-content:space-between; color:#aaa; margin-bottom:5px; }}
  #cslabels span:nth-child(2) {{ color:#888; }}
  #cscontrols {{ display:flex; gap:6px; align-items:center; justify-content:center; }}
  #cscontrols input {{ width:90px; background:#222; color:#fff; border:1px solid #555;
    border-radius:3px; padding:2px 4px; font-family:monospace; font-size:11px; }}
  #cscontrols button {{ background:#444; color:#eee; border:none; border-radius:3px;
    padding:2px 8px; cursor:pointer; font-size:11px; }}
  #cscontrols label {{ color:#aaa; font-size:11px; }}
</style>
</head>
<body>
<div id="info">
  <b>{title}</b><br>
  {len(records)} points &nbsp;|&nbsp;
  <span style="color:#7cf">Left-drag</span> rotate &nbsp;
  <span style="color:#7cf">Right-drag</span> pan &nbsp;
  <span style="color:#7cf">scroll</span> zoom &nbsp;
  <span style="color:#7cf">click</span> show ellipsoid
</div>
<div id="cscale">
  <div id="csrangewrap">
    <div id="csbar"></div>
    <input type="range" id="sl_min" min="0" max="1000" value="0">
    <input type="range" id="sl_max" min="0" max="1000" value="1000">
  </div>
  <div id="cslabels"><span id="csmin">–</span><span>σ (m)</span><span id="csmax">–</span></div>
  <div id="cscontrols">
    <label>Min <input id="inp_min" type="number" step="any"></label>
    <label>Max <input id="inp_max" type="number" step="any"></label>
    <button onclick="resetColorScale()">Auto</button>
  </div>
  <div id="cscontrols" style="margin-top:5px">
    <label style="color:#aaa;font-size:11px">Pt size
      <input id="sl_ptsize" type="range" min="1" max="200" value="20"
        style="width:120px;vertical-align:middle"
        oninput="ptMat.size = bsize * this.value / 10000; if(lidarCloud) lidarCloud.material.size = ptMat.size; if(markerMesh) markerMesh.scale.setScalar(ptMat.size * 0.3); document.getElementById('lbl_ptsize').textContent=(this.value/10).toFixed(1)+'‰'">
    </label>
    <span id="lbl_ptsize" style="color:#fff;font-size:11px">2.0‰</span>
    <button id="btn_cam" onclick="toggleCameras()" style="margin-left:8px">Hide cameras</button>
    {"" if not has_dist else '<button id="btn_mode" onclick="toggleColorMode()" style="margin-left:8px">Show distance</button>'}
    {"" if not has_metrics else '<button id="btn_bounded" onclick="setColorMode(&#39;bd&#39;)" style="margin-left:8px">Bounded</button><button id="btn_ae" onclick="setColorMode(&#39;ae&#39;)" style="margin-left:8px">|error|</button><button id="btn_se" onclick="setColorMode(&#39;se&#39;)" style="margin-left:8px">error²</button>'}
    {"" if not has_lidar else '<button id="btn_lidar" onclick="toggleLidar()" style="margin-left:8px">Show LiDAR RGB</button>'}
    {"" if not has_both else f'<button id="btn_covswitch" onclick="toggleCovSource()" style="margin-left:8px;background:#268;font-weight:bold">Showing: {cov_label}</button>'}
  </div>
</div>
<div id="status" style="position:fixed;bottom:12px;left:12px;padding:6px 10px;
  background:rgba(0,0,0,.6);border-radius:4px;font-size:12px;font-family:monospace;
  color:#aaa;pointer-events:none">
  Click a point to see its covariance ellipsoid
</div>
<div id="panel">
  <h3>&#9651; Covariance Ellipsoid</h3>
  <div class="row"><span class="lbl">Point&nbsp;#</span><span class="val" id="pidx">–</span></div>
  <div class="row"><span class="lbl">Position&nbsp;(m)</span><span class="val" id="ppos">–</span></div>
  <div class="row"><span class="lbl">σ&nbsp;mean&nbsp;(m)</span><span class="val" id="psig">–</span></div>
  <hr style="border-color:#444;margin:6px 0">
  <div class="row"><span class="lbl">σ₁ (true, m)</span><span class="val" id="ps1">–</span></div>
  <div class="row"><span class="lbl">σ₂ (true, m)</span><span class="val" id="ps2">–</span></div>
  <div class="row"><span class="lbl">σ₃ (true, m)</span><span class="val" id="ps3">–</span></div>
  <hr style="border-color:#444;margin:6px 0">
  {"" if not has_metrics else '<hr style="border-color:#444;margin:6px 0"><div class="row"><span class="lbl">Bounded</span><span class="val" id="pbounded">–</span></div><div class="row"><span class="lbl">|σ − d| (m)</span><span class="val" id="pae">–</span></div><div class="row"><span class="lbl">(σ − d)² (m²)</span><span class="val" id="pse">–</span></div>'}
  <div class="row"><span class="lbl">Display&nbsp;scale</span>
    <span class="val" id="dispScaleLabel">×1.0</span>
  </div>
  <div style="margin-top:4px">
    <input id="sl_dispscale" type="range" min="0" max="1000" value="500"
      style="width:100%;vertical-align:middle;cursor:ew-resize"
      oninput="setDispScaleFromSlider(this.value)">
  </div>
  <div class="row" style="margin-top:4px"><span class="lbl">Ellipsoid&nbsp;r (m)</span>
    <span class="val" id="prad">–</span>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TrackballControls.js"></script>
<script>
// ── embedded data ──────────────────────────────────────────────────────────
const DATA = {data_js};
const CAMS = {cam_js};   // [{{c:[x,y,z], R:[[…],[…],[…]]}}, …]
const LIDAR = {lidar_js}; // [{{p:[x,y,z], r:[R,G,B]}}, …]

// ── scene setup ────────────────────────────────────────────────────────────
const W = window.innerWidth, H = window.innerHeight;
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(W, H);
document.body.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(50, W/H, 0.001, 100000);

// fit camera to point cloud using robust percentiles (ignore outliers)
const pts = DATA.map(d => d.p);
function percentile(arr, p) {{
  const s = arr.slice().sort((a,b) => a-b);
  const k = (s.length - 1) * p;
  const f = Math.floor(k), c = Math.ceil(k);
  return f === c ? s[f] : s[f] * (c - k) + s[c] * (k - f);
}}
const bbox = {{ min:[0,0,0], max:[0,0,0] }};
for (let i = 0; i < 3; i++) {{
  const vals = pts.map(p => p[i]);
  bbox.min[i] = percentile(vals, 0.02);
  bbox.max[i] = percentile(vals, 0.98);
}}
const bsize  = Math.max(...[0,1,2].map(i => bbox.max[i]-bbox.min[i]));
const bcen   = [0,1,2].map(i => (bbox.min[i]+bbox.max[i])/2);
camera.position.set(bcen[0]+bsize, bcen[1]+bsize*0.6, bcen[2]+bsize);
camera.lookAt(bcen[0], bcen[1], bcen[2]);

const controls = new THREE.TrackballControls(camera, renderer.domElement);
controls.target.set(bcen[0], bcen[1], bcen[2]);
controls.rotateSpeed    = 2.0;
controls.zoomSpeed      = 3.0;
controls.panSpeed       = 0.8;
controls.dynamicDampingFactor = 0.15;
controls.update();

// Double-click re-centres the orbit target on the nearest point,
// which resets the zoom baseline and lets you keep zooming in.
renderer.domElement.addEventListener('dblclick', e => {{
  const idx = nearestPointToClick(e.clientX, e.clientY);
  if (idx >= 0) {{
    controls.target.set(positions[idx*3], positions[idx*3+1], positions[idx*3+2]);
    controls.update();
    statusEl.textContent = 'Re-centred on point ' + idx;
    statusEl.style.color = '#7cf';
  }}
}});

// ── point cloud ─────────────────────────────────────────────────────────────
const N = DATA.length;
const positions = new Float32Array(N*3);
const colors    = new Float32Array(N*3);

// covariance source toggle (declared early so getColorVals can use getCovSig)
const hasBoth = !!(DATA[0] && DATA[0].s2);
let covSource = 1;  // 1 = primary (s/v), 2 = secondary (s2/v2)
const covLabels = [null, "{cov_label}", "{cov2_label}"];
function getCovS(d) {{ return covSource === 2 && d.s2 ? d.s2 : d.s; }}
function getCovV(d) {{ return covSource === 2 && d.v2 ? d.v2 : d.v; }}
function getCovSig(d) {{ return covSource === 2 && d.sxy2 !== undefined ? d.sxy2 : d.sxy; }}

// colour by sigma: blue (small) → red (large)
// Use 2nd/98th percentiles to exclude outliers from the colour scale
const hasDist = DATA[0].ad !== undefined;
let colorMode = 'sxy';  // 'sxy' or 'ad'

const hasMetrics = DATA[0].bd !== undefined;
function getColorVals() {{
  if (colorMode === 'ad' && hasDist) return DATA.map(d => d.ad);
  if (colorMode === 'ae' && hasMetrics) return DATA.map(d => d.ae);
  if (colorMode === 'se' && hasMetrics) return DATA.map(d => d.se);
  if (colorMode === 'bd' && hasMetrics) return DATA.map(d => d.bd);
  return DATA.map(d => getCovSig(d));
}}

function computeAutoRange(vals) {{
  const sorted = vals.slice().sort((a, b) => a - b);
  return [sorted[Math.floor(sorted.length * 0.01)], sorted[Math.floor(sorted.length * 0.99)]];
}}

let [autoMin, autoMax] = computeAutoRange(getColorVals());
let sigMin = autoMin;
let sigMax = autoMax;

function valToSlider(v) {{ return Math.round((v - autoMin) / (autoMax - autoMin + 1e-12) * 1000); }}
function sliderToVal(t) {{ return autoMin + t / 1000 * (autoMax - autoMin); }}

function sigToColor(s) {{
  // Log-scale mapping: spread small-value differences across more of the colormap
  const logMin = Math.log(Math.max(sigMin, 1e-12));
  const logMax = Math.log(Math.max(sigMax, 1e-12));
  const logS   = Math.log(Math.max(s, 1e-12));
  const t = Math.max(0, Math.min(1, (logS - logMin)/(logMax - logMin + 1e-12)));
  // viridis-ish: blue→cyan→green→yellow→red
  if (t < 0.25) {{ const u=t/0.25;    return [0, u, 1]; }}
  if (t < 0.5)  {{ const u=(t-0.25)/0.25; return [0, 1, 1-u]; }}
  if (t < 0.75) {{ const u=(t-0.5)/0.25;  return [u, 1, 0]; }}
  const u=(t-0.75)/0.25; return [1, 1-u, 0];
}}

for (let i=0; i<N; i++) {{
  positions[i*3]   = DATA[i].p[0];
  positions[i*3+1] = DATA[i].p[1];
  positions[i*3+2] = DATA[i].p[2];
  const [r,g,b] = sigToColor(DATA[i].sxy);
  colors[i*3]=r; colors[i*3+1]=g; colors[i*3+2]=b;
}}

const ptGeo = new THREE.BufferGeometry();
ptGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
ptGeo.setAttribute('color',    new THREE.BufferAttribute(colors,    3));
const ptMat = new THREE.PointsMaterial({{ size: bsize*0.002, vertexColors: true, sizeAttenuation: true }});
const cloud = new THREE.Points(ptGeo, ptMat);
scene.add(cloud);

// ── cameras ──────────────────────────────────────────────────────────────────
let camDotsObj = null, camFrustObj = null;
if (CAMS.length > 0) {{
  const fD = bsize * 0.04;   // frustum depth
  const fH = bsize * 0.025;  // frustum half-width

  const camPosArr = new Float32Array(CAMS.length * 3);
  const lineVerts  = [];

  CAMS.forEach((cam, i) => {{
    const [cx, cy, cz] = cam.c;
    camPosArr[i*3] = cx; camPosArr[i*3+1] = cy; camPosArr[i*3+2] = cz;

    const R  = cam.R;
    const fw = R[2], ri = R[0], dw = R[1];   // forward, right, down (COLMAP Y=down)

    // near-plane centre
    const nx = cx+fD*fw[0], ny = cy+fD*fw[1], nz = cz+fD*fw[2];

    // 4 corners: ±right, ±down
    const corners = [
      [nx+fH*ri[0]-fH*dw[0], ny+fH*ri[1]-fH*dw[1], nz+fH*ri[2]-fH*dw[2]],
      [nx-fH*ri[0]-fH*dw[0], ny-fH*ri[1]-fH*dw[1], nz-fH*ri[2]-fH*dw[2]],
      [nx-fH*ri[0]+fH*dw[0], ny-fH*ri[1]+fH*dw[1], nz-fH*ri[2]+fH*dw[2]],
      [nx+fH*ri[0]+fH*dw[0], ny+fH*ri[1]+fH*dw[1], nz+fH*ri[2]+fH*dw[2]],
    ];
    // apex → corners
    for (const co of corners) lineVerts.push(cx,cy,cz, ...co);
    // rectangle
    for (let j=0; j<4; j++) lineVerts.push(...corners[j], ...corners[(j+1)%4]);
  }});

  // camera centre dots
  const cpGeo = new THREE.BufferGeometry();
  cpGeo.setAttribute('position', new THREE.BufferAttribute(camPosArr, 3));
  camDotsObj = new THREE.Points(cpGeo,
    new THREE.PointsMaterial({{ color: 0xffa500, size: bsize*0.02, sizeAttenuation: true }}));
  scene.add(camDotsObj);

  // frustum wireframes
  const fGeo = new THREE.BufferGeometry();
  fGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(lineVerts), 3));
  camFrustObj = new THREE.LineSegments(fGeo,
    new THREE.LineBasicMaterial({{ color: 0xffa500 }}));
  scene.add(camFrustObj);
}}

let camsVisible = true;
window.toggleCameras = function() {{
  if (!camDotsObj && !camFrustObj) return;
  camsVisible = !camsVisible;
  if (camDotsObj)  camDotsObj.visible  = camsVisible;
  if (camFrustObj) camFrustObj.visible = camsVisible;
  document.getElementById('btn_cam').textContent = camsVisible ? 'Hide cameras' : 'Show cameras';
}};

// ── LiDAR RGB layer ──────────────────────────────────────────────────────────
let lidarCloud = null;
let lidarVisible = false;
if (LIDAR.length > 0) {{
  const lN = LIDAR.length;
  const lPos = new Float32Array(lN * 3);
  const lCol = new Float32Array(lN * 3);
  for (let i = 0; i < lN; i++) {{
    lPos[i*3]   = LIDAR[i].p[0];
    lPos[i*3+1] = LIDAR[i].p[1];
    lPos[i*3+2] = LIDAR[i].p[2];
    lCol[i*3]   = LIDAR[i].r[0] / 255;
    lCol[i*3+1] = LIDAR[i].r[1] / 255;
    lCol[i*3+2] = LIDAR[i].r[2] / 255;
  }}
  const lGeo = new THREE.BufferGeometry();
  lGeo.setAttribute('position', new THREE.BufferAttribute(lPos, 3));
  lGeo.setAttribute('color',    new THREE.BufferAttribute(lCol, 3));
  lidarCloud = new THREE.Points(lGeo,
    new THREE.PointsMaterial({{ size: bsize*0.002, vertexColors: true, sizeAttenuation: true }}));
  lidarCloud.visible = false;
  scene.add(lidarCloud);
}}

window.toggleLidar = function() {{
  if (!lidarCloud) return;
  lidarVisible = !lidarVisible;
  lidarCloud.visible = lidarVisible;
  cloud.visible = !lidarVisible;
  const btn = document.getElementById('btn_lidar');
  if (btn) btn.textContent = lidarVisible ? 'Show covariance' : 'Show LiDAR RGB';
}};

// ── axis indicator (bottom-right corner) ─────────────────────────────────────
const axScene  = new THREE.Scene();
const axCamera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
axCamera.position.set(0, 0, 2.5);
axScene.add(new THREE.AxesHelper(1));

function makeAxisLabel(text, color) {{
  const cv = document.createElement('canvas');
  cv.width = 64; cv.height = 64;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = color;
  ctx.font = 'bold 52px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial(
    {{ map: new THREE.CanvasTexture(cv), depthTest: false, transparent: true }}));
  sp.scale.set(0.55, 0.55, 1);
  return sp;
}}
const lx = makeAxisLabel('X','#ff5555'); lx.position.set(1.4, 0,   0  ); axScene.add(lx);
const ly = makeAxisLabel('Y','#55ff55'); ly.position.set(0,   1.4, 0  ); axScene.add(ly);
const lz = makeAxisLabel('Z','#5599ff'); lz.position.set(0,   0,   1.4); axScene.add(lz);

// ── color scale controls ──────────────────────────────────────────────────────
function syncUI() {{
  document.getElementById('sl_min').value  = valToSlider(sigMin);
  document.getElementById('sl_max').value  = valToSlider(sigMax);
  document.getElementById('inp_min').value = sigMin.toExponential(3);
  document.getElementById('inp_max').value = sigMax.toExponential(3);
  document.getElementById('csmin').textContent = sigMin.toExponential(3);
  document.getElementById('csmax').textContent = sigMax.toExponential(3);
}}

function updateColors() {{
  const vals = getColorVals();
  if (colorMode === 'bd') {{
    // Binary coloring: green=bounded (σ > error), red=unbounded
    for (let i=0; i<N; i++) {{
      if (vals[i] > 0.5) {{ colors[i*3]=0.2; colors[i*3+1]=0.9; colors[i*3+2]=0.2; }}
      else               {{ colors[i*3]=0.9; colors[i*3+1]=0.15; colors[i*3+2]=0.15; }}
    }}
  }} else {{
    for (let i=0; i<N; i++) {{
      const [r,g,b] = sigToColor(vals[i]);
      colors[i*3]=r; colors[i*3+1]=g; colors[i*3+2]=b;
    }}
  }}
  ptGeo.attributes.color.needsUpdate = true;
  const labels = {{ sxy:'σ (m)', ad:'distance (m)', bd:'bounded (green=yes)', ae:'|σ − d| (m)', se:'(σ − d)² (m²)' }};
  document.querySelector('#cslabels span:nth-child(2)').textContent = labels[colorMode] || 'σ (m)';
  syncUI();
}}

syncUI();  // init labels & sliders

window.resetColorScale = function() {{
  [autoMin, autoMax] = computeAutoRange(getColorVals());
  sigMin = autoMin; sigMax = autoMax; updateColors();
}};

window.toggleColorMode = function() {{
  if (!hasDist) return;
  colorMode = colorMode === 'sxy' ? 'ad' : 'sxy';
  const btn = document.getElementById('btn_mode');
  if (btn) btn.textContent = colorMode === 'ad' ? 'Show σ' : 'Show distance';
  resetColorScale();
}};

window.setColorMode = function(mode) {{
  colorMode = mode;
  // Update toggle button label when switching away
  const btn = document.getElementById('btn_mode');
  if (btn && mode !== 'ad' && mode !== 'sxy') btn.textContent = 'Show distance';
  resetColorScale();
}};
document.getElementById('sl_min').addEventListener('input', function() {{
  const v = sliderToVal(parseInt(this.value));
  if (v < sigMax) {{ sigMin = v; updateColors(); }}
  else {{ this.value = valToSlider(sigMax); }}
}});
document.getElementById('sl_max').addEventListener('input', function() {{
  const v = sliderToVal(parseInt(this.value));
  if (v > sigMin) {{ sigMax = v; updateColors(); }}
  else {{ this.value = valToSlider(sigMin); }}
}});
document.getElementById('inp_min').addEventListener('change', function() {{
  const v = parseFloat(this.value);
  if (!isNaN(v) && v < sigMax) {{ sigMin = v; updateColors(); }}
}});
document.getElementById('inp_max').addEventListener('change', function() {{
  const v = parseFloat(this.value);
  if (!isNaN(v) && v > sigMin) {{ sigMax = v; updateColors(); }}
}});

// ── covariance source toggle ──────────────────────────────────────────────────
// (declarations moved above color initialization to avoid TDZ)
function toggleCovSource() {{
  covSource = covSource === 1 ? 2 : 1;
  const btn = document.getElementById('btn_covswitch');
  if (btn) {{
    btn.textContent = 'Showing: ' + covLabels[covSource];
    btn.style.background = covSource === 1 ? '#268' : '#862';
  }}
  // Recompute auto color range from the new sigma source, then refresh colors
  [autoMin, autoMax] = computeAutoRange(getColorVals());
  sigMin = autoMin; sigMax = autoMax;
  updateColors();
  if (selectedIdx >= 0) {{ buildEllipsoid(selectedIdx); updatePanel(selectedIdx); }}
}}

// ── ellipsoid ────────────────────────────────────────────────────────────────
let ellipsoidMesh = null;
let markerMesh    = null;
let selectedIdx   = -1;
let dispScale     = 1.0;   // display multiplier on top of scene-relative base size

// Base display radius = bsize * 0.08 (always visible regardless of true σ).
// The ellipsoid shape (axis ratios) still reflects the true covariance.
const BASE_DISPLAY_R = bsize * 0.08;

// Global scale: the largest sigma across ALL points maps to BASE_DISPLAY_R.
// This keeps ellipsoid sizes comparable across points.
const globalSigmaMax = DATA.reduce((mx, d) => Math.max(mx, d.s[0], d.s[1], d.s[2]), 0) || 1e-9;
const globalDisplayFactor = BASE_DISPLAY_R / globalSigmaMax;

function buildEllipsoid(idx) {{
  if (ellipsoidMesh) {{ scene.remove(ellipsoidMesh); ellipsoidMesh.geometry.dispose(); ellipsoidMesh.material.dispose(); }}
  if (markerMesh)    {{ scene.remove(markerMesh);    markerMesh.geometry.dispose();    markerMesh.material.dispose(); }}

  const d   = DATA[idx];
  const s   = getCovS(d);                   // [s0, s1, s2] true std-devs, ascending
  const sMax = Math.max(...s) || 1e-9;

  // Use the global scale so all ellipsoids are directly size-comparable.
  const sc = s.map(v => v * globalDisplayFactor * dispScale);

  const [v0, v1, v2] = getCovV(d);   // eigenvectors as rows
  const rotMat = new THREE.Matrix4();
  rotMat.set(
    v0[0], v1[0], v2[0], 0,
    v0[1], v1[1], v2[1], 0,
    v0[2], v1[2], v2[2], 0,
       0,     0,     0,  1
  );

  const geo = new THREE.SphereGeometry(1, 32, 20);
  const mat = new THREE.MeshBasicMaterial({{
    color: 0x44aaff, wireframe: true, transparent: true, opacity: 0.75
  }});
  ellipsoidMesh = new THREE.Mesh(geo, mat);
  ellipsoidMesh.position.set(d.p[0], d.p[1], d.p[2]);
  ellipsoidMesh.scale.set(sc[0], sc[1], sc[2]);
  ellipsoidMesh.setRotationFromMatrix(rotMat);
  scene.add(ellipsoidMesh);

  // yellow marker dot — radius tracks current point size
  const mGeo = new THREE.SphereGeometry(1, 8, 8);
  const mMat = new THREE.MeshBasicMaterial({{ color: 0xffff00 }});
  markerMesh  = new THREE.Mesh(mGeo, mMat);
  markerMesh.position.set(d.p[0], d.p[1], d.p[2]);
  markerMesh.scale.setScalar(ptMat.size * 0.3);
  scene.add(markerMesh);

  document.getElementById('prad').textContent =
    sMax.toExponential(3) + ' m (true σ_max)';
}}

function updatePanel(idx) {{
  const d    = DATA[idx];
  const orig = pts[idx];
  document.getElementById('panel').style.display = 'block';
  document.getElementById('pidx').textContent = idx;
  document.getElementById('ppos').textContent = orig.map(v => v.toFixed(2)).join(', ');
  const cs = getCovS(d);
  document.getElementById('psig').textContent = getCovSig(d).toExponential(3) + ' m';
  document.getElementById('ps1').textContent  = cs[0].toExponential(3) + ' m';
  document.getElementById('ps2').textContent  = cs[1].toExponential(3) + ' m';
  document.getElementById('ps3').textContent  = cs[2].toExponential(3) + ' m';
  if (hasMetrics) {{
    const bdEl = document.getElementById('pbounded');
    bdEl.textContent = d.bd ? 'Yes (σ > d)' : 'No (σ ≤ d)';
    bdEl.style.color = d.bd ? '#4f4' : '#f44';
    document.getElementById('pae').textContent = d.ae.toExponential(3) + ' m';
    document.getElementById('pse').textContent = d.se.toExponential(3) + ' m²';
  }}
}}

// Slider 0–1000 maps logarithmically to 0.1× – 20×
// slider 500 → ×1.0 (midpoint)
const DS_MIN = 0.1, DS_MAX = 50.0;
function sliderToDispScale(v) {{
  const t = v / 1000;   // 0..1
  return DS_MIN * Math.pow(DS_MAX / DS_MIN, t);
}}
function dispScaleToSlider(s) {{
  return Math.round(Math.log(s / DS_MIN) / Math.log(DS_MAX / DS_MIN) * 1000);
}}

window.setDispScale = function(n) {{
  dispScale = n;
  document.getElementById('dispScaleLabel').textContent = '×' + n.toFixed(1);
  document.getElementById('sl_dispscale').value = dispScaleToSlider(n);
  if (selectedIdx >= 0) buildEllipsoid(selectedIdx);
}};

window.setDispScaleFromSlider = function(v) {{
  setDispScale(sliderToDispScale(parseInt(v)));
}};

// ── click: nearest-screen-point (more reliable than Three.js raycasting) ─────
const _v3 = new THREE.Vector3();
const CLICK_PX = 30;   // pixel radius for picking

function nearestPointToClick(cx, cy) {{
  let bestIdx = -1;
  let bestZ = Infinity; // Track depth to prioritize foreground points
  const hw = window.innerWidth / 2, hh = window.innerHeight / 2;
  
  for (let i = 0; i < N; i++) {{
    _v3.set(positions[i*3], positions[i*3+1], positions[i*3+2]);
    _v3.project(camera);

    // Ignore points behind the camera or beyond the far clipping plane
    if (_v3.z > 1.0 || _v3.z < -1.0) continue;

    const sx = (_v3.x + 1) * hw;
    const sy = (1 - _v3.y) * hh;
    const d2 = (sx - cx)**2 + (sy - cy)**2;

    // If it's within the click radius, pick the one closest to the camera (lowest z)
    if (d2 < CLICK_PX * CLICK_PX) {{
      if (_v3.z < bestZ) {{
        bestZ = _v3.z;
        bestIdx = i;
      }}
    }}
  }}
  return bestIdx;
}}

let mouseDownPos = {{ x:0, y:0 }};
const statusEl = document.getElementById('status');

renderer.domElement.addEventListener('pointerdown', e => {{
  mouseDownPos = {{ x: e.clientX, y: e.clientY }};
}});

renderer.domElement.addEventListener('pointerup', e => {{
  // If the mouse moved more than 10 pixels, treat it as a camera rotation, not a click
  if (Math.hypot(e.clientX - mouseDownPos.x, e.clientY - mouseDownPos.y) > 10) return;
  
  // Also ignore right-clicks (button 2) which are for panning
  if (e.button !== 0) return;

  const idx = nearestPointToClick(e.clientX, e.clientY);
  if (idx >= 0) {{
    selectedIdx = idx;
    buildEllipsoid(idx);
    updatePanel(idx);
    statusEl.textContent = `Point ${{idx}} selected`;
    statusEl.style.color = '#7cf';
  }} else {{
    statusEl.textContent = 'No point within ' + CLICK_PX + 'px — click closer to a point';
    statusEl.style.color = '#f84';
  }}
}});

// ── resize ───────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  const W2=window.innerWidth, H2=window.innerHeight;
  renderer.setSize(W2, H2);
  camera.aspect = W2/H2;
  camera.updateProjectionMatrix();
}});

// ── auto-show first ellipsoid ────────────────────────────────────────────────
if (DATA.length > 0) {{
  selectedIdx = 0;
  buildEllipsoid(0);
  updatePanel(0);
  statusEl.textContent = 'Auto-selected point 0  |  click any point to change';
  statusEl.style.color = '#7cf';
}}

// ── animate ──────────────────────────────────────────────────────────────────
const AX_SIZE = 120;   // axis gizmo viewport size in pixels
(function animate() {{
  requestAnimationFrame(animate);
  controls.update();

  // main scene (full viewport)
  const W2 = window.innerWidth, H2 = window.innerHeight;
  renderer.setViewport(0, 0, W2, H2);
  renderer.setScissor(0, 0, W2, H2);
  renderer.setScissorTest(true);
  renderer.render(scene, camera);

  // axis gizmo (bottom-right corner)
  // Position axCamera in the same direction as the main camera relative to its target,
  // so it always looks at the origin of axScene regardless of orbit angle.
  axCamera.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(2.5);
  axCamera.lookAt(0, 0, 0);
  renderer.setViewport(W2 - AX_SIZE, 0, AX_SIZE, AX_SIZE);
  renderer.setScissor(W2 - AX_SIZE, 0, AX_SIZE, AX_SIZE);
  renderer.clearDepth();
  renderer.render(axScene, axCamera);

  renderer.setScissorTest(false);
}})();
</script>
</body>
</html>"""
    return html


def main():
    import http.server
    import threading
    import webbrowser
    import tempfile
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True,
                        help="Path to sensor_cov.npz")
    parser.add_argument("--las", default=None,
                        help="Path to fused_all.las for RGB colors (NN-matched to NPZ points)")
    parser.add_argument("--port", type=int, default=0,
                        help="HTTP port (0 = pick a free port automatically)")
    parser.add_argument("--max_points", type=int, default=200_000,
                        help="Max points to visualize (random subsample). 0=all")
    args = parser.parse_args()

    d = np.load(args.npz)
    xyz = d["xyz"]

    # Determine primary covariance: prefer gt_cov, fall back to pred_cov, then legacy "cov"
    if "gt_cov" in d:
        cov = d["gt_cov"]
        cov_label = "gt_cov"
    elif "pred_cov" in d:
        cov = d["pred_cov"]
        cov_label = "pred_cov"
    elif "cov" in d:
        cov = d["cov"]
        cov_label = "cov (legacy)"
    else:
        raise KeyError(f"No covariance key found in {args.npz}; available: {list(d.keys())}")

    # Determine secondary covariance (if both exist)
    cov2 = cov2_label = None
    if "gt_cov" in d and "pred_cov" in d:
        cov2 = d["pred_cov"]
        cov2_label = "pred_cov"

    # Always recompute sigma from the active cov — the npz's stored "sigma"
    # field is computed from pred_cov in train.py, so trusting it would make
    # gt_cov and pred_cov color identically.
    sigma = np.sqrt(cov[:, 0, 0] + cov[:, 1, 1] + cov[:, 2, 2])

    sigma2 = None
    if cov2 is not None:
        sigma2 = np.sqrt(cov2[:, 0, 0] + cov2[:, 1, 1] + cov2[:, 2, 2])

    cam_xyz = d["cam_xyz"] if "cam_xyz" in d else None
    cam_R   = d["cam_R"]   if "cam_R"   in d else None
    aligned_distance = d["aligned_distance"] if "aligned_distance" in d else None

    # ── Optional LiDAR RGB layer from companion LAS file ────────────────────
    lidar_xyz = lidar_rgb = None
    if args.las is not None:
        import laspy
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path as MplPath
        print(f"Loading LAS for RGB: {args.las}")
        # Build convex hull + margin before reading LAS
        hull = ConvexHull(xyz[:, :2])
        hull_pts = xyz[hull.vertices, :2]
        centroid = hull_pts.mean(axis=0)
        margin = 2.0  # meters
        directions = hull_pts - centroid
        dists_to_c = np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-6)
        hull_expanded = hull_pts + directions / dists_to_c * margin
        poly = MplPath(hull_expanded)
        # Stream LAS in chunks, only keeping points inside the hull
        xyz_chunks, rgb_chunks = [], []
        n_total = 0
        with laspy.open(args.las) as reader:
            for chunk in reader.chunk_iterator(1_000_000):
                n_total += len(chunk)
                xy = np.column_stack([chunk.x, chunk.y])
                mask = poly.contains_points(xy)
                if mask.any():
                    xyz_chunks.append(np.column_stack(
                        [chunk.x, chunk.y, chunk.z])[mask].astype(np.float64))
                    crgb = np.column_stack(
                        [chunk.red, chunk.green, chunk.blue])[mask]
                    # Normalize 16-bit color to 0-255
                    if crgb.max() > 255:
                        crgb = (crgb / 256).astype(np.uint8)
                    else:
                        crgb = crgb.astype(np.uint8)
                    rgb_chunks.append(crgb)
        if xyz_chunks:
            lidar_xyz = np.concatenate(xyz_chunks)
            lidar_rgb = np.concatenate(rgb_chunks)
            print(f"  LAS: {n_total} total -> {len(lidar_xyz)} in block hull")
        else:
            print(f"  LAS: {n_total} total -> 0 in block hull (no overlap)")
        # Subsample to match max_points
        max_lidar = args.max_points if args.max_points > 0 else 500_000
        if len(lidar_xyz) > max_lidar:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(lidar_xyz), max_lidar, replace=False)
            idx.sort()
            lidar_xyz, lidar_rgb = lidar_xyz[idx], lidar_rgb[idx]
            print(f"  Subsampled LiDAR to {len(lidar_xyz)} points")

    print(f"Loaded {len(xyz)} points from {args.npz}")
    if aligned_distance is not None:
        print(f"aligned_distance: min={aligned_distance.min():.4f}  "
              f"median={float(np.median(aligned_distance)):.4f}  "
              f"max={aligned_distance.max():.4f} m")

    def _print_stats(name, s):
        print(f"[{name}] sigma: min={s.min():.4f}  "
              f"median={float(np.median(s)):.4f}  max={s.max():.4f} m", end="")
        if aligned_distance is not None:
            bnd = (s > aligned_distance).astype(np.uint8).mean() * 100
            mae = float(np.abs(s - aligned_distance).mean())
            rmse = float(np.sqrt(((s - aligned_distance) ** 2).mean()))
            print(f"  |  bounded: {bnd:.1f}%  |  MAE: {mae:.4f} m  |  RMSE: {rmse:.4f} m")
        else:
            print()

    _print_stats(cov_label, sigma)
    if sigma2 is not None:
        _print_stats(cov2_label, sigma2)

    # Per-point metrics for color-by-error toggles (use primary sigma)
    bounded = abs_error = sq_error = None
    if aligned_distance is not None:
        bounded   = (sigma > aligned_distance).astype(np.uint8)
        abs_error = np.abs(sigma - aligned_distance)
        sq_error  = (sigma - aligned_distance) ** 2

    if args.max_points > 0 and len(xyz) > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xyz), args.max_points, replace=False)
        idx.sort()
        xyz, cov, sigma = xyz[idx], cov[idx], sigma[idx]
        if cov2 is not None:
            cov2, sigma2 = cov2[idx], sigma2[idx]
        if aligned_distance is not None:
            aligned_distance = aligned_distance[idx]
        if bounded is not None:
            bounded, abs_error, sq_error = bounded[idx], abs_error[idx], sq_error[idx]
        print(f"Subsampled to {len(xyz)} points for visualization")

    parts = Path(args.npz).parts
    title = "/".join(parts[-3:]) if len(parts) >= 3 else args.npz

    html = make_html(xyz, cov, sigma,
                     cov2=cov2, sigma2=sigma2,
                     cov_label=cov_label, cov2_label=cov2_label or "",
                     cam_xyz=cam_xyz, cam_R=cam_R,
                     aligned_distance=aligned_distance,
                     bounded=bounded, abs_error=abs_error, sq_error=sq_error,
                     lidar_xyz=lidar_xyz, lidar_rgb=lidar_rgb,
                     title=title)

    # write to a temp directory so the HTTP server can serve it
    tmp_dir  = tempfile.mkdtemp(prefix="sensor_cov_viewer_")
    html_path = os.path.join(tmp_dir, "index.html")
    Path(html_path).write_text(html, encoding="utf-8")

    # start a one-shot HTTP server in a background thread
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None   # silence request logs
    httpd = http.server.HTTPServer(("127.0.0.1", args.port), handler)
    port  = httpd.server_address[1]
    url   = f"http://127.0.0.1:{port}/index.html"

    os.chdir(tmp_dir)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    print(f"Serving at {url}  (Ctrl-C to quit)")
    webbrowser.open(url)

    try:
        t.join()   # block until Ctrl-C
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()