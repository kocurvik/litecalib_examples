"""The exported assets (assets/) -> the light files the project page loads (web/):

    python convert.py              (needs numpy and Pillow; rewrites web/ from scratch)

Inputs, per scene folder assets/<scene>/ (what the paper repo's video/prepare_assets.py writes):
    mde_<method>_{fisheye,litecalib}.npz  single-image geometry (the paper repo's video/prepare_mde_assets.py: moge3,
                                       unidepth_v2, unik3d, pi3x, mapanything, da3, vggt_omega): points (H, W, 3),
                                       colors (H, W, 3), mask (H, W); mde_{fisheye,litecalib}.npz = the older MoGe-3 files
    mde_<method>_planes.json           plane fits of its walls (angles_deg), optional (mde_planes.json: the older MoGe-3's)
    mde_frame.jpg, mde_undist.jpg      its input photo, raw and undistorted with LiteCalib
    <method>_{fisheye,litecalib,gt}.npz  3D reconstruction (e.g. pi3x): points, colors, c2w (N, 4, 4), conf optional;
                                       gt = photos undistorted with the ground-truth camera, optional
    <method>_summary.json              {"fisheye": {"maa5": ...}, "litecalib": ..., "gt": ...}, optional
    fisheye/*.JPG, undist_litecalib/*.JPG  its input photos, raw and undistorted with LiteCalib

Outputs:
    web/manifest.json                  {task: {scene: {method: {files, gt?, photos, <stats>}}}}, paths relative to the repo
    web/clouds/<task>/<scene>/<method>_<variant>.bin.gz   subsampled point clouds (format below)
    web/photos/<scene>/...             480 px wide copies of the input photos (the originals stay in assets/)

Cloud format, gzipped, little endian: b"LCP2", uint32 n_points, uint32 n_cams, uint32 0, float32 bbox min[3], max[3],
float32 cams[n_cams][12] (camera-to-world 3x4, row-major), then the points in Morton order, each channel as a plane of
deltas to the previous point (wrapping): xyz int16 (quantized to the bbox) as x, y, z planes, each split into its low
then its high bytes, then rgb uint8 as r, g, b planes. Coordinates are three.js's: x right, y up, z backward
(OpenCV's x, -y, -z).
"""
import gzip
import json
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT / "web"
FLIP = np.diag([1.0, -1.0, -1.0])
THUMB_WIDTH = 480
MDE_POINTS, RECON_POINTS = 180_000, 200_000
CUT_TOP = {"berlin_living": 20}  # percent of the highest points dropped (recon), the video's cut through the ceiling


def rel(path):
    return path.relative_to(ROOT).as_posix()


def morton_order(q, bits=10):
    u = ((q.astype(np.int64) + 32768) >> (16 - bits)).astype(np.uint64)
    code = np.zeros(len(u), np.uint64)
    for i in range(bits):
        for a in range(3):
            code |= ((u[:, a] >> np.uint64(i)) & np.uint64(1)) << np.uint64(3 * i + a)
    return np.argsort(code, kind="stable")


def write_cloud(path, P, C, cams):
    path.parent.mkdir(parents=True, exist_ok=True)

    lo, hi = P.min(0), P.max(0)
    q = np.round((P - lo) / np.maximum(hi - lo, 1e-9) * 65535 - 32768).clip(-32768, 32767).astype(np.int16)
    rgb = (C.clip(0, 1) * 255).round().astype(np.uint8)
    o = morton_order(q)
    q, rgb = q[o], rgb[o]

    dq = (np.diff(q.astype(np.int32), axis=0, prepend=0) & 0xFFFF).astype("<u2")  # wrapping int16 deltas
    drgb = (np.diff(rgb.astype(np.int16), axis=0, prepend=0) & 0xFF).astype(np.uint8)
    xyz = b"".join(np.ascontiguousarray(dq[:, a]).view(np.uint8).reshape(-1, 2).T.tobytes() for a in range(3))

    cams = np.asarray(cams, dtype="<f4").reshape(-1, 12)

    data = (b"LCP2" + struct.pack("<III", len(P), len(cams), 0) + struct.pack("<6f", *lo, *hi) + cams.tobytes()
            + xyz + np.ascontiguousarray(drgb.T).tobytes())

    path.write_bytes(gzip.compress(data, 9, mtime=0))
    print(f"{rel(path)}: {len(P)} points, {len(cams)} cameras, {path.stat().st_size / 1e6:.2f} MB")
    return rel(path)


def thumbs(scene, sources, folder):
    """[thumbnail, original] per photo"""

    res = []
    for src in sources:
        dst = OUT / "photos" / scene / folder / (src.stem + ".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src).convert("RGB")
        im.thumbnail((THUMB_WIDTH, 10_000))
        im.save(dst, quality=82)
        res.append([rel(dst), rel(src)])
    return res


def subsample(n, k, seed=0):
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False)) if n > k else np.arange(n)


def viewer_c2w(c2w):
    M = np.eye(4)
    M[:3, :3] = FLIP @ c2w[:3, :3] @ FLIP
    M[:3, 3] = FLIP @ c2w[:3, 3]
    return M[:3].reshape(-1)


def colors01(C):
    C = np.asarray(C, dtype=np.float32)
    return C / 255.0 if C.max() > 1.5 else C


def rnd(x, d=1):
    return None if x is None else round(float(x), d)


def wall_angle(angles):
    """the angle between the first two walls of the plane fits (e.g. 'left wall / window wall')"""

    for k, v in (angles or {}).items():
        a, _, b = k.partition(" / ")
        if "wall" in a and "wall" in b:
            return v
    return None


def mde_methods(d):
    """method -> its npz name prefix and its plane fits: mde_<method>_{fisheye,litecalib}.npz (+ _planes.json), and
    the older MoGe-3 names mde_{fisheye,litecalib}.npz (+ mde_planes.json) when there is no mde_moge3_*"""
    res = {}
    for f in sorted(d.glob("mde_*_fisheye.npz")):
        m = f.name[len("mde_"):-len("_fisheye.npz")]
        if (d / f"mde_{m}_litecalib.npz").exists():
            res[m] = (f"mde_{m}_", d / f"mde_{m}_planes.json")
    if "moge3" not in res and all((d / f"mde_{t}.npz").exists() for t in ("fisheye", "litecalib")):
        res["moge3"] = ("mde_", d / "mde_planes.json")
    return res


def mde(d):
    res = {}
    for method, (prefix, planes) in mde_methods(d).items():
        entry = {"files": []}
        for tag in ("fisheye", "litecalib"):
            z = np.load(d / f"{prefix}{tag}.npz")
            mask = z["mask"].astype(bool) if "mask" in z else np.ones(z["points"].shape[:-1], bool)
            P = z["points"].astype(np.float32).reshape(-1, 3)[mask.reshape(-1)]
            C = colors01(z["colors"]).reshape(-1, 3)[mask.reshape(-1)]
            keep = np.isfinite(P).all(1)
            P, C = P[keep], C[keep]
            i = subsample(len(P), MDE_POINTS)
            entry["files"].append(write_cloud(OUT / "clouds" / "mde" / d.name / f"{method}_{tag}.bin.gz", P[i] @ FLIP, C[i], [viewer_c2w(np.eye(4))]))

        if planes.exists():
            p = json.loads(planes.read_text())
            entry["angle"] = [rnd(wall_angle(p.get(t, {}).get("angles_deg"))) for t in ("fisheye", "litecalib")]

        entry["photos"] = {"fisheye": thumbs(d.name, [d / "mde_frame.jpg"], "mde"),
                           "litecalib": thumbs(d.name, [d / "mde_undist.jpg"], "mde")}
        res[method] = entry
    return res


def recon_cloud(d, method, tag):
    z = np.load(d / f"{method}_{tag}.npz")
    P = z["points"].astype(np.float32).reshape(-1, 3)
    C = colors01(z["colors"]).reshape(-1, 3)
    keep = np.isfinite(P).all(1)

    if "conf" in z:
        keep &= z["conf"].astype(np.float32).reshape(-1) > 0.3

    P, C = P[keep], C[keep]
    c2w = z["c2w"].astype(np.float64).copy()

    # one scale per cloud (a feed-forward reconstruction's scale is arbitrary): the median distance between its cameras
    Cc = c2w[:, :3, 3]
    s = float(np.median(np.linalg.norm(Cc[:, None] - Cc[None], axis=2)[np.triu_indices(len(Cc), 1)])) if len(Cc) > 1 else 0.0
    if s < 1e-6:
        s = float(np.median(np.linalg.norm(P - Cc[0], axis=1)))

    P /= s
    c2w[:, :3, 3] /= s

    cut = CUT_TOP.get(d.name, 0)
    if cut:  # the camera's y points down: the highest points have the smallest y
        keep = P[:, 1] > np.percentile(P[:, 1], cut)
        P, C = P[keep], C[keep]

    i = subsample(len(P), RECON_POINTS)
    return write_cloud(OUT / "clouds" / "recon" / d.name / f"{method}_{tag}.bin.gz", P[i] @ FLIP, C[i], [viewer_c2w(m) for m in c2w])


def recon(d):
    res = {}
    for f in sorted(d.glob("*_fisheye.npz")):
        method = f.name[:-len("_fisheye.npz")]
        if method == "mde" or method.startswith("mde_") or not (d / f"{method}_litecalib.npz").exists():
            continue

        entry = {"files": [recon_cloud(d, method, t) for t in ("fisheye", "litecalib")]}
        if (d / f"{method}_gt.npz").exists():
            entry["gt"] = recon_cloud(d, method, "gt")

        if (d / f"{method}_summary.json").exists():
            s = json.loads((d / f"{method}_summary.json").read_text())
            entry["maa5"] = [rnd(s.get(t, {}).get("maa5")) for t in ("fisheye", "litecalib")]
            entry["maa5_gt"] = rnd(s.get("gt", {}).get("maa5"))

        entry["photos"] = {"fisheye": thumbs(d.name, sorted((d / "fisheye").glob("*.[jJ][pP][gG]")), "fisheye"),
                           "litecalib": thumbs(d.name, sorted((d / "undist_litecalib").glob("*.[jJ][pP][gG]")), "litecalib")}
        res[method] = entry
    return res


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()

    manifest = {"mde": {}, "recon": {}}
    for d in sorted(p for p in ASSETS.iterdir() if p.is_dir()):
        for task, fn in (("mde", mde), ("recon", recon)):
            r = fn(d)
            if r:
                manifest[task][d.name] = r

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({t: {s: list(m) for s, m in v.items()} for t, v in manifest.items()}, indent=1))