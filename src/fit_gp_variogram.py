#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial covariance estimation on a 0.01-deg gridded field.
import os
=======================
问题：在 0.01deg 网格上做指数协方差 GP 拟合，检验"相关长度 phi 相对域对角
phi/domain 是否随域放大而系统性下降、并稳定地掉到 1 以下"。
模型：C(h) = sigma^2 * exp(-h/phi) + tau^2 * 1{h=0}
     gamma(h) = tau^2 + sigma^2 (1 - exp(-h/phi))
参数用最大似然（负对数边际似然最小化, L-BFGS-B, log 变换）。
归一化：坐标先转为等距圆柱投影下的 km（域中心纬度 cos 因子），再整体除以域对角 D_km，
         于是域对角在拟合坐标里 = 1，phi_norm 直接等于 phi/domain_diag。
可复现：随机种子固定（本脚本点集用规则网格，仅 fudge offset，无需随机抽样）。
内存：任何时刻不保留多份 3000x3000 float64 稠密矩阵；每个 fit 结束即 del + gc。
"""
import os, sys, json, gc, time, zipfile, traceback
import numpy as np

import netCDF4
from PIL import Image, ImageDraw
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter

# ------------------------------------------------------------------ paths
BASE = os.environ.get("P19_DATA_RAW",
                     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "raw"))
NC   = os.path.join(BASE, "acag/2026-09-17/V6GL03.CNNPM25.AS.201501-201512.nc")
GADM0= os.path.join(BASE, "gadm/2026-09-17/gadm41_CHN_0.json.zip")
GADM1= os.path.join(BASE, "gadm/2026-09-17/gadm41_CHN_1.json.zip")
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

SEED = 20260918
np.random.seed(SEED)

# ------------------------------------------------------------------ domains
DOMAINS = {
    # name : (lon0, lon1, lat0, lat1, mask_kind)
    "YRD4"      : (114.9, 123.1, 27.0, 35.2, "provinces"),   # 长三角四省 (Shanghai/Jiangsu/Zhejiang/Anhui)
    "MidYangtze": (110.0, 123.0, 25.0, 36.0, "china"),       # 长江中下游
    "EastChina" : (105.0, 135.0, 18.0, 45.0, "china"),       # 中国东部
    "Nationwide": (73.0,  135.0, 18.0, 54.0, "china"),       # 全国
}
YRD_PROV = {"Shanghai", "Jiangsu", "Zhejiang", "Anhui"}

# ------------------------------------------------------------------ geo helpers
def load_json_from_zip(zpath):
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(".json")][0]
        with z.open(name) as f:
            return json.load(f)

def iter_polygons(geom):
    t = geom.get("type")
    if t == "Polygon":
        yield geom["coordinates"]
    elif t == "MultiPolygon":
        for p in geom["coordinates"]:
            yield p
    elif t == "GeometryCollection":
        for g in geom["geometries"]:
            yield from iter_polygons(g)

def collect_polys(geojson, name_field=None, names=None):
    outers, inners = [], []
    for feat in geojson["features"]:
        if names is not None:
            nm = feat["properties"].get(name_field)
            if nm not in names:
                continue
        for poly in iter_polygons(feat["geometry"]):
            outers.append(poly[0])
            for h in poly[1:]:
                inners.append(h)
    return outers, inners

def rasterize(outers, inners, geo):
    lon_min, lat_min, dlon, dlat, W, H = geo
    lat_max = lat_min + dlat * (H - 1)
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)

    def to_px(ring):
        # PIL: (col=x=lon, row=y=lat); row 0 at top (lat_max)
        return [((lon - lon_min) / dlon, (lat_max - lat) / dlat) for lon, lat in ring]

    for r in outers:
        if len(r) >= 3:
            d.polygon(to_px(r), fill=1)
    for r in inners:
        if len(r) >= 3:
            d.polygon(to_px(r), fill=0)
    # 图像 row0 在顶部=lat_max；翻转为与 netCDF 一致的升序行( row0=lat_min )，
    # 这样 mask[row,col] 与 pm[row,col]、lat[row]、lon[col] 完全对齐。
    return np.array(img, dtype=bool)[::-1, :]

# ------------------------------------------------------------------ GP MLE
def _nll_grad(d, z, theta, n):
    """C = s2*R + tau2*I (R=exp(-d/phi), R_ii=1)。精确 profile 掉常数均值。
    返回 (f, grad)。grad 解析式，M = A - q qᵀ + g gᵀ/S，A=C^{-1}, g=A1, q=A r。"""
    ls2, lphi, ltau = theta
    s2 = np.exp(ls2); phi = np.exp(lphi); tau2 = np.exp(ltau)
    R = np.exp(-d / phi)
    C = R * s2
    dg = s2 + tau2
    C.flat[::n + 1] = dg + 1e-9 * dg
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return 1e12, np.zeros(3)
    A = np.linalg.inv(C)
    del C
    ones = np.ones(n)
    g = A @ ones
    gz = A @ z
    S = ones @ g
    mu = (ones @ gz) / S
    r = z - mu
    q = A @ r
    quad = r @ q
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    f = 0.5 * (quad + logdet + np.log(S) + (n - 1) * np.log(2 * np.pi))

    W = (d / phi) * R
    W.flat[::n + 1] = 0.0
    trA = np.trace(A)
    e_AR = np.einsum("ij,ij->", A, R, optimize=True)
    e_AW = np.einsum("ij,ij->", A, W, optimize=True)
    Rq = R @ q; Rg = R @ g
    Wq = W @ q; Wg = W @ g
    gr = np.empty(3)
    gr[0] = 0.5 * s2 * (e_AR - (q @ Rq) - (g @ Rg) / S)      # d/d log s2
    gr[1] = 0.5 * s2 * (e_AW - (q @ Wq) - (g @ Wg) / S)      # d/d log phi
    gr[2] = 0.5 * tau2 * (trA - (q @ q) - (g @ g) / S)       # d/d log tau2
    del A, R, W, L
    return f, gr


def fit_gp(x_km, y_km, z, D_km, seeds=(0.08, 0.5), maxiter=80):
    """x_km,y_km 已按 D_km 归一化；z 为观测。返回 dict。"""
    n = len(z)
    start = time.time()
    xy = np.column_stack([x_km, y_km])
    d = cdist(xy, xy, "euclidean")           # n x n float64（唯一一份）
    z = np.asarray(z, dtype=np.float64)

    PHI_LO, PHI_HI = 1e-3, 5.0
    LOGB = (np.log(1e-12), np.log(1e10))
    bounds = [LOGB, (np.log(PHI_LO), np.log(PHI_HI)), LOGB]

    def fg(theta):
        return _nll_grad(d, z, theta, n)

    v = np.var(z) + 1e-6
    best = None
    for sp in seeds:
        x0 = np.array([np.log(v), np.log(sp), np.log(0.1 * v + 1e-9)])
        try:
            res = minimize(lambda t: fg(t), x0, jac=True, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    res = best
    s2 = float(np.exp(res.x[0])); phi = float(np.exp(res.x[1])); tau2 = float(np.exp(res.x[2]))
    bound_hit = bool(phi <= PHI_LO * 1.5 or phi >= PHI_HI / 1.5)

    # 条件数：用 chol 对角比值的平方做廉价代理（避免 O(n^3) SVD 与大内存峰值）
    try:
        R = np.exp(-d / phi); C = s2 * R
        dg = s2 + tau2; C.flat[::n + 1] = dg + 1e-9 * dg
        L = np.linalg.cholesky(C)
        dL = np.diag(L)
        cond = float((dL.max() / dL.min()) ** 2)
        del R, C, L
    except Exception:
        cond = float("nan")

    out = dict(
        n=n, n_actual=n,
        sigma2=s2, tau2=tau2,
        phi_norm=phi,
        phi_km=phi * D_km,
        ratio_phi_domain=phi,
        D_km=D_km,
        logml=-float(res.fun), nll=float(res.fun),
        converged=bool(res.success), nit=int(res.nit),
        bound_hit=bound_hit, cond=cond,
        secs=round(time.time() - start, 2),
    )
    del d
    gc.collect()
    return out

# ------------------------------------------------------------------ local mean (detrend) via coarse binning
class Detrender:
    """在粗网格(0.05deg)上算陆地局部均值，再双线性采样。"""
    def __init__(self, pm, landmask, lat, lon, cf=5):
        H, W = pm.shape
        self.cf = cf
        self.H, self.W = H, W
        Hc, Wc = H // cf, W // cf
        self.Hc, self.Wc = Hc, Wc
        # 分块聚合，避免任何时刻出现全尺寸 float64 副本（~448MB）
        sums = np.zeros((Hc, Wc), dtype=np.float64)
        cnts = np.zeros((Hc, Wc), dtype=np.float64)
        blk = 100  # 粗网格行块
        for r0 in range(0, Hc, blk):
            r1 = min(r0 + blk, Hc)
            sub = pm[r0 * cf:r1 * cf, :Wc * cf].astype(np.float64)
            mk = landmask[r0 * cf:r1 * cf, :Wc * cf]
            np.copyto(sub, 0.0, where=~mk)
            sub = sub.reshape(r1 - r0, cf, Wc, cf)
            sums[r0:r1] = sub.sum(axis=(1, 3))
            cnts[r0:r1] = mk.reshape(r1 - r0, cf, Wc, cf).sum(axis=(1, 3))
            del sub, mk
        self.sums = sums
        self.cnts = cnts
        gc.collect()
        self._cache = {}

    def mean_map(self, win_deg):
        if win_deg in self._cache:
            return self._cache[win_deg]
        cell_deg = 0.05
        size = max(1, int(round(win_deg / cell_deg)))
        s = uniform_filter(self.sums, size=size, mode="constant")
        c = uniform_filter(self.cnts, size=size, mode="constant")
        m = np.where(c > 0, s / np.maximum(c, 1e-9), np.nan)
        self._cache[win_deg] = m
        return m

    def sample(self, rows, cols, win_deg):
        m = self.mean_map(win_deg)
        cf = self.cf
        fr = rows / cf - 0.5
        fc = cols / cf - 0.5
        r0 = np.clip(np.floor(fr).astype(int), 0, self.Hc - 2)
        c0 = np.clip(np.floor(fc).astype(int), 0, self.Wc - 2)
        dr = fr - r0; dc = fc - c0
        v = (m[r0, c0] * (1 - dr) * (1 - dc) + m[r0 + 1, c0] * dr * (1 - dc) +
             m[r0, c0 + 1] * (1 - dr) * dc + m[r0 + 1, c0 + 1] * dr * dc)
        return v

# ------------------------------------------------------------------ point sampling (regular lattice)
def sample_points(mask, lat, lon, bbox, n, offset=(0.0, 0.0), cand_stride=5):
    lon0, lon1, lat0, lat1 = bbox
    i0 = int(np.searchsorted(lat, lat0)); i1 = int(np.searchsorted(lat, lat1))
    j0 = int(np.searchsorted(lon, lon0)); j1 = int(np.searchsorted(lon, lon1))
    sub = mask[i0:i1 + 1, j0:j1 + 1]
    coarse = sub[::cand_stride, ::cand_stride]
    rr, cc = np.where(coarse)
    if len(rr) == 0:
        raise RuntimeError("no land points in domain")
    land_frac = len(rr) / coarse.size
    RR = rr * cand_stride + i0
    CC = cc * cand_stride + j0
    lat_c = 0.5 * (lat0 + lat1)
    kx = 111.320 * np.cos(np.deg2rad(lat_c))   # km per deg lon
    ky = 110.574                               # km per deg lat
    x = lon[CC] * kx
    y = lat[RR] * ky
    pts = np.column_stack([x, y])

    X0, X1 = lon0 * kx, lon1 * kx
    Y0, Y1 = lat0 * ky, lat1 * ky
    Wd, Ht = X1 - X0, Y1 - Y0
    D_km = float(np.hypot(Wd, Ht))

    # 点阵密度按陆地占比放大，使“落在陆地上的点数≈目标 n”
    m = int(np.ceil(np.sqrt(n / max(land_frac, 1e-3))))
    gx = np.linspace(X0, X1, m + 2)[1:-1]
    gy = np.linspace(Y0, Y1, m + 2)[1:-1]
    gx = gx + offset[0] * (Wd / (m + 1))
    gy = gy + offset[1] * (Ht / (m + 1))
    gx = np.clip(gx, X0, X1); gy = np.clip(gy, Y0, Y1)
    node_x, node_y = np.meshgrid(gx, gy)
    nodes = np.column_stack([node_x.ravel(), node_y.ravel()])

    tree = cKDTree(pts)
    _, idx = tree.query(nodes, k=1)
    idx = np.unique(idx)
    sel = pts[idx]
    sel_rows = RR[idx]; sel_cols = CC[idx]

    # 拟合坐标：以域对角 D_km 归一化
    xc = sel[:, 0].mean(); yc = sel[:, 1].mean()
    xn = (sel[:, 0] - xc) / D_km
    yn = (sel[:, 1] - yc) / D_km
    return dict(xn=xn, yn=yn, rows=sel_rows, cols=sel_cols, D_km=D_km,
                n_actual=len(idx), mean_spacing_km=D_km / np.sqrt(len(idx)))

# ------------------------------------------------------------------ main
def main():
    t_start = time.time()
    print("== loading netCDF ==", flush=True)
    ds = netCDF4.Dataset(NC)
    pm = ds.variables["PM25"][:].astype(np.float32)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    ds.close()
    H, W = pm.shape
    print("field", pm.shape, "lat", lat[0], lat[-1], "lon", lon[0], lon[-1], flush=True)

    geo = (float(lon[0]), float(lat[0]), float(lon[1] - lon[0]),
           float(lat[1] - lat[0]), W, H)

    print("== rasterizing masks ==", flush=True)
    g0 = load_json_from_zip(GADM0)
    o0, i0_ = collect_polys(g0)
    china_mask = rasterize(o0, i0_, geo)
    del g0, o0, i0_; gc.collect()

    g1 = load_json_from_zip(GADM1)
    o1, i1_ = collect_polys(g1, name_field="NAME_1", names=YRD_PROV)
    yrd_mask = rasterize(o1, i1_, geo)
    del g1, o1, i1_; gc.collect()
    print("china land px", int(china_mask.sum()), " yrd px", int(yrd_mask.sum()), flush=True)

    print("== building detrend maps ==", flush=True)
    det = Detrender(pm, china_mask, lat, lon, cf=5)
    print("== detrend maps ready ==", flush=True)

    masks = {"china": china_mask, "provinces": yrd_mask}

    # ---------- Step 2 : design matrix ----------
    design_rows = []
    NS = [1000, 2000, 3000]
    PREPS = [("raw", None), ("detrend1", 1.0)]
    WIN_MAIN = 1.0

    for dname, (lon0, lon1, lat0, lat1, kind) in DOMAINS.items():
        bbox = (lon0, lon1, lat0, lat1)
        mask = masks[kind]
        for n in NS:
            try:
                P = sample_points(mask, lat, lon, bbox, n, offset=(0.0, 0.0))
            except RuntimeError as e:
                print("SKIP", dname, n, e); continue
            z_raw = pm[P["rows"], P["cols"]].astype(np.float64)
            lm = det.sample(P["rows"], P["cols"], WIN_MAIN)
            z_det = z_raw - lm
            for prep, _ in PREPS:
                z = z_raw if prep == "raw" else z_det
                try:
                    r = fit_gp(P["xn"], P["yn"], z, P["D_km"])
                except Exception:
                    traceback.print_exc(); continue
                r.update(domain=dname, prep=prep, n_target=n,
                         n_actual=P["n_actual"], mean_spacing_km=round(P["mean_spacing_km"], 2))
                design_rows.append(r)
                print(f"  [design] {dname:11s} {prep:8s} n={P['n_actual']:4d} "
                      f"phi={r['phi_km']:8.1f}km ratio={r['ratio_phi_domain']:.3f} "
                      f"conv={r['converged']} bound={r['bound_hit']} t={r['secs']}s", flush=True)
            del z_raw, lm, z_det
            gc.collect()

    # ---------- Step 3 : stability ----------
    stab_rows = []
    N_STAB = 2000
    OFFSETS = [(0.0, 0.0), (0.33, 0.33), (0.67, 0.67)]
    WINDOWS = [0.5, 1.0, 2.0]

    for dname, (lon0, lon1, lat0, lat1, kind) in DOMAINS.items():
        bbox = (lon0, lon1, lat0, lat1)
        mask = masks[kind]
        # (a) 换 n（detrend, offset0, win1）
        for n in NS:
            P = sample_points(mask, lat, lon, bbox, n, offset=(0, 0))
            z_raw = pm[P["rows"], P["cols"]].astype(np.float64)
            lm = det.sample(P["rows"], P["cols"], 1.0)
            for prep, z in (("detrend", z_raw - lm), ("raw", z_raw)):
                r = fit_gp(P["xn"], P["yn"], z, P["D_km"])
                r.update(domain=dname, prep=prep, n_target=n, n_actual=P["n_actual"],
                         offset="0,0", window_deg=1.0, axis="n")
                stab_rows.append(r)
            del z_raw, lm; gc.collect()
        # (b) 换 offset（detrend, n=2000, win1）
        for off in OFFSETS:
            P = sample_points(mask, lat, lon, bbox, N_STAB, offset=off)
            z_raw = pm[P["rows"], P["cols"]].astype(np.float64)
            lm = det.sample(P["rows"], P["cols"], 1.0)
            r = fit_gp(P["xn"], P["yn"], z_raw - lm, P["D_km"])
            r.update(domain=dname, prep="detrend", n_target=N_STAB, n_actual=P["n_actual"],
                     offset=f"{off[0]},{off[1]}", window_deg=1.0, axis="offset")
            stab_rows.append(r)
            del z_raw, lm; gc.collect()
        # (c) 换 window（detrend, n=2000, offset0）
        P = sample_points(mask, lat, lon, bbox, N_STAB, offset=(0, 0))
        z_raw = pm[P["rows"], P["cols"]].astype(np.float64)
        for wd in WINDOWS:
            lm = det.sample(P["rows"], P["cols"], wd)
            r = fit_gp(P["xn"], P["yn"], z_raw - lm, P["D_km"])
            r.update(domain=dname, prep="detrend", n_target=N_STAB, n_actual=P["n_actual"],
                     offset="0,0", window_deg=wd, axis="window")
            stab_rows.append(r)
            del lm; gc.collect()
        del z_raw; gc.collect()
        print(f"  [stability] {dname} done", flush=True)

    # ---------- write outputs ----------
    import csv
    def write_csv(path, rows):
        if not rows: return
        cols = ["domain", "prep", "n_target", "n_actual", "phi_km", "ratio_phi_domain",
                "sigma2", "tau2", "logml", "converged", "bound_hit", "cond", "D_km",
                "mean_spacing_km", "nit", "secs"]
        cols = [c for c in cols if c in rows[0]] + [c for c in rows[0] if c not in cols]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)

    write_csv(os.path.join(OUT, "results_phi.csv"), design_rows)
    write_csv(os.path.join(OUT, "stability.csv"), stab_rows)

    meta = dict(
        seed=SEED,
        model="C(h)=sigma2*exp(-h/phi)+tau2*1{h=0}",
        normalization="coords->km(equirect @ domain-center lat), then /domain_diag; "
                      "so ratio_phi_domain == phi in normalized units",
        domains={k: dict(lon0=v[0], lon1=v[1], lat0=v[2], lat1=v[3], mask=v[4])
                 for k, v in DOMAINS.items()},
        n_grid=[1000, 2000, 3000],
        preps=["raw", "detrend1(win=1.0deg~111km)"],
        stability=dict(n=[1000, 2000, 3000], offsets=[list(o) for o in OFFSETS],
                       windows_deg=WINDOWS, n_fixed=N_STAB),
        total_secs=round(time.time() - t_start, 1),
    )
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dict(meta=meta, design=design_rows, stability=stab_rows), f,
                  indent=2, ensure_ascii=False)
    print("== DONE in %.1fs ==" % (time.time() - t_start), flush=True)

if __name__ == "__main__":
    main()
