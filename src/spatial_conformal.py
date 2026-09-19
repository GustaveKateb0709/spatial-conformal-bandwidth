#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatially weighted versus equal-weight conformal intervals.
import os
=============================================================
模型：GP，指数协方差 C(h)=sigma^2 exp(-h/phi)+tau^2 1{h=0}，参数最大似然。
坐标按域对角归一化。
流程（split conformal，空间分块留出）：
  - 域内点切 BxB 粗块，依次留一块作测试；其余块随机 70% 训练 / 30% 标定（种子固定）。
  - 用“仅训练集”拟合 GP -> 在标定点算非一致性分数 V_i=|y_i-mean_i|/sd_i。
  - 等权基线 GSCP： t_G = V 的第 ceil((m+1)(1-alpha)) 次序统计量。
  - 空间加权 sLSCP： 核 K(d)=exp(-(d/eta)^2)，eta=2phi；对每个测试点按权重取加权分位数 t_s。
       p_j = w_j/(sum_j w_j + 1)，自身项 p_0=1/(sum w+1) 作为“+∞ 处的质量”
       （这是加权共形预测的标准写法；把它并入有限累计会整体下移一个次序统计量、
        并破坏“等权退化”恒等式，故不作为有限累计的一部分）。
  - 区间 = mean(s) ± t * sd(s)，alpha=0.10。
自检（合成数据）：两种方法的无条件覆盖率都须落在 90%±2%；且 eta->∞（等权）时
              t_s 必须与 t_G 完全相同（逐点相等）。
"""
import os, sys, json, gc, time, csv
import numpy as np

P1 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, P1)
import fit_gp_variogram as M                     # 复用拟合/采样/去趋势/掩膜函数
from scipy.spatial.distance import cdist

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

SEED = 20260918
ALPHA = 0.10
np.random.seed(SEED)

# ============================================================ core
def gp_train_predict(xtr, ytr, xq, seeds=(0.08, 0.5), maxiter=80):
    """训练集拟合 GP（复用 fit_gp 的模型/初值），返回 (pred_mean, pred_sd, fit_dict)。"""
    r = M.fit_gp(xtr[:, 0], xtr[:, 1], ytr, 1.0, seeds=seeds, maxiter=maxiter)
    s2, phi, tau2 = r["sigma2"], r["phi_norm"], r["tau2"]
    n = len(ytr)
    dtr = cdist(xtr, xtr)
    C = s2 * np.exp(-dtr / phi)
    dg = s2 + tau2
    C.flat[::n + 1] = dg + 1e-9 * dg
    A = np.linalg.inv(C)
    del C, dtr
    ones = np.ones(n)
    A1 = A @ ones
    Ay = A @ ytr
    S = ones @ A1
    mu = (ones @ Ay) / S
    resid = ytr - mu
    Ar = A @ resid
    dq = cdist(xq, xtr)
    K = s2 * np.exp(-dq / phi)
    pred = mu + K @ Ar
    varr = (s2 + tau2) - np.einsum("ij,ij->i", K, K @ A, optimize=True)
    sd = np.sqrt(np.maximum(varr, 1e-12))
    del A, K, dq
    gc.collect()
    return pred, sd, r


def gscp_t(V, alpha=ALPHA):
    m = len(V)
    Vs = np.sort(V)
    q = int(np.ceil((m + 1) * (1 - alpha)))
    return np.inf if q > m else float(Vs[q - 1])


def slscp_t(V, xtest, xcal, eta, alpha=ALPHA):
    """返回每个测试点的加权分位数 t_s（数组）。"""
    order = np.argsort(V)
    Vs = V[order]
    Dtc = cdist(xtest, xcal)
    W = np.exp(-(Dtc / eta) ** 2)[:, order]
    P = W / (W.sum(axis=1, keepdims=True) + 1.0)
    C = np.cumsum(P, axis=1)
    ok = C >= (1 - alpha)
    has = ok.any(axis=1)
    idx = np.argmax(ok, axis=1)
    t = np.where(has, Vs[idx], np.inf)
    del Dtc, W, P, C
    gc.collect()
    return t


def assign_blocks(X, B=2):
    qx = np.quantile(X[:, 0], np.linspace(0, 1, B + 1)[1:-1])
    qy = np.quantile(X[:, 1], np.linspace(0, 1, B + 1)[1:-1])
    bx = np.searchsorted(qx, X[:, 0], side="right")
    by = np.searchsorted(qy, X[:, 1], side="right")
    return bx * B + by


def run_one(X, y, B=2, alpha=ALPHA, refit=True, phi_override=None, eta_override=None,
            rng=None):
    """在给定点集 (X,y) 上跑 BxB 分块留出，返回逐折结果列表。"""
    rng = np.random.default_rng(SEED) if rng is None else rng
    blk = assign_blocks(X, B)
    folds = []
    for b in np.unique(blk):
        test = np.where(blk == b)[0]
        rest = np.where(blk != b)[0]
        perm = rng.permutation(len(rest))
        ntr = int(round(0.7 * len(rest)))
        tr = rest[perm[:ntr]]
        cal = rest[perm[ntr:]]
        xtr, ytr = X[tr], y[tr]
        xcal, ycal = X[cal], y[cal]
        xtest, ytest = X[test], y[test]
        if refit:
            m_all, s_all, rfit = gp_train_predict(xtr, ytr, np.vstack([xcal, xtest]))
            mcal, scal = m_all[:len(cal)], s_all[:len(cal)]
            mtest, stest = m_all[len(cal):], s_all[len(cal):]
            phi = rfit["phi_norm"]; bound = rfit["bound_hit"]
        else:
            phi = phi_override; bound = False
            m_all, s_all, rfit = gp_predict_fixed(xtr, ytr, np.vstack([xcal, xtest]),
                                                  ref_params=phi_override)
            mcal, scal = m_all[:len(cal)], s_all[:len(cal)]
            mtest, stest = m_all[len(cal):], s_all[len(cal):]
        phi_use = phi if phi_override is None else phi_override
        eta = eta_override if eta_override is not None else 2.0 * phi_use
        V = np.abs(ycal - mcal) / scal
        tG = gscp_t(V, alpha)
        ts = slscp_t(V, xtest, xcal, eta, alpha)
        covG = float(np.mean(np.abs(ytest - mtest) <= tG * stest))
        covS = float(np.mean(np.abs(ytest - mtest) <= ts * stest))
        widG = float(np.mean(2 * tG * stest))
        wS_pts = 2 * ts * stest
        wG_pts = 2 * tG * stest
        finite = np.isfinite(ts)
        frac_finite = float(np.mean(finite))
        widS = float(np.mean(wS_pts))            # = inf if any infinite
        widS_fin = float(np.mean(wS_pts[finite])) if finite.any() else np.nan
        ratio_fin = (widS_fin / widG) if (finite.any() and widG > 0) else np.nan
        med_ratio_fin = (float(np.median(wS_pts[finite] / wG_pts[finite]))
                         if finite.any() else np.nan)
        fold = dict(block=int(b), n_tr=len(tr), n_cal=len(cal), n_test=len(test),
                    phi=float(phi_use), eta=float(eta), bound=bool(bound),
                    t_G=float(tG), t_S_mean=float(np.mean(ts[finite])) if finite.any() else np.nan,
                    frac_finite=frac_finite,
                    cov_G=covG, cov_S=covS, width_G=widG, width_S=widS,
                    width_S_finite=widS_fin,
                    ratio=(widS / widG if widG > 0 else np.nan),
                    ratio_finite=ratio_fin, med_ratio_finite=med_ratio_fin)
        folds.append(fold)
    return folds


def gp_predict_fixed(xtr, ytr, xq, ref_params):
    """用给定 phi（归一化）预测（sigma2/tau2 由训练集矩估计，仅自检用）。"""
    phi = ref_params
    v = np.var(ytr)
    s2, tau2 = 0.9 * v, 0.1 * v
    n = len(ytr)
    dtr = cdist(xtr, xtr)
    C = s2 * np.exp(-dtr / phi); C.flat[::n + 1] = s2 + tau2 + 1e-9
    A = np.linalg.inv(C)
    ones = np.ones(n); A1 = A @ ones; Ay = A @ ytr
    mu = (ones @ Ay) / (ones @ A1)
    Ar = A @ (ytr - mu)
    K = s2 * np.exp(-cdist(xq, xtr) / phi)
    pred = mu + K @ Ar
    var = (s2 + tau2) - np.einsum("ij,ij->i", K, K @ A, optimize=True)
    return pred, np.sqrt(np.maximum(var, 1e-12)), dict(phi_norm=phi)


# ============================================================ self-test
def write_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def selftest(R=15, B=3, grid=45, phi_true=0.2, s2_true=4.0, tau2_true=0.5):
    print("== SELF-TEST (synthetic GP) ==", flush=True)
    path = os.path.join(OUT, "selftest.csv")
    if os.path.exists(path):
        os.remove(path)
    # 规则网格 [0,1]^2
    g = np.linspace(0, 1, grid)
    X = np.array([[a, b] for b in g for a in g], dtype=float)
    covG_all, covS_all, rat_all = [], [], []
    degen_ok = True
    for r in range(R):
        rs = np.random.default_rng(1000 + r)
        d = cdist(X, X)
        C = s2_true * np.exp(-d / phi_true); C.flat[::len(X) + 1] = s2_true + tau2_true
        L = np.linalg.cholesky(C + 1e-10 * np.eye(len(X)))
        y = L @ rs.normal(size=len(X)) + 5.0
        del C, L, d; gc.collect()
        folds = run_one(X, y, B=B, rng=np.random.default_rng(SEED + r))
        for f in folds:
            write_row(path, dict(rep=r, **f))
            covG_all.append(f["cov_G"]); covS_all.append(f["cov_S"])
            if np.isfinite(f["ratio_finite"]):
                rat_all.append(f["ratio_finite"])
        # 退化检验：eta 极大 -> t_s 应逐点等于 t_G
        blk = assign_blocks(X, B)
        for b in np.unique(blk):
            test = np.where(blk == b)[0]; rest = np.where(blk != b)[0]
            perm = rs.permutation(len(rest)); ntr = int(round(0.7 * len(rest)))
            tr = rest[perm[:ntr]]; cal = rest[perm[ntr:]]
            m_all, s_all, rfit = gp_train_predict(X[tr], y[tr], np.vstack([X[cal], X[test]]))
            mcal, scal = m_all[:len(cal)], s_all[:len(cal)]
            xtest = X[test]
            V = np.abs(y[cal] - mcal) / scal
            tG = gscp_t(V)
            ts = slscp_t(V, xtest, X[cal], eta=np.inf)
            if not np.allclose(ts, tG, rtol=1e-9, atol=1e-12):
                degen_ok = False
    cg, cs = np.mean(covG_all), np.mean(covS_all)
    pass_cov = (abs(cg - 0.9) <= 0.02) and (abs(cs - 0.9) <= 0.02)
    print(f"  GSCP coverage = {cg:.4f} | sLSCP coverage = {cs:.4f} "
          f"(need 0.90±0.02) -> {'PASS' if pass_cov else 'FAIL'}", flush=True)
    print(f"  mean width ratio sLSCP/GSCP = {np.mean(rat_all):.4f}", flush=True)
    print(f"  degeneracy (eta->inf, t_s==t_G for all pts): {'PASS' if degen_ok else 'FAIL'}",
          flush=True)
    summary = dict(R=R, grid=grid, phi_true=phi_true, s2_true=s2_true, tau2_true=tau2_true,
                   cov_G=cg, cov_S=cs, pass_cov=bool(pass_cov),
                   mean_ratio=float(np.mean(rat_all)), degen_pass=bool(degen_ok))
    with open(os.path.join(OUT, "selftest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================================================ real data
def load_field():
    import netCDF4
    ds = netCDF4.Dataset(M.NC)
    pm = ds.variables["PM25"][:].astype(np.float32)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    ds.close()
    H, W = pm.shape
    geo = (float(lon[0]), float(lat[0]), float(lon[1] - lon[0]), float(lat[1] - lat[0]), W, H)
    g0 = M.load_json_from_zip(M.GADM0); o0, i0 = M.collect_polys(g0)
    china = M.rasterize(o0, i0, geo); del g0, o0, i0; gc.collect()
    g1 = M.load_json_from_zip(M.GADM1)
    o1, i1 = M.collect_polys(g1, name_field="NAME_1", names=M.YRD_PROV)
    yrd = M.rasterize(o1, i1, geo); del g1, o1, i1; gc.collect()
    det = M.Detrender(pm, china, lat, lon, cf=5)
    return pm, lat, lon, {"china": china, "provinces": yrd}, det


def real_run(domains=("YRD4", "EastChina", "Nationwide"), N_TARGET=2600, B=3, win=1.0):
    print("== REAL DATA ==", flush=True)
    pm, lat, lon, masks, det = load_field()
    path = os.path.join(OUT, "results.csv")
    if os.path.exists(path):
        os.remove(path)
    domain_pts = {}
    for dname in domains:
        lon0, lon1, lat0, lat1, kind = M.DOMAINS[dname]
        bbox = (lon0, lon1, lat0, lat1)
        mask = masks[kind]
        P = M.sample_points(mask, lat, lon, bbox, N_TARGET, offset=(0, 0))
        z_raw = pm[P["rows"], P["cols"]].astype(np.float64)
        lm = det.sample(P["rows"], P["cols"], win)
        Xd = np.column_stack([P["xn"], P["yn"]])
        domain_pts[dname] = dict(X=Xd, z_raw=z_raw, z_det=z_raw - lm,
                                 D_km=P["D_km"], n=P["n_actual"],
                                 spacing=P["mean_spacing_km"])
        print(f"  {dname}: n={P['n_actual']} D={P['D_km']:.0f}km spacing={P['mean_spacing_km']:.1f}km",
              flush=True)
        del z_raw, lm; gc.collect()
    del pm; gc.collect()

    for dname, dd in domain_pts.items():
        X = dd["X"]
        for prep, y in (("detrend", dd["z_det"]), ("raw", dd["z_raw"])):
            t0 = time.time()
            folds = run_one(X, y, B=B, rng=np.random.default_rng(SEED))
            # 分块汇总
            covG = np.mean([f["cov_G"] for f in folds])
            covS = np.mean([f["cov_S"] for f in folds])
            wG = np.mean([f["width_G"] for f in folds])
            wS = np.mean([f["width_S"] for f in folds])
            wS_fin = np.nanmean([f["width_S_finite"] for f in folds])
            frac_fin = np.mean([f["frac_finite"] for f in folds])
            ratio_fin = np.nanmean([f["ratio_finite"] for f in folds])
            med_ratio_fin = np.nanmean([f["med_ratio_finite"] for f in folds])
            row = dict(domain=dname, prep=prep, n_all=dd["n"], D_km=round(dd["D_km"], 1),
                       spacing_km=round(dd["spacing"], 1), B=B,
                       cov_G=round(covG, 4), cov_S=round(covS, 4),
                       width_G=round(wG, 4), width_S=round(wS, 4),
                       width_S_finite=round(wS_fin, 4), frac_finite=round(frac_fin, 4),
                       ratio_finite=round(ratio_fin, 4), med_ratio_finite=round(med_ratio_fin, 4),
                       phi_mean=round(np.mean([f["phi"] for f in folds]), 4),
                       bound_any=bool(any(f["bound"] for f in folds)),
                       tG_mean=round(np.mean([f["t_G"] for f in folds]), 4),
                       secs=round(time.time() - t0, 1))
            write_row(path, row)
            # 逐折也写一份
            for f in folds:
                write_row(os.path.join(OUT, "results_byfold.csv"),
                          dict(domain=dname, prep=prep, **f))
            print(f"  [{dname}/{prep}] covG={covG:.3f} covS={covS:.3f} "
                  f"wG={wG:.3f} wS_fin={wS_fin:.3f} ratio_fin={ratio_fin:.4f} "
                  f"frac_finite={frac_fin:.3f} phi={row['phi_mean']:.4f} "
                  f"bound={row['bound_any']} ({row['secs']}s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("selftest", "all"):
        st = selftest()
    if mode in ("real", "all"):
        real_run()
    print("== DONE ==", flush=True)
