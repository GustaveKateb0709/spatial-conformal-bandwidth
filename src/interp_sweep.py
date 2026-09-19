#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation design with a bandwidth scan (independent implementation).
import os
为什么补这一步
--------------
整块空间留出（外推）的结果显示：去趋势后 φ 变小，核窄到在留出的整块里几乎没有标定点，
加权分位数无解、区间变成无穷宽。但 Mao et al. (2024) 的理论前提是 **infill asymptotics**
——目标点附近有标定点。外推场景根本不满足这个前提，用它评判该方法不公平。
所以本脚本做内插场景：随机留出（测试点与标定点空间交错），并扫描核带宽 η。
主指标（修正后）：**条件覆盖**（域切成 5x5 格，看格间覆盖率的离散度），宽度作为代价。
"""
import os, sys, gc, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spatial_conformal as SC

M = SC.M
OUT = HERE
CSV = os.path.join(OUT, "results_interp.csv")
ETA_MULT = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
R_REP = 5
N_TARGET = 2600
FRAC_TEST, FRAC_CAL = 0.30, 0.21
FIELDS = ["domain", "prep", "rep", "eta_mult", "eta", "phi", "n_tr", "n_cal", "n_test",
          "nn_dist_med_km", "cov", "cov_gap_std", "cov_gap_min", "cov_gap_max",
          "frac_gap_in_band", "n_gaps", "width_mean", "frac_finite", "ratio_mean",
          "secs"]


def append(row, path=CSV):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def gap_metrics(Xte, covered, K=5, min_per=8):
    """把测试点按空间分位切成 KxK 格，返回格间覆盖率的离散度。"""
    if len(Xte) < K * K * min_per:
        return np.nan, np.nan, np.nan, np.nan, 0
    q = np.linspace(0, 1, K + 1)[1:-1]
    bx = np.searchsorted(np.quantile(Xte[:, 0], q), Xte[:, 0], side="right")
    by = np.searchsorted(np.quantile(Xte[:, 1], q), Xte[:, 1], side="right")
    gid = bx * K + by
    cs = [covered[gid == g].mean() for g in range(K * K) if (gid == g).sum() >= min_per]
    if len(cs) < 4:
        return np.nan, np.nan, np.nan, np.nan, len(cs)
    cs = np.asarray(cs)
    return (float(cs.std(ddof=1)), float(cs.min()), float(cs.max()),
            float(np.mean((cs >= 0.80) & (cs <= 1.00))), len(cs))


def main():
    t_all = time.time()
    print("== load field ==", flush=True)
    pm, lat, lon, masks, det = SC.load_field()

    for dname in ("YRD4", "EastChina"):
        lon0, lon1, lat0, lat1, kind = M.DOMAINS[dname]
        P = M.sample_points(masks[kind], lat, lon, (lon0, lon1, lat0, lat1),
                            N_TARGET, offset=(0.0, 0.0))
        z_raw = pm[P["rows"], P["cols"]].astype(np.float64)
        lm = det.sample(P["rows"], P["cols"], 1.0)
        X = np.column_stack([P["xn"], P["yn"]])
        D_km = P["D_km"]; sp = P["mean_spacing_km"]
        n = len(z_raw)
        print("\n### %s  n=%d  D=%.0fkm  spacing=%.1fkm" % (dname, n, D_km, sp), flush=True)

        for prep, y in (("detrend", z_raw - lm), ("raw", z_raw)):
            for rep in range(R_REP):
                t0 = time.time()
                rng = np.random.default_rng(4000 + rep)
                perm = rng.permutation(n)
                n_te = int(round(FRAC_TEST * n))
                n_ca = int(round(FRAC_CAL * n))
                te = perm[:n_te]; ca = perm[n_te:n_te + n_ca]; tr = perm[n_te + n_ca:]

                pred, sd, fit = SC.gp_train_predict(X[tr], y[tr],
                                                    np.vstack([X[ca], X[te]]))
                mca, sca = pred[:len(ca)], sd[:len(ca)]
                mte, ste = pred[len(ca):], sd[len(ca):]
                V = np.abs(y[ca] - mca) / sca

                # 到最近标定点的距离（内插程度的证据）
                from scipy.spatial import cKDTree
                dnn = cKDTree(X[ca]).query(X[te], k=1)[0]
                nn_med = float(np.median(dnn) * D_km)

                tG = SC.gscp_t(V)
                covG_all = np.abs(y[te] - mte) <= tG * ste
                gs, gmin, gmax, band, ng = gap_metrics(X[te], covG_all)

                def rec(mult, eta, ts, tag):
                    cov = np.abs(y[te] - mte) <= ts * ste
                    finite = np.isfinite(ts)
                    wg = 2 * tG * ste
                    ws = 2 * ts * ste
                    ratio = float(np.mean(ws[finite] / wg[finite])) if finite.any() else np.nan
                    gs2, gmin2, gmax2, band2, ng2 = gap_metrics(X[te], cov)
                    return dict(domain=dname, prep=prep, rep=rep, eta_mult=tag,
                                eta=round(float(eta) * D_km, 1), phi=round(fit["phi_norm"], 5),
                                n_tr=len(tr), n_cal=len(ca), n_test=len(te),
                                nn_dist_med_km=round(nn_med, 1),
                                cov=round(float(cov.mean()), 4),
                                cov_gap_std=round(gs2, 4) if gs2 == gs2 else np.nan,
                                cov_gap_min=round(gmin2, 3) if gmin2 == gmin2 else np.nan,
                                cov_gap_max=round(gmax2, 3) if gmax2 == gmax2 else np.nan,
                                frac_gap_in_band=round(band2, 3) if band2 == band2 else np.nan,
                                n_gaps=ng2,
                                width_mean=round(float(np.mean(ws[finite])), 4) if finite.any() else np.inf,
                                frac_finite=round(float(finite.mean()), 4),
                                ratio_mean=round(ratio, 4) if ratio == ratio else np.nan,
                                secs=round(time.time() - t0, 1))

                # 等权基线（tag=inf）
                append(rec(1.0, np.inf, np.full(len(te), tG), "baseline"))
                print("  [%s/%s r%d] baseline  cov=%.3f gapSD=%.3f band=%.2f nn=%.0fkm" %
                      (dname, prep, rep, covG_all.mean(),
                       gs if gs == gs else np.nan, band if band == band else np.nan, nn_med),
                      flush=True)

                for mult in ETA_MULT:
                    eta = mult * fit["phi_norm"]
                    ts = SC.slscp_t(V, X[te], X[ca], eta)
                    r = rec(mult, eta, ts, mult)
                    append(r)
                    print("     eta=%4.1fphi cov=%.3f gapSD=%.3f band=%.2f width=%.3f "
                          "fin=%.2f ratio=%.3f" %
                          (mult, r["cov"], r["cov_gap_std"], r["frac_gap_in_band"],
                           r["width_mean"] if np.isfinite(r["width_mean"]) else -1,
                           r["frac_finite"], r["ratio_mean"] if r["ratio_mean"] == r["ratio_mean"] else -1),
                          flush=True)
                del pred, sd, mca, sca, mte, ste, V
                gc.collect()

        del z_raw, lm, X; gc.collect()

    print("\n== DONE in %.0fs ==" % (time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
