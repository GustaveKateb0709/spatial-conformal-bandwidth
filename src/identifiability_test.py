#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identifiability of the correlation length at given sampling densities.
import os
为什么必须做这一步
------------------
前一批拟合显示：去趋势后各域 phi ≈ 16-28 km，ratio ≈ 0.004-0.021。
但采样点阵的间距是这样的——
  全国域 5583 x 3981 km，n≈1400 -> 点间距 ~114 km；n≈2575 -> 间距 ~87 km
  长三角  790 x  893 km，n≈1290 -> 点间距 ~22 km；  n≈3438 -> 间距 ~13 km
如果 phi 只有 20-30 km，而点间距 90-110 km，则任意两点的协方差 exp(-90/25) ≈ 0.03，
数据几乎无法约束 phi —— 拟合值可能只是似然面上的一个由数值细节决定的点，不是数据估计。
本脚本用**已知 phi 的模拟场**，在同样的采样密度上重跑同一个 fit_gp，
看能不能把 phi 恢复回来。这是判断"phi 可辨识下界"的唯一可靠办法。
判据：若 phi_true=20 km 时估计值系统性地高估数倍（或撞上界），
则"去趋势后 phi≈20 km"这一结论在本采样密度下**不可信**，必须提高点密度或改变设计。
用法：python3 identifiability_test.py > ident.log 2>&1
"""
import os, sys, json, time
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_gp_variogram import fit_gp          # 复用同一个拟合函数，保证同口径

OUT = os.path.dirname(os.path.abspath(__file__))

# 域几何（km），与 fit_gp_variogram.DOMAINS 里的包围盒一致
DOMAINS_KM = {
    "YRD4":       (114.9, 123.1, 27.0, 35.2),
    "MidYangtze": (110.0, 123.0, 25.0, 36.0),
    "EastChina":  (105.0, 135.0, 18.0, 45.0),
    "Nationwide": (73.0,  135.0, 18.0, 54.0),
}

PHI_TRUE_KM = [10, 20, 50, 100, 200, 500, 1000, 2000]
# 每域取低密度那一档（对应真实 n_actual 的低值，是该域最不利的情形）；
# 全国额外跑一档高密度作对照，用来分离"域太大"与"点太稀"两个因素。
N_LEVELS = {
    "YRD4":       [1290],
    "MidYangtze": [1100],
    "EastChina":  [1230],
    "Nationwide": [1400, 2600],
}


def grid_m(n):
    """模拟用规则格点：n = m^2，故 m = round(sqrt(n))。"""
    return int(round(np.sqrt(n)))
S2_TRUE, TAU2_TRUE = 5.0, 1.3
SEED = 20260918


def domain_km(lon0, lon1, lat0, lat1):
    lat_c = 0.5 * (lat0 + lat1)
    kx = 111.320 * np.cos(np.deg2rad(lat_c))
    ky = 110.574
    Wd = (lon1 - lon0) * kx
    Ht = (lat1 - lat0) * ky
    return Wd, Ht


def lattice(Wd, Ht, m):
    gx = np.linspace(0.0, Wd, m)
    gy = np.linspace(0.0, Ht, m)
    X, Y = np.meshgrid(gx, gy)
    return X.ravel(), Y.ravel()


def simulate(x, y, phi_true, s2, tau2, seed):
    n = len(x)
    d = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    C = s2 * np.exp(-d / phi_true)
    C.flat[::n + 1] += tau2
    del d
    try:
        L = np.linalg.cholesky(C + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C + 1e-8 * np.eye(n))
        L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    del C
    rng = np.random.default_rng(seed)
    z = L @ rng.standard_normal(n)
    del L
    return z


def main():
    t0 = time.time()
    rows = []
    for dom, box in DOMAINS_KM.items():
        Wd, Ht = domain_km(*box)
        D_km = float(np.hypot(Wd, Ht))
        for n in N_LEVELS[dom]:
            m = grid_m(n)
            x, y = lattice(Wd, Ht, m)
            n_act = len(x)
            sp_x, sp_y = Wd / (m - 1), Ht / (m - 1)
            sp_mean = float(0.5 * (sp_x + sp_y))
            print("\n=== %s  m=%d  n=%d  间距(经向 %.0f km / 纬向 %.0f km, 均值 %.0f km)  D=%.0f km ==="
                  % (dom, m, n_act, sp_x, sp_y, sp_mean, D_km), flush=True)
            xn, yn = x / D_km, y / D_km
            for phi_true in PHI_TRUE_KM:
                if phi_true < sp_mean * 0.5:
                    note = "  <<< phi_true 远小于点间距，理论不可辨识"
                else:
                    note = ""
                z = simulate(x, y, phi_true, S2_TRUE, TAU2_TRUE, SEED + phi_true)
                r = fit_gp(xn, yn, z, D_km)
                rec = dict(domain=dom, m=m, n_actual=n_act, spacing_km=round(sp_mean, 1),
                           D_km=round(D_km, 1),
                           phi_true_km=phi_true,
                           phi_est_km=round(r["phi_km"], 1),
                           ratio_est=round(r["ratio_phi_domain"], 4),
                           ratio_true=round(phi_true / D_km, 4),
                           sigma2=round(r["sigma2"], 3), tau2=round(r["tau2"], 3),
                           sigma2_frac=round(r["sigma2"] / (r["sigma2"] + r["tau2"] + 1e-12), 3),
                           logml=round(r["logml"], 2),
                           converged=r["converged"], bound_hit=r["bound_hit"],
                           secs=r["secs"])
                rows.append(rec)
                hdr = "  phi_true=%6.0f km -> phi_est=%8.1f km  (true ratio %.4f -> est %.4f)  s2f=%.2f bound=%s%s"
                print(hdr % (phi_true, r["phi_km"], phi_true / D_km,
                             r["ratio_phi_domain"], rec["sigma2_frac"],
                             r["bound_hit"], note), flush=True)
                # 增量落盘，中途挂掉也不丢
                with open(os.path.join(OUT, "identifiability.csv"), "w") as f:
                    import csv
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    for rr in rows:
                        w.writerow(rr)

    print("\n== DONE in %.0fs ==" % (time.time() - t0), flush=True)
    with open(os.path.join(OUT, "identifiability.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
