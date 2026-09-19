#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity of the fitted correlation length to window and lattice phase.
import os
核心操作是"剥离大尺度趋势"，必须证明结论不是由这一个参数凑出来的。
扫描两个轴（每域）：
  window : 去趋势窗口 0.5 / 1.0 / 2.0 度（约 55 / 111 / 222 km）
  offset : 点阵相位 0 / 0.33 / 0.67（避免"网格对齐"造成的偶然性）
另加 raw 基准作对照。
增量落盘（每格写完即 flush），中途挂掉不丢。
"""
import os, sys, csv, gc, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_gp_variogram import (load_json_from_zip, collect_polys, rasterize,
                              Detrender, fit_gp, sample_points, DOMAINS, YRD_PROV,
                              NC, GADM0, GADM1, OUT)

WIN = [0.5, 1.0, 2.0]
OFF = [(0.0, 0.0), (0.33, 0.33), (0.67, 0.67)]
N_STAB = 2000
CSV_PATH = os.path.join(OUT, "stability.csv")
FIELDS = ["domain", "axis", "setting", "prep", "n_actual", "mean_spacing_km",
          "D_km", "phi_km", "ratio_phi_domain", "sigma2", "tau2", "sigma2_frac",
          "logml", "converged", "bound_hit", "cond", "secs"]


def append_row(r):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(r)


def main():
    t0 = time.time()
    print("== load ==", flush=True)
    import netCDF4
    ds = netCDF4.Dataset(NC)
    pm = ds.variables["PM25"][:].astype(np.float32)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    ds.close()
    H, W = pm.shape
    geo = (float(lon[0]), float(lat[0]), float(lon[1] - lon[0]),
           float(lat[1] - lat[0]), W, H)

    g0 = load_json_from_zip(GADM0); o0, i0 = collect_polys(g0)
    china = rasterize(o0, i0, geo); del g0, o0, i0; gc.collect()
    g1 = load_json_from_zip(GADM1); o1, i1 = collect_polys(g1, name_field="NAME_1", names=YRD_PROV)
    yrd = rasterize(o1, i1, geo); del g1, o1, i1; gc.collect()
    det = Detrender(pm, china, lat, lon, cf=5)
    masks = {"china": china, "provinces": yrd}
    print("== ready ==", flush=True)

    for dname, (lon0, lon1, lat0, lat1, kind) in DOMAINS.items():
        bbox = (lon0, lon1, lat0, lat1)
        mask = masks[kind]
        P = sample_points(mask, lat, lon, bbox, N_STAB, offset=(0.0, 0.0))
        z_raw = pm[P["rows"], P["cols"]].astype(np.float64)

        # (0) raw 基准
        r = fit_gp(P["xn"], P["yn"], z_raw, P["D_km"])
        r.update(domain=dname, axis="none", setting="raw", prep="raw",
                 n_actual=P["n_actual"], mean_spacing_km=round(P["mean_spacing_km"], 2))
        r["sigma2_frac"] = round(r["sigma2"] / (r["sigma2"] + r["tau2"] + 1e-12), 3)
        append_row(r)
        print("  [stab] %-11s raw        phi=%8.1f ratio=%.4f" %
              (dname, r["phi_km"], r["ratio_phi_domain"]), flush=True)

        # (a) window 轴
        for wd in WIN:
            lm = det.sample(P["rows"], P["cols"], wd)
            r = fit_gp(P["xn"], P["yn"], z_raw - lm, P["D_km"])
            r.update(domain=dname, axis="window", setting="win%.1f" % wd, prep="detrend",
                     n_actual=P["n_actual"], mean_spacing_km=round(P["mean_spacing_km"], 2))
            r["sigma2_frac"] = round(r["sigma2"] / (r["sigma2"] + r["tau2"] + 1e-12), 3)
            append_row(r)
            print("  [stab] %-11s win=%.1f    phi=%8.1f ratio=%.4f bound=%s" %
                  (dname, wd, r["phi_km"], r["ratio_phi_domain"], r["bound_hit"]), flush=True)
            del lm; gc.collect()

        # (b) offset 轴
        for off in OFF:
            Q = sample_points(mask, lat, lon, bbox, N_STAB, offset=off)
            zq = pm[Q["rows"], Q["cols"]].astype(np.float64)
            lm = det.sample(Q["rows"], Q["cols"], 1.0)
            r = fit_gp(Q["xn"], Q["yn"], zq - lm, Q["D_km"])
            r.update(domain=dname, axis="offset", setting="off%.2f" % off[0], prep="detrend",
                     n_actual=Q["n_actual"], mean_spacing_km=round(Q["mean_spacing_km"], 2))
            r["sigma2_frac"] = round(r["sigma2"] / (r["sigma2"] + r["tau2"] + 1e-12), 3)
            append_row(r)
            print("  [stab] %-11s off=%.2f   phi=%8.1f ratio=%.4f" %
                  (dname, off[0], r["phi_km"], r["ratio_phi_domain"]), flush=True)
            del zq, lm; gc.collect()

        del z_raw; gc.collect()
        print("  [stab] %s DONE  t=%.0fs" % (dname, time.time() - t0), flush=True)

    print("== ALL DONE in %.0fs ==" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
