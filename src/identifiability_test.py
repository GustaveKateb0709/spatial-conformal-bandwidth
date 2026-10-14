#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identifiability of the correlation length at given sampling densities.
import os
Why this step is needed
-----------------------
Earlier fits show: after detrending, phi ≈ 16-28 km across domains, ratio ≈ 0.004-0.021.
But the lattice spacings are:
  Nationwide 5583 x 3981 km, n≈1400 -> spacing ~114 km; n≈2575 -> spacing ~87 km
  YRD         790 x  893 km, n≈1290 -> spacing ~22 km;  n≈3438 -> spacing ~13 km
If phi is only 20-30 km while the spacing is 90-110 km, the covariance between any two
points is exp(-90/25) ≈ 0.03, so the data barely constrain phi -- the fitted value may
just be a point on the likelihood surface fixed by numerical detail, not a data estimate.
This script reruns the same fit_gp on simulated fields with known phi at the same
sampling densities to check whether phi is recovered; the only reliable way to bound
phi identifiability. Criterion: if phi_true=20 km is overestimated severalfold (or hits
the bound), the claim "phi ≈ 20 km after detrending" is not credible at these densities;
point density or the design must change.
Usage: python3 identifiability_test.py > ident.log 2>&1
"""
import os, sys, json, time
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_gp_variogram import fit_gp          # reuse the same fitting function for consistency

OUT = os.path.dirname(os.path.abspath(__file__))

# domain geometry (km), matching the bounding boxes in fit_gp_variogram.DOMAINS
DOMAINS_KM = {
    "YRD4":       (114.9, 123.1, 27.0, 35.2),
    "MidYangtze": (110.0, 123.0, 25.0, 36.0),
    "EastChina":  (105.0, 135.0, 18.0, 45.0),
    "Nationwide": (73.0,  135.0, 18.0, 54.0),
}

PHI_TRUE_KM = [10, 20, 50, 100, 200, 500, 1000, 2000]
# use the low-density level per domain (the low end of real n_actual: the worst case);
# Nationwide also runs a high-density control to separate "large domain" from "sparse points".
N_LEVELS = {
    "YRD4":       [1290],
    "MidYangtze": [1100],
    "EastChina":  [1230],
    "Nationwide": [1400, 2600],
}


def grid_m(n):
    """Regular lattice for simulation: n = m^2, so m = round(sqrt(n))."""
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
            print("\n=== %s  m=%d  n=%d  spacing (lon %.0f km / lat %.0f km, mean %.0f km)  D=%.0f km ==="
                  % (dom, m, n_act, sp_x, sp_y, sp_mean, D_km), flush=True)
            xn, yn = x / D_km, y / D_km
            for phi_true in PHI_TRUE_KM:
                if phi_true < sp_mean * 0.5:
                    note = "  <<< phi_true far below point spacing; unidentifiable in theory"
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
                # incremental flush to disk; nothing lost if the run dies midway
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
