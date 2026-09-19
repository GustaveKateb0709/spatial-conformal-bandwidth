# spatial-conformal-bandwidth

Analysis pipeline for a study of **when spatially weighted conformal prediction actually
changes prediction intervals** — and what it costs when it does.

The paper asks three questions on a 0.01° gridded PM2.5 product over four nested domains in
China: what makes the spatial weighting informative (grid resolution, or trend removal);
what it buys and what it costs; and which fitted quantities can be trusted at feasible
sample sizes. The main results are that removing the large-scale trend, not refining the
grid, is what moves the kernel bandwidth into the informative regime, and that the benefit
holds only under interpolation, not extrapolation.

## Contents

```
src/
  fit_gp_variogram.py       maximum-likelihood fits of an exponential-covariance Gaussian
                            process over four nested domains; design matrix
  identifiability_test.py   synthetic-field experiment: which correlation lengths can be
                            recovered at a given sampling density
  stability_run.py          sensitivity of the fitted correlation length to the detrending
                            window and to the lattice phase
  spatial_conformal.py      equal-weight and spatially weighted conformal intervals under
                            two hold-out designs
  interp_sweep.py           interpolation design with a bandwidth scan (independent
                            implementation, used for cross-checking)
results/
  design_from_runlog.csv    fitted phi / domain diagonal, by domain, treatment and sample size
  identifiability.csv       recovered versus true correlation length
  stability.csv             window and lattice-phase sensitivity
  results_scenarios.csv     coverage, conditional-coverage spread, width and infinite-interval
                            fraction across scenarios and bandwidths
  paper_numbers.json        every number quoted in the manuscript, traced to the files above
requirements.txt
LICENSE                     MIT
```

## Data

The gridded product is the **Surface PM2.5 V6.GL.03** dataset of the Atmospheric Composition
Analysis Group (Washington University in St. Louis), distributed through the AWS Open Data
bucket `s3://satpmdata/` under a **CC BY 4.0** licence. It can be retrieved without
authentication or registration:

```bash
curl -L --fail -o V6GL03.CNNPM25.AS.201501-201512.nc \
  "https://satpmdata.s3.amazonaws.com/V6GL03/FineResolution/AS/Annual/V6GL03.CNNPM25.AS.201501-201512.nc"
```

Administrative boundaries (GADM v4.1) are used for land masking and cartography only and are
**not redistributed** here. The standard national base map is likewise not redistributed.

## Running the analysis

```bash
pip install -r requirements.txt

export P19_DATA_RAW=/path/to/raw          # must contain acag/... and gadm/...
cd src
python fit_gp_variogram.py        # design matrix
python identifiability_test.py    # identifiability experiment
python stability_run.py           # window and phase sensitivity
python spatial_conformal.py all   # interval construction and scenario comparison
python interp_sweep.py            # interpolation design with bandwidth scan
```

Each script writes CSV/JSON into its own directory; the runs are deterministic (fixed seeds
are set inside the scripts). All four domains are subsets of the same Asia grid file, so a
single download is sufficient.

## Notes on the method

- Covariance parameters are estimated by **maximum likelihood**, not from an empirical
  variogram: on a field containing a large-scale gradient the two disagree substantially.
- Points are placed on a **regular lattice**, not sampled at random. Random sampling at a fixed
  count produces mean spacings comparable to the correlation lengths of interest, which by
  itself forces the kernel weights toward uniformity.
- Detrending subtracts a local mean computed over a square window (**1.0° ≈ 111 km** by
  default). The window is a modelling choice; `stability_run.py` reports its effect.
- Configurations in which more than 10% of test intervals are unbounded are flagged as
  unusable. Such configurations can show near-perfect coverage in every region — an artefact of
  unbounded intervals, not a good result — and are excluded from bandwidth selection.

## Licence

MIT. See `LICENSE`.
