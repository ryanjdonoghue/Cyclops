# CYCLOPS-v26 — commented, shareable, runnable bundle

This is a **self-contained** copy of the CYCLOPS-v26 ocean biogeochemistry model: an ocean
carbon + nitrogen box model with nitrate isotopes and a seaweed-CDR capability, on a
vertically-resolved (26-box) thermocline. The model source here is **fully commented** (an
explanatory comment on every function and section) and is **behaviour-identical** to the
released model — verified by stripping the comments (recovers the original byte-for-byte) and
by reproducing identical outputs on a 300-year test.

## What's in the box

```
CYCLOPSv26_commented_shareable/
├── CYCLOPS-CY2SW_Python/          # the code (run from here)
│   ├── cyclops_v26.py            # the model engine — FULLY COMMENTED (read this to learn the model)
│   ├── _v26ncycle.py             # build(): assembles a ready-to-run model instance
│   ├── _v26setup.py              # expand_thermocline(): builds the 26-box geometry/circulation
│   ├── run_experiments.py        # one helper (seafloor hypsometry for sediment denitrification)
│   └── run_example.py            # ← RUN THIS: build, spin up, print diagnostics, demo a seaweed run
├── CYCLOPS-CY2SW_C++/            # input data the model reads (keep next to the Python folder)
│   ├── GITCY/CYCLOPSpp_INPUTunicode_silica.txt      # base geometry / initial conditions
│   └── CIRCULATIONS/NADW_HainGBC2010_MYNADW2.txt    # the ocean transport matrix
├── CYCLOPSv26_HowToRun_Guide.docx # step-by-step guide (build, run, read outputs, experiments)
├── requirements.txt
└── README.md                     # this file
```

Keep the `CYCLOPS-CY2SW_Python/` and `CYCLOPS-CY2SW_C++/` folders **side by side** — the code
finds its data files at `../CYCLOPS-CY2SW_C++` relative to the Python folder.

## Requirements

Python 3.10+ and two packages:

```
pip install -r requirements.txt      # numpy, dill
```

## Run it

```
cd CYCLOPS-CY2SW_Python
python run_example.py
```

This builds the model, spins up the carbon cycle then the nitrogen cycle to steady state,
saves the converged baseline to `base_v26.pkl`, prints the standard diagnostics, and runs a
short illustrative seaweed deployment. Full spin-up takes ~1 minute (the ocean nitrogen
reservoir adjusts over millennia); set `QUICK = True` at the top of `run_example.py` for a
faster, less-converged demo.

A correctly converged baseline reads approximately: **mean nitrate ≈ 30.4 µmol/kg, N:P ≈ 14.1,
mean-nitrate δ¹⁵N ≈ 5.0‰, atmospheric CO₂ ≈ 279 ppm**, with **N₂ fixation ≈ total
denitrification ≈ 38 TgN/yr** and the oxygen-deficient zone in the North-Pacific thermocline
(boxes 11 and 21).

## Learn / use the model

- To **understand** the model, read `CYCLOPS-CY2SW_Python/cyclops_v26.py` top to bottom — each
  section and function has an explanatory comment (tagged `#::`).
- To **use** it (build, run, read outputs, run seaweed / other CDR experiments, restart from a
  saved baseline, configuration knobs, sanity checks, troubleshooting), follow
  `CYCLOPSv26_HowToRun_Guide.docx`.

## Notes

- Only the input data needed for a standard preindustrial build + run and seaweed experiments
  is bundled. Deglacial / radiocarbon forcing files (used only by `load_dgl_forcing` /
  `load_q14c` / `change_circ` for those specialized experiments) are **not** included; the
  documented workflow does not need them.
- The vertical resolution is set by the module constant `NLAYER` at the top of
  `cyclops_v26.py` (`3` = the 26-box default; `1` = the original 18-box model).
