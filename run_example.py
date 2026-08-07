"""
run_example.py  --  a ready-to-run demonstration of CYCLOPS-v26.

What it does, in order:
  1. builds a fresh model instance,
  2. spins up the carbon cycle, then switches the nitrogen cycle on and equilibrates it,
  3. saves the converged baseline to base_v26.pkl (so you never have to spin up again),
  4. prints the standard diagnostics (nitrate, N:P, d15N, CO2, fixation, denitrification, O2 by box),
  5. runs a short illustrative seaweed-CDR deployment and prints the response.

Run it from THIS folder:
    cd CYCLOPS-CY2SW_Python
    python run_example.py

Full spin-up takes roughly a minute (the ocean nitrogen reservoir has a multi-millennial
adjustment time). Set QUICK=True below for a faster, less-converged demo.
"""
import numpy as np, dill
import cyclops_v26 as C          # the (commented) model engine
import _v26ncycle as NC          # build() helper

QUICK = False                    # True -> shorter spin-up (approximate), False -> fully converged
CARBON_YEARS = 4000
NITROGEN_YEARS = 8500 if not QUICK else 3000
TG = 14e-6 * 1e-12               # micromol N -> TgN  (for reporting fluxes)


def diagnostics(oc, at, ge, pa, window=300, label=""):
    """Zero the flux counters, run `window` years, and print the standard readouts."""
    b, t = oc.box, oc.tracer
    b.fixedNtotal[:] = 0; b.lostNtotal[:] = 0; b.lostNsedtotal[:] = 0
    C.run_ex(oc, at, ge, pa, window, 0)
    V = sum(b.vol)
    NO3 = (t.N * b.vol).sum() / V
    P = (t.P * b.vol).sum() / V
    d15 = C.IsoDelN((t.N15 * b.vol).sum(), (t.N * b.vol).sum())
    fix = b.fixedNtotal.sum() * TG / window
    wc = b.lostNtotal.sum() * TG / window
    sed = b.lostNsedtotal.sum() * TG / window
    print(f"\n----- {label} -----")
    print(f"  mean nitrate = {NO3:6.2f} umol/kg    N:P = {NO3/P:5.2f}    mean d15N = {d15:4.2f} permil")
    print(f"  atmospheric CO2 = {at.ppm:6.1f} ppm")
    print(f"  N2 fixation = {fix:5.1f}   denit(water-column) = {wc:5.1f}   denit(sediment) = {sed:5.1f}   TgN/yr")
    print(f"  O2 upper thermocline 8-11 : {[round(t.O2[i]) for i in range(8,12)]}")
    print(f"  O2 mid   thermocline 18-21: {[round(t.O2[i]) for i in range(18,22)]}   <- ODZ in the North Pacific (box 11 & 21)")
    return dict(NO3=NO3, NP=NO3/P, d15=d15, fix=fix, wc=wc, sed=sed)


def main():
    # 1-2. build + spin up
    print("Building model ...")
    E = NC.build(Fv=6.0)
    oc, at, ge, pa = E.ocean, E.atmosphere, E.geosphere, E.param

    print(f"Spinning up carbon cycle ({CARBON_YEARS} yr) ...")
    C.run_ex(oc, at, ge, pa, CARBON_YEARS, 0)

    print("Switching nitrogen cycle on ...")
    pa.Ncycle = True
    C.ncycle_init_tracers(oc, pa)
    pa.watercolumndenitrification = True
    pa.sedimentdenitrification = True

    print(f"Equilibrating nitrogen cycle ({NITROGEN_YEARS} yr) ...")
    C.run_ex(oc, at, ge, pa, NITROGEN_YEARS, 0)

    # 3. save
    dill.dump((oc, at, ge, pa), open("base_v26.pkl", "wb"))
    print("Saved converged baseline -> base_v26.pkl")

    # 4. baseline diagnostics
    base = diagnostics(oc, at, ge, pa, label="BASELINE (no seaweed)")

    # 5. a short illustrative seaweed deployment (1 PgC/yr, deep remineralization, 70 years).
    #    NOTE: the model's built-in init_sw depth-cycling is dimensioned for the original 18-box
    #    layout and does not run on the default 26-box (NLAYER=3) grid. So we impose the seaweed
    #    here with a transparent MANUAL forcing that works on the resolved grid: each year remove
    #    N+P from the upper-latitude thermocline at the seaweed N:P (=32), and remineralize that
    #    organic matter in the deep boxes (adding N+P back and consuming O2 at the Redfield O:C).
    #    (cyclops_v27.py provides a fuller native version that respires the carbon in the OMZ
    #    manager; this manual version is enough to illustrate the response.)
    import copy
    oc2, at2, ge2, pa2 = copy.deepcopy((oc, at, ge, pa))
    KG = C.KGPERM
    SRC = [8, 9, 10, 11]           # upper-thermocline low-latitude boxes (nutrient uptake)
    DEEP = [14, 15, 16, 17]        # deep boxes (deep remineralization of the sinking seaweed)
    SW_C = 1.0e15 / 12.0           # 1 PgC/yr in mol C
    SW_P = SW_C / 800.0; SW_N = 32.0 * SW_P; SW_O2 = SW_C * 170.0 / 106.0

    def seaweed_year(o):
        b, t = o.box, o.tracer
        Vs = sum(b.vol[i] for i in SRC); Vd = sum(b.vol[i] for i in DEEP)
        for i in SRC:                          # uptake: remove N+P at seaweed N:P
            w = b.vol[i] / Vs
            dP = SW_P * w * 1e6 / (b.vol[i] * KG); dN = SW_N * w * 1e6 / (b.vol[i] * KG)
            f15 = t.N15[i] / max(t.N[i], 1e-30); f18 = t.NO18[i] / max(t.N[i], 1e-30)
            t.P[i] -= dP; t.N[i] -= dN; t.N15[i] -= f15 * dN; t.NO18[i] -= f18 * dN
        for i in DEEP:                         # remineralize at depth + consume O2
            w = b.vol[i] / Vd
            dP = SW_P * w * 1e6 / (b.vol[i] * KG); dN = SW_N * w * 1e6 / (b.vol[i] * KG)
            dO = SW_O2 * w * 1e6 / (b.vol[i] * KG)
            f15 = t.N15[i] / max(t.N[i], 1e-30); f18 = t.NO18[i] / max(t.N[i], 1e-30)
            t.P[i] += dP; t.N[i] += dN; t.N15[i] += f15 * dN; t.NO18[i] += f18 * dN
            t.O2[i] = max(0.0, t.O2[i] - dO)

    for yr in range(70):                       # 70-year deployment
        seaweed_year(oc2); C.run_ex(oc2, at2, ge2, pa2, 1, 0)
    C.run_ex(oc2, at2, ge2, pa2, 300, 0)       # let the signal develop
    sw = diagnostics(oc2, at2, ge2, pa2, label="AFTER 70-yr seaweed deployment (1 PgC/yr, deep)")
    print(f"\n  seaweed effect: d(N:P) = {sw['NP']-base['NP']:+.3f}   d(d15N) = {sw['d15']-base['d15']:+.3f} permil")
    print("\nDone. Re-load the baseline any time with:")
    print("    import dill, cyclops_v26 as C")
    print("    oc, at, ge, pa = dill.load(open('base_v26.pkl','rb'))")


if __name__ == "__main__":
    main()
