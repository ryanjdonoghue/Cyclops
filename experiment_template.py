"""
experiment_template.py  --  COPY THIS FILE for each new experiment.

Workflow:
  1. Duplicate this file, e.g.  cp experiment_template.py my_experiment.py
  2. Edit only the block marked  ==== YOUR EXPERIMENT ====  near the bottom.
  3. Run it:  python my_experiment.py

The top of the file is reusable boilerplate you normally don't touch:
  - load_baseline()  : reuse the converged base_v26.pkl (or build+spin it up once),
  - diagnostics()    : print the standard readouts for a model state,
  - seaweed()        : impose one model-year of a depth-cycled seaweed deployment.

Box index reference (used when you pick which boxes to force / read):
  surface  0 Atl  1 Ind  2 SPac  3 NPac  4 NAtl  5 OAZ  6 SAZ  7 PAZ
  upper thermocline  8 Atl  9 Ind 10 SPac 11 NPac      (11 & 21 = the ODZ)
  middle thermocline 18 Atl 19 Ind 20 SPac 21 NPac
  lower  thermocline 22 Atl 23 Ind 24 SPac 25 NPac
  deep   12 NAtl(NCW) 13 SOcean(CDW) 14 Atl 15 Ind 16 SPac 17 NPac
"""
import os, copy, numpy as np, dill
import cyclops_v26 as C
import _v26ncycle as NC

TG = 14e-6 * 1e-12          # micromol N -> TgN   (for reporting nitrogen fluxes)
KG = C.KGPERM
BASE_PKL = "base_v26.pkl"


def load_baseline():
    """Return a converged (oc, at, ge, pa). Reuse base_v26.pkl if present, else build + spin up."""
    if os.path.exists(BASE_PKL):
        print(f"Loading baseline from {BASE_PKL}")
        return dill.load(open(BASE_PKL, "rb"))
    print("Building + spinning up baseline (~1 min, done once) ...")
    E = NC.build(Fv=6.0); oc, at, ge, pa = E.ocean, E.atmosphere, E.geosphere, E.param
    C.run_ex(oc, at, ge, pa, 4000, 0)                       # carbon spin-up
    pa.Ncycle = True; C.ncycle_init_tracers(oc, pa)
    pa.watercolumndenitrification = True; pa.sedimentdenitrification = True
    C.run_ex(oc, at, ge, pa, 8500, 0)                       # nitrogen spin-up
    dill.dump((oc, at, ge, pa), open(BASE_PKL, "wb"))
    return oc, at, ge, pa


def diagnostics(oc, at, ge, pa, window=300, label=""):
    """Zero the flux counters, run `window` years, print + return the standard readouts."""
    b, t = oc.box, oc.tracer
    b.fixedNtotal[:] = 0; b.lostNtotal[:] = 0; b.lostNsedtotal[:] = 0
    C.run_ex(oc, at, ge, pa, window, 0)
    V = sum(b.vol); NO3 = (t.N*b.vol).sum()/V; P = (t.P*b.vol).sum()/V
    r = dict(NO3=NO3, NP=NO3/P,
             d15=C.IsoDelN((t.N15*b.vol).sum(), (t.N*b.vol).sum()),
             CO2=at.ppm,
             fix=b.fixedNtotal.sum()*TG/window,
             wc=b.lostNtotal.sum()*TG/window,
             sed=b.lostNsedtotal.sum()*TG/window)
    print(f"[{label}]  NO3={r['NO3']:.2f}  N:P={r['NP']:.2f}  d15N={r['d15']:.2f}  CO2={r['CO2']:.1f} ppm"
          f"  | fix={r['fix']:.1f}  denit(WC)={r['wc']:.1f}  denit(sed)={r['sed']:.1f} TgN/yr")
    return r


def seaweed(oc, draw, remin, F_PgC, NP=32.0, CP=800.0):
    """Impose ONE model-year of depth-cycled seaweed: remove N+P at the seaweed N:P from the
    `draw` boxes, and remineralize it in the `remin` boxes (add N+P back, consume O2). Call this
    each year, right before C.run_ex(oc,...,1,0). Set F_PgC=0 (or remin=None) for a control."""
    if F_PgC <= 0 or remin is None:
        return
    b, t = oc.box, oc.tracer
    SW_C = F_PgC * 1e15 / 12.0
    SW_P = SW_C / CP; SW_N = NP * SW_P; SW_O2 = SW_C * 170.0 / 106.0
    Vs = sum(b.vol[i] for i in draw); Vd = sum(b.vol[i] for i in remin)
    for i in draw:
        w = b.vol[i]/Vs
        dP = SW_P*w*1e6/(b.vol[i]*KG); dN = SW_N*w*1e6/(b.vol[i]*KG)
        f15 = t.N15[i]/max(t.N[i],1e-30); f18 = t.NO18[i]/max(t.N[i],1e-30)
        t.P[i]-=dP; t.N[i]-=dN; t.N15[i]-=f15*dN; t.NO18[i]-=f18*dN
    for i in remin:
        w = b.vol[i]/Vd
        dP = SW_P*w*1e6/(b.vol[i]*KG); dN = SW_N*w*1e6/(b.vol[i]*KG); dO = SW_O2*w*1e6/(b.vol[i]*KG)
        f15 = t.N15[i]/max(t.N[i],1e-30); f18 = t.NO18[i]/max(t.N[i],1e-30)
        t.P[i]+=dP; t.N[i]+=dN; t.N15[i]+=f15*dN; t.NO18[i]+=f18*dN; t.O2[i]=max(0.0, t.O2[i]-dO)


# =====================================================================================
# ============================  YOUR EXPERIMENT  ======================================
# =====================================================================================
# Edit only below. The example runs a control and a sustained deep-sinking seaweed
# deployment and prints the difference. Replace with whatever you want to test.

def main():
    base = load_baseline()

    # --- control (no seaweed) ---
    oc, at, ge, pa = copy.deepcopy(base)
    ctrl = diagnostics(oc, at, ge, pa, label="control")

    # --- experiment: sustained 1 PgC/yr, deep remineralization, for 1000 years ---
    oc, at, ge, pa = copy.deepcopy(base)
    DRAW  = [8, 9, 10, 11]         # take nutrients from the upper thermocline
    REMIN = [14, 15, 16, 17]       # remineralize deep  (use [18,19,20,21] for a shallow test)
    for _ in range(1000):
        seaweed(oc, DRAW, REMIN, F_PgC=1.0, NP=32.0)
        C.run_ex(oc, at, ge, pa, 1, 0)
    exp = diagnostics(oc, at, ge, pa, label="seaweed (deep, 1 PgC/yr)")

    # --- report the difference ---
    print("\nEXPERIMENT MINUS CONTROL:")
    print(f"  change in N:P  = {exp['NP']-ctrl['NP']:+.3f}")
    print(f"  change in d15N = {exp['d15']-ctrl['d15']:+.3f} permil")
    print(f"  change in denitrification = {(exp['wc']+exp['sed'])-(ctrl['wc']+ctrl['sed']):+.1f} TgN/yr")


if __name__ == "__main__":
    main()
