"""
experiments_assessment.py  --  reproduce ALL the analyses in the document
"CYCLOPSv26_Model_and_NitrogenAssessment".

It runs, prints results for, and (if matplotlib is available) plots:

  A. The realistic 70-year, 1 PgC/yr deployment  -> negligible N:P / d15N change   (doc Fig 2A)
  B. Sustained 0.5 PgC/yr, shallow vs deep remineralization  -> depth dependence   (doc Fig 2B)
  C. Where N2 fixation lives (per-box) and the denitrification->fixation coupling   (doc "Where N2 fixation lives" / "spatially coupled")
  D. Per-box fixation response, deep vs shallow (1 PgC/yr) -> fixation can reverse   (doc "spatially non-uniform")
  E. Intensity sweep 0..3 PgC/yr  -> the O2->ODZ->denitrification tipping point      (doc Fig 3A)
  F. Cultivated-stoichiometry sweep (seaweed N:P 16..40)  -> fixation lever only     (doc Fig 3B)
  G. Basin of deployment (Atlantic vs Pacific)  -> topology-sensitive (reported with caveat)

HOW SEAWEED IS IMPOSED.  The released built-in init_sw depth-cycling is dimensioned for the
original 18-box layout and does not run on the default 26-box grid, so -- exactly as in the
assessment -- seaweed is imposed with a transparent MANUAL forcing: each year remove N+P from
the upper-thermocline low-latitude boxes at the seaweed N:P, and remineralize that organic
matter at a chosen depth (add N+P back, consume O2 at the Redfield O:C). This is the forcing
underlying the assessment's numbers; the newer cyclops_v27.py has a fuller native version.

RUN IT:   cd CYCLOPS-CY2SW_Python ; python experiments_assessment.py
It reuses base_v26.pkl if present (created by run_example.py); otherwise it builds + spins up
the baseline first (~1 minute). Set QUICK=True for shorter, approximate horizons.

Total run time is a few minutes at full horizons (many sustained runs to near-steady state).
"""
import os, copy, numpy as np, dill

import cyclops_v26 as C
import _v26ncycle as NC

# ----------------------------------------------------------------------------- config
QUICK = False                         # True -> shorter horizons (faster, approximate)
BASE_PKL = "base_v26.pkl"
SUSTAINED_YEARS = 1200 if not QUICK else 400     # spin to near-steady before measuring
MEAS_WINDOW     = 200  if not QUICK else 100      # averaging window for fluxes
RECOVERY_YEARS  = 2000 if not QUICK else 400      # follow-on after the 70-yr deployment
TG   = 14e-6 * 1e-12                  # micromol N -> TgN
KG   = C.KGPERM
ALL4 = [8, 9, 10, 11]                 # upper-thermocline low-lat boxes (nutrient uptake)
MID  = [18, 19, 20, 21]               # middle thermocline (shallow remineralization)
DEEP = [14, 15, 16, 17]               # deep boxes (deep remineralization)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ----------------------------------------------------------------- baseline & forcing
def load_or_build_baseline():
    """Return a converged (oc, at, ge, pa). Reuse base_v26.pkl if present, else build+spin."""
    if os.path.exists(BASE_PKL):
        print(f"Loading converged baseline from {BASE_PKL} ...")
        return dill.load(open(BASE_PKL, "rb"))
    print("No baseline found -- building and spinning up (~1 min) ...")
    E = NC.build(Fv=6.0); oc, at, ge, pa = E.ocean, E.atmosphere, E.geosphere, E.param
    C.run_ex(oc, at, ge, pa, 4000, 0)
    pa.Ncycle = True; C.ncycle_init_tracers(oc, pa)
    pa.watercolumndenitrification = True; pa.sedimentdenitrification = True
    C.run_ex(oc, at, ge, pa, 8500, 0)
    dill.dump((oc, at, ge, pa), open(BASE_PKL, "wb"))
    return oc, at, ge, pa


def make_force(draw, remin, F, NP, CP=800.0):
    """Return a per-year seaweed forcing closure (F in PgC/yr). remin=None -> control."""
    SW_C = F * 1e15 / 12.0
    SW_P = SW_C / CP; SW_N = NP * SW_P; SW_O2 = SW_C * 170.0 / 106.0
    def force(oc):
        if F <= 0 or remin is None:
            return
        b, t = oc.box, oc.tracer
        Vs = sum(b.vol[i] for i in draw); Vd = sum(b.vol[i] for i in remin)
        for i in draw:                                   # uptake at seaweed N:P
            w = b.vol[i] / Vs
            dP = SW_P*w*1e6/(b.vol[i]*KG); dN = SW_N*w*1e6/(b.vol[i]*KG)
            f15 = t.N15[i]/max(t.N[i],1e-30); f18 = t.NO18[i]/max(t.N[i],1e-30)
            t.P[i]-=dP; t.N[i]-=dN; t.N15[i]-=f15*dN; t.NO18[i]-=f18*dN
        for i in remin:                                  # remineralize + consume O2
            w = b.vol[i] / Vd
            dP = SW_P*w*1e6/(b.vol[i]*KG); dN = SW_N*w*1e6/(b.vol[i]*KG); dO = SW_O2*w*1e6/(b.vol[i]*KG)
            f15 = t.N15[i]/max(t.N[i],1e-30); f18 = t.NO18[i]/max(t.N[i],1e-30)
            t.P[i]+=dP; t.N[i]+=dN; t.N15[i]+=f15*dN; t.NO18[i]+=f18*dN; t.O2[i]=max(0.0,t.O2[i]-dO)
    return force


def globals_now(oc):
    """Current global mean nitrate, N:P and mean-nitrate d15N (no time-stepping)."""
    b, t = oc.box, oc.tracer; V = sum(b.vol)
    NO3 = (t.N*b.vol).sum()/V; P = (t.P*b.vol).sum()/V
    d15 = C.IsoDelN((t.N15*b.vol).sum(), (t.N*b.vol).sum())
    return NO3, NO3/P, d15


def run_scenario(base, draw, remin, F, NP, years, window=MEAS_WINDOW):
    """Run a sustained scenario from `base`; return averaged fluxes + state after `years`+window."""
    oc, at, ge, pa = copy.deepcopy(base)
    force = make_force(draw, remin, F, NP)
    for _ in range(years):
        force(oc); C.run_ex(oc, at, ge, pa, 1, 0)
    b, t = oc.box, oc.tracer
    b.fixedNtotal[:] = 0; b.lostNtotal[:] = 0; b.lostNsedtotal[:] = 0
    for _ in range(window):
        force(oc); C.run_ex(oc, at, ge, pa, 1, 0)
    V = sum(b.vol); NO3 = (t.N*b.vol).sum()/V; P = (t.P*b.vol).sum()/V
    return dict(
        NO3=NO3, NP=NO3/P, d15=C.IsoDelN((t.N15*b.vol).sum(),(t.N*b.vol).sum()),
        fix=b.fixedNtotal.sum()*TG/window, wc=b.lostNtotal.sum()*TG/window,
        sed=b.lostNsedtotal.sum()*TG/window,
        fixbox=b.fixedNtotal*TG/window, O2=np.array(t.O2), Nbox=np.array(t.N),
        anoxV=sum(b.anoxV[i] for i in C.INTERIOR_IDX)/1e15, O2_deepNPac=t.O2[17])


# ============================================================================ analyses
def analysisA_70yr(base):
    print("\n" + "="*78 + "\nA. Realistic 70-year, 1 PgC/yr deep deployment (doc Fig 2A)\n" + "="*78)
    # control and experiment in parallel, sampling N:P and d15N over time
    oc0,a0,g0,p0 = copy.deepcopy(base); oc1,a1,g1,p1 = copy.deepcopy(base)
    force = make_force(ALL4, DEEP, 1.0, 32.0)
    NP0,d0 = globals_now(oc0)[1], globals_now(oc0)[2]
    ts, dNP, dd15 = [], [], []
    total = 70 + RECOVERY_YEARS
    for yr in range(total):
        if yr < 70:
            force(oc1)                       # deploy for 70 yr
        C.run_ex(oc0,a0,g0,p0,1,0)           # control
        C.run_ex(oc1,a1,g1,p1,1,0)           # experiment
        if yr % 25 == 0 or yr == total-1:
            _,NPc,dc = globals_now(oc0); _,NPe,de = globals_now(oc1)
            ts.append(yr); dNP.append(NPe-NPc); dd15.append(de-dc)
    dNP, dd15 = np.array(dNP), np.array(dd15)
    ipk = int(np.argmax(np.abs(dNP)))
    print(f"  peak |change in N:P|   = {dNP[ipk]:+.4f}  at ~{ts[ipk]} yr after start")
    print(f"  max  |change in d15N|  = {np.abs(dd15).max():.4f} permil")
    print("  => a 70-yr deployment leaves a negligible nitrogen-cycle imprint (as in the doc).")
    return dict(ts=ts, dNP=dNP, dd15=dd15)


def analysisB_shallow_deep(base):
    print("\n" + "="*78 + "\nB. Sustained 0.5 PgC/yr: shallow vs deep remineralization (doc Fig 2B)\n" + "="*78)
    ctrl = run_scenario(base, ALL4, None, 0.0, 32.0, SUSTAINED_YEARS)
    sh   = run_scenario(base, ALL4, MID,  0.5, 32.0, SUSTAINED_YEARS)
    dp   = run_scenario(base, ALL4, DEEP, 0.5, 32.0, SUSTAINED_YEARS)
    print(f"  {'':8s} {'denit(WC+sed)':>14s} {'d-denit':>9s} {'N:P':>7s} {'d-N:P':>8s}")
    for lab, r in [("control", ctrl), ("shallow", sh), ("deep", dp)]:
        tot = r['wc']+r['sed']
        print(f"  {lab:8s} {tot:14.1f} {tot-(ctrl['wc']+ctrl['sed']):+9.1f} {r['NP']:7.2f} {r['NP']-ctrl['NP']:+8.3f}")
    print("  => deep remineralization drives the larger denitrification response and N:P drawdown.")
    return dict(ctrl=ctrl, shallow=sh, deep=dp)


def analysisC_fixation_niche(base):
    print("\n" + "="*78 + "\nC. Where N2 fixation lives + denitrification->fixation coupling (baseline)\n" + "="*78)
    r = run_scenario(base, ALL4, None, 0.0, 32.0, 0, window=300)   # baseline only
    nm = {0:'Atl_s',1:'Ind_s',2:'SPac_s',3:'NPac_s',4:'NA_s',5:'OAZ',6:'SAZ',7:'PAZ'}
    print("  surface box   fixation(TgN/yr)   surf-NO3")
    for i in range(8):
        print(f"    {nm[i]:6s}     {r['fixbox'][i]:6.1f}          {r['Nbox'][i]:6.2f}")
    print("  => fixation fires only in the low-lat boxes (SO surface = 0); NPac_s (above the ODZ)")
    print(f"     is the strongest fixer, the denitrification->overlying-fixation coupling.")
    return r


def analysisD_perbox_reverse(base):
    print("\n" + "="*78 + "\nD. Per-box fixation response, deep vs shallow (1 PgC/yr) -> can reverse sign\n" + "="*78)
    ctrl = run_scenario(base, ALL4, None, 0.0, 32.0, SUSTAINED_YEARS)
    dp   = run_scenario(base, ALL4, DEEP, 1.0, 32.0, SUSTAINED_YEARS)
    sh   = run_scenario(base, ALL4, MID,  1.0, 32.0, SUSTAINED_YEARS)
    print(f"  NPac_s fixation (TgN/yr):  control={ctrl['fixbox'][3]:.1f}"
          f"   deep={dp['fixbox'][3]:.1f} ({dp['fixbox'][3]-ctrl['fixbox'][3]:+.1f})"
          f"   shallow={sh['fixbox'][3]:.1f} ({sh['fixbox'][3]-ctrl['fixbox'][3]:+.1f})")
    print("  => deep-sinking raises fixation over the ODZ; shallow/oxic can lower it (sign flip).")
    return dict(ctrl=ctrl, deep=dp, shallow=sh)


def analysisE_tipping(base):
    print("\n" + "="*78 + "\nE. Intensity sweep 0..3 PgC/yr -> O2->ODZ->denitrification tipping point (doc Fig 3A)\n" + "="*78)
    Fs = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
    rows = []
    print(f"  {'PgC/yr':>7s} {'denit':>7s} {'anoxV(1e15 m3)':>15s} {'deepNPac O2':>12s}")
    for F in Fs:
        r = run_scenario(base, ALL4, (DEEP if F>0 else None), F, 32.0, SUSTAINED_YEARS)
        tot = r['wc']+r['sed']; rows.append((F, tot, r['anoxV'], r['O2_deepNPac']))
        print(f"  {F:7.2f} {tot:7.1f} {r['anoxV']:15.1f} {r['O2_deepNPac']:12.0f}")
    print("  => nonlinear knee near 0.5-0.75 PgC/yr where the deep N. Pacific crosses into anoxia.")
    return rows


def analysisF_stoichiometry(base):
    print("\n" + "="*78 + "\nF. Cultivated-stoichiometry sweep (seaweed N:P 16..40, 1 PgC/yr deep) (doc Fig 3B)\n" + "="*78)
    NPs = [16, 24, 32, 40]; rows = []
    print(f"  {'seaweed N:P':>11s} {'fixation':>9s} {'denit':>7s} {'ocean N:P':>10s} {'d15N':>6s}")
    for NP in NPs:
        r = run_scenario(base, ALL4, DEEP, 1.0, float(NP), SUSTAINED_YEARS)
        tot = r['wc']+r['sed']; rows.append((NP, r['fix'], tot, r['NP'], r['d15']))
        print(f"  {NP:11d} {r['fix']:9.1f} {tot:7.1f} {r['NP']:10.2f} {r['d15']:6.2f}")
    print("  => denitrification is invariant (carbon-driven); fixation rises with seaweed N:P.")
    return rows


def analysisG_basin(base):
    print("\n" + "="*78 + "\nG. Basin of deployment: Atlantic vs Pacific (1 PgC/yr) -- reported with caveat\n" + "="*78)
    ctrl = run_scenario(base, ALL4, None, 0.0, 32.0, SUSTAINED_YEARS)
    atl  = run_scenario(base, [8],  [14], 1.0, 32.0, SUSTAINED_YEARS)   # draw+remin Atlantic column
    pac  = run_scenario(base, [11], [17], 1.0, 32.0, SUSTAINED_YEARS)   # draw+remin N.Pacific column
    for lab, r in [("control", ctrl), ("Atlantic", atl), ("Pacific", pac)]:
        print(f"  {lab:9s}: denit(WC)={r['wc']:5.1f}  denit(sed)={r['sed']:5.1f}  N:P={r['NP']:.2f}")
    print("  CAVEAT: sensitive to the coarse box topology (denitrification needs BOTH low O2 AND")
    print("          organic-carbon supply); treat the Atlantic-vs-Pacific ranking as a hypothesis.")
    return dict(ctrl=ctrl, atl=atl, pac=pac)


# ================================================================================ plots
def make_figures(A, B, E, F):
    if not HAVE_MPL:
        print("\n(matplotlib not available -- skipping figures; numbers above are the results.)")
        return
    # Figure 2 (A: 70-yr time series; B: shallow vs deep)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.axvspan(0, 70, color='0.9'); ax1b = ax1.twinx()
    ax1.plot(A['ts'], A['dNP'], 'o-', color='#2E5A8C', label='change in N:P')
    ax1b.plot(A['ts'], A['dd15'], 's--', color='#C0392B', label='change in d15N')
    ax1.set_xlabel('years since deployment start'); ax1.set_ylabel('change in mean-ocean N:P', color='#2E5A8C')
    ax1b.set_ylabel('change in mean-nitrate d15N (permil)', color='#C0392B')
    ax1.set_title('(A) Realistic 70-yr, 1 PgC/yr deployment', fontsize=10)
    labs = ['control','shallow','deep']
    tot = [B[k]['wc']+B[k]['sed'] for k in ['ctrl','shallow','deep']]
    dnp = [B[k]['NP']-B['ctrl']['NP'] for k in ['ctrl','shallow','deep']]
    x = np.arange(3); ax2.bar(x-0.2, tot, 0.4, color='#2E5A8C', label='total denit (TgN/yr)')
    ax2b = ax2.twinx(); ax2b.bar(x+0.2, dnp, 0.4, color='#8E44AD', label='change in N:P')
    ax2.set_xticks(x); ax2.set_xticklabels(labs); ax2.set_ylabel('denitrification (TgN/yr)', color='#2E5A8C')
    ax2b.set_ylabel('change in N:P', color='#8E44AD'); ax2.set_title('(B) Sustained 0.5 PgC/yr: depth', fontsize=10)
    plt.tight_layout(); plt.savefig('assessment_fig2.png', dpi=140); plt.close()

    # Figure 3 (A: tipping point; B: stoichiometry)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    Ff=[r[0] for r in E]; den=[r[1] for r in E]; anx=[r[2] for r in E]
    ax1.plot(Ff, den, 'o-', color='#2E5A8C', label='total denit'); ax1c = ax1.twinx()
    ax1c.plot(Ff, anx, '^-', color='#C0392B', label='anoxic volume')
    ax1.set_xlabel('sustained export (PgC/yr)'); ax1.set_ylabel('denit (TgN/yr)', color='#2E5A8C')
    ax1c.set_ylabel('anoxic volume (1e15 m3)', color='#C0392B'); ax1.set_title('(A) Tipping point', fontsize=10)
    NPs=[r[0] for r in F]; fx=[r[1] for r in F]; dn=[r[2] for r in F]
    ax2.plot(NPs, fx, 'o-', color='#27AE60', label='N2 fixation'); ax2.plot(NPs, dn, 's--', color='#888', label='denitrification')
    ax2.set_xlabel('cultivated seaweed N:P'); ax2.set_ylabel('flux (TgN/yr)'); ax2.set_xticks(NPs)
    ax2.set_title('(B) Stoichiometry lever', fontsize=10); ax2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig('assessment_fig3.png', dpi=140); plt.close()
    print("\nSaved figures: assessment_fig2.png, assessment_fig3.png")


def main():
    base = load_or_build_baseline()
    print("\nBaseline:  NO3=%.2f  N:P=%.2f  d15N=%.2f  CO2=%.1f" % (globals_now(base[0])[0],
          globals_now(base[0])[1], globals_now(base[0])[2], base[1].ppm))
    A = analysisA_70yr(base)
    B = analysisB_shallow_deep(base)
    analysisC_fixation_niche(base)
    analysisD_perbox_reverse(base)
    E = analysisE_tipping(base)
    F = analysisF_stoichiometry(base)
    analysisG_basin(base)
    make_figures(A, B, E, F)
    print("\nAll assessment analyses complete.")


if __name__ == "__main__":
    main()
