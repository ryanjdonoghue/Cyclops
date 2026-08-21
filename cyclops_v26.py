# =============================================================================================
#  CYCLOPS-v26  (FULLY-COMMENTED READING COPY)
# ---------------------------------------------------------------------------------------------
#  This file is a BEHAVIOUR-IDENTICAL copy of cyclops_v26.py with explanatory comments added.
#  Every line of executable code is byte-for-byte the same as the released model; the ONLY
#  additions are comment lines (starting with '#'). Comments inserted by this reading copy are
#  marked with the tag  #::  so they can be mechanically stripped to recover the original.
#
#  WHAT THE MODEL IS.  CYCLOPS is an ocean biogeochemical BOX model. The ocean is cut into a few
#  dozen well-mixed reservoirs ('boxes'); water is moved between them by a fixed transport matrix
#  (the circulation); each box carries chemical tracers (carbon, alkalinity, phosphate, oxygen,
#  silica, the C isotopes, and the nitrogen set nitrate/15N/18O) that are updated once per model
#  YEAR by the loop run_ex(). v26 extends the Yi-Hain-Sigman seaweed-CDR carbon model with an
#  explicit nitrogen cycle + nitrate isotopes (ported from the Sigman-2009 Pascal model) on a
#  vertically-resolved thermocline (NLAYER sub-layers => NB boxes).
#
#  HOW TO READ IT (top to bottom):
#    1. Module constants        -- box indices, physical constants, isotope standards.
#    2. Isotope helpers         -- delta <-> absolute-amount conversions and Rayleigh math.
#    3. State containers        -- classes holding the per-box arrays (no physics).
#    4. Setup / input           -- build a model, load geometry/circulation, configure forcings.
#    5. Physics (circulation)   -- circ_advect moves tracers between boxes.
#    6. Biology (carbon core)   -- prod/remin: production, export rain, remineralization.
#    7. Carbonate chem & air-sea gas exchange, external sources (rivers, volcano, 14C).
#    8. Nitrogen cycle          -- fixation, water-column & sediment denitrification.
#    9. OMZ sub-volume manager  -- the sub-grid suboxic pocket that lets a box-mean model
#                                  host a realistic anoxic core (the crown jewel).
#   10. run_ex                  -- the annual time-step loop that calls everything in order.
#
#  To actually RUN the model see the companion guide: 'CYCLOPSv26_HowToRun_Guide'.
# =============================================================================================

"""
CYCLOPS-CY2SW: Python translation of the CYCLOPS biogeochemical box model
with seaweed (SW) CDR interventions.

Original C++ code by Mathis Hain (mhain@ucsc.edu), first used in Hain et al 2014 (EPSL).
Seaweed extensions by Paul Yi (Paul_unpublished).
Python translation 2026.

This is an 18-box ocean model with:
  - Boxes 0-3: low-latitude surface
  - Box 4: North Atlantic / boreal surface
  - Box 5: open Antarctic Zone (OAZ) surface
  - Box 6: Subantarctic Zone (SAZ) surface
  - Box 7: Polar Antarctic Zone (PAZ) surface
  - Boxes 8-11: intermediate (below surface boxes 0-3)
  - Box 12: North Component Water (NCW) deep
  - Box 13: Deep Southern Ocean (DSO) deep
  - Boxes 14-17: deep Atlantic, Indian, South Pacific, North Pacific

Tracers: PO4, DIC, d13C, Alkalinity, NO3, Salinity, Temperature, Si, d30Si, d14C,
         regenerated PO4 (phyto, depth-cycled SW, surface SW, artif-upwell SW, SAZ SW)
"""

import numpy as np
import os
import copy
import sys

# ============================================================================
# CONSTANTS
# ============================================================================
# ---- Generalized N-layer low-latitude thermocline ----
NLAYER = 3            # number of thermocline sub-layers (>=1). 1 => original 18-box.
NCOL = 4             # low-latitude columns (boxes 8,9,10,11 are the top layer)
NBOXES = 18 + NCOL*(NLAYER-1)   # 18 base + extra layers appended at 18,19,...
NB = NBOXES
SURF_IDX=[0,1,2,3,4,5,6,7]
DEEP_IDX=[12,13,14,15,16,17]
UPTHERM_IDX=[8,9,10,11]               # top thermocline layer (layer 0)
# THERM_LAYERS[k] = list of the 4 box indices for thermocline layer k (k=0 top .. NLAYER-1 bottom)
THERM_LAYERS=[[8,9,10,11]]
for k in range(1,NLAYER):
    base=18+(k-1)*NCOL
    THERM_LAYERS.append([base,base+1,base+2,base+3])
INTERMEDIATE_IDX=[b for layer in THERM_LAYERS for b in layer]   # all thermocline boxes
INTERIOR_IDX=INTERMEDIATE_IDX+DEEP_IDX
# per-column list of layer boxes (top->bottom), and deep box beneath each column
THERM_COLUMN=[[THERM_LAYERS[k][c] for k in range(NLAYER)] for c in range(NCOL)]
DEEP_OF={8:14,9:15,10:16,11:17}
LOWTHERM_IDX=THERM_LAYERS[-1]   # backward-compat alias (bottom layer)
NSURF = 8
NINT = 4
NDEEP = 6

KGPERM = 1.028e3
MOLESATMOSPHERE = 1.77e20
SECPERYEAR = 3.1536e7
C14HALF = 5730.0
EARTHAREA = 5.1e18
NBSC14RATIO = 1.176e-12
AVOGADRO = 6.02204e23
TOTALLAYERS = 18
TOTALSURFACEAREA = 3.265e14
TOTALOCEANVOLUME = 1.35e18

# N isotope constants
RAIR = 3676.5e-06       # 15N/14N of atmospheric N2
RPDB = 1.12372e-02      # 13C/12C PDB standard (also used for NO18 and O218)
LASTSURFBOX = 8         # boxes 0-7 are surface (0-indexed)
FIRSTDEEPBOX = 13       # boxes 13-17 are deep (0-indexed: 12-17 in Python)
TOTALDEPTHLEVELS = 5    # seafloor depth levels per box

# ============================================================================
# N ISOTOPE HELPER FUNCTIONS
# ============================================================================
#::
#:: ############################################################################################
#:: ## ISOTOPE HELPER FUNCTIONS -- convert between delta-values (per mil) and
#:: ## absolute rare-isotope amounts. Tracers are stored as AMOUNTS (e.g. 15N atoms), not
#:: ## ratios, so that mixing/transport conserve mass automatically; deltas are computed
#:: ## only for output.
#:: ############################################################################################
#::
#:: --- def IsoConcN ---
#:: delta(per mil, Rair) -> absolute 15N amount for a given N pool.
def IsoConcN(DelValue):
    """Convert delta15N to fractional abundance F (using Rair standard)."""
    V1 = (DelValue / 1000.0 + 1.0) * RAIR
    return V1 / (1.0 + V1)

#::
#:: --- def IsoDelN ---
#:: absolute 15N amount + total N -> delta-15N (per mil vs atmospheric N2, Rair).
def IsoDelN(Raretope, Element):
    """Convert fractional abundances back to delta15N (using Rair standard)."""
    if Element - Raretope == 0:
        return 0.0
    return ((Raretope / (Element - Raretope)) - RAIR) * 1000.0 / RAIR

#::
#:: --- def IsoConcPDB ---
#:: delta(per mil, PDB) -> absolute rare-isotope amount (used for 13C, and for the 18O of nitrate/O2).
def IsoConcPDB(DelValue):
    """Convert delta to fractional abundance F (using Rpdb standard).
    Used for NO18 and O218 tracers."""
    V1 = (DelValue / 1000.0 + 1.0) * RPDB
    return V1 / (1.0 + V1)

#::
#:: --- def IsoDelPDB ---
#:: absolute rare-isotope amount + total -> delta (per mil vs PDB).
def IsoDelPDB(Raretope, Element):
    """Convert fractional abundances back to delta (using Rpdb standard).
    Used for NO18 and O218 tracers."""
    if Element - Raretope == 0:
        return 0.0
    return ((Raretope / (Element - Raretope)) - RPDB) * 1000.0 / RPDB

#::
#:: --- def RtoF ---
#:: isotope ratio R=rare/common -> fraction F=rare/(rare+common). Used by the Rayleigh math.
def RtoF(R):
    """Convert isotope ratio R to fractional abundance F."""
    return R / (1.0 + R)

#::
#:: --- def FtoR ---
#:: fraction F -> ratio R. Inverse of RtoF.
def FtoR(F):
    """Convert fractional abundance F to isotope ratio R."""
    if F >= 1.0:
        return 1e30
    return F / (1.0 - F)


# ============================================================================
# DATA CLASSES
# ============================================================================
#::
#:: ############################################################################################
#:: ## STATE CONTAINERS -- plain classes that hold the model's per-box arrays
#:: ## (tracers, geometry, carbonate constants, seafloor hypsometry, parameters). No physics
#:: ## here; these are the data structures the engine reads and writes each year.
#:: ############################################################################################
#::
#:: --- class TracerList ---
#:: Per-box dissolved tracers: DIC(C), alkalinity(Alk), phosphate(P), oxygen(O2), silica,
#:: 13C/14C, and the nitrogen set nitrate(N), its 15N(N15) and the 18O of nitrate(NO18), plus O2-18(O218).
class TracerList:
    """All ocean tracers (18-element vectors)."""
    def __init__(self):
        self.P = np.zeros(NB)
        self.Preg = np.zeros(NB)
        self.PregSW = np.zeros(NB)
        self.PregSurfSW = np.zeros(NB)
        self.PregArtifUpwellSW = np.zeros(NB)
        self.PregSAZSW = np.zeros(NB)
        self.C = np.zeros(NB)
        self.dc13 = np.zeros(NB)
        self.Alk = np.zeros(NB)
        self.Alkreg = np.zeros(NB)
        self.N = np.zeros(NB)
        self.Sal = np.full(NB, 34.7)
        self.Temp = np.full(NB, 5.0)
        self.Si = np.full(NB, 92.0)
        self.dc30 = np.zeros(NB)
        self.dc14 = np.zeros(NB)
        self.N15 = np.zeros(NB)      # 15N concentration (F * N)
        self.NO18 = np.zeros(NB)     # 18O in nitrate (F * N, using Rpdb)
        self.O2 = np.zeros(NB)       # dissolved O2 (mol/kg)
        self.O218 = np.zeros(NB)     # 18O in O2 (F * O2, using Rpdb)
        self.H2Od18O = np.zeros(NB)  # water d18O (permil)


#::
#:: --- class VentTracerList ---
#:: Ventilation-tracer bookkeeping (idealized age/ventilation tracers).
class VentTracerList:
    """Ventilation tracers."""
    def __init__(self):
        self.vent = np.zeros((NB, 8))
        self.trueage = np.zeros(NB)
        self.pref14Cage = np.zeros(NB)


#::
#:: --- class RainList ---
#:: The sinking 'rain' fluxes leaving the surface each year: organic P/N, CaCO3, silica, and their isotopes.
class RainList:
    """Particle rain fluxes (8-element vectors for surface boxes)."""
    def __init__(self):
        self.P = np.zeros(8)
        self.Ca = np.zeros(8)
        self.Si = np.zeros(8)
        self.d13Corg = np.zeros(8)
        self.d13Ccc = np.zeros(8)
        self.d30Si = np.zeros(8)
        self.d14Corg = np.zeros(8)
        self.d14Ccc = np.zeros(8)
        self.d15Norg = np.zeros(8)   # delta15N of organic N
        self.d18ONorg = np.zeros(8)  # delta18O of organic N (vs Rpdb)


#::
#:: --- class Basin ---
#:: Per-basin scalar properties.
class Basin:
    """Per-basin seafloor properties (arrays indexed by the 18 hypsometry DEPTH LEVELS)."""
    def __init__(self):
        self.FArea = np.zeros(18)
        self.FCa = np.zeros(18)
        self.K1 = np.zeros(20)
        self.K2 = np.zeros(20)
        self.Kb = np.zeros(20)
        self.Kw = np.zeros(20)
        self.Ks = np.zeros(20)
        self.Hsitu = np.full(20, 1e-2)
        self.CO3situ = np.zeros(20)
        self.omega = np.zeros(20)


#::
#:: --- class Cchem ---
#:: Carbonate-chemistry working variables per box (pH solver scratch).
class Cchem:
    """Carbonate chemistry solution."""
    def __init__(self):
        self.H = np.full(NB, -99.0)
        self.H2CO3 = np.full(NB, -99.0)
        self.HCO3 = np.full(NB, -99.0)
        self.CO3 = np.full(NB, -99.0)
        self.pCO2 = np.full(NB, -99.0)
        self.BOH4 = np.full(NB, -99.0)
        self.beta = np.full(NB, -99.0)
        self.omega = np.full(NB, -99.0)
        self.omegasitu = np.zeros(4)
        self.LysDepth = np.zeros(4)


#::
#:: --- class Seafloor ---
#:: Seafloor hypsometry: how much sea floor sits at each of the depth levels in each box
#:: (used by the depth-integrated sediment denitrification and CaCO3 dissolution).
class Seafloor:
    """Seafloor dissolution properties."""
    def __init__(self):
        self.NCW = Basin()
        self.DSO = Basin()
        self.Atl = Basin()
        self.Ind = Basin()
        self.SPac = Basin()
        self.NPac = Basin()
        self.SFdepth = np.zeros(18)   # 18 hypsometry depth levels (NOT boxes)
        self.Fdiss = np.zeros(4)
        self.LysDepth = np.zeros(4)
        self.CSH = np.zeros(4)


#::
#:: --- class Ktable ---
#:: Temperature/salinity-dependent carbonate equilibrium constants per surface box.
class Ktable:
    """Equilibrium constant tables."""
    def __init__(self):
        self.K0 = np.zeros(NB)
        self.K1 = np.zeros(NB)
        self.K2 = np.zeros(NB)
        self.Kb = np.zeros(NB)
        self.Ks = np.zeros(NB)


#::
#:: --- class BoxList ---
#:: Box GEOMETRY and persistent state: volumes, areas, depths (top/bottom), the mass<->concentration
#:: factors CtoN/NtoC, production targets, and the OMZ sub-volume state (CompOMZ, CompANOX, omzV, anoxV, dV...).
class BoxList:
    """Box properties."""
    def __init__(self):
        self.vol = np.zeros(NB)
        self.vol_inv = np.zeros(NB)
        self.setP = np.zeros(8)
        self.setProdP = np.full(8, -99.0)
        self.setSW = np.zeros(8)
        self.setSurfSW = np.zeros(8)
        self.setOAE = np.zeros(8)
        self.setIronFertP = np.zeros(8)
        self.setArtifUpwellSW = np.zeros(8)
        self.setSAZSW = np.zeros(8)
        self.setSi = np.zeros(8)
        self.ORGe = np.full(8, 20.0)
        self.CaRatio = np.zeros(8)
        self.top = np.zeros(NB)
        self.bottom = np.zeros(NB)
        self.NtoC = np.zeros(NB)
        self.CtoN = np.zeros(NB)
        self.Area = np.zeros(8)
        # N-cycle / OMZ state (per box)
        self.completeVolume = np.zeros(NB)  # total box volume (incl. OMZ sub-volumes)
        self.dV = np.zeros(NB)              # total suboxic sub-volume
        self.omzV = np.zeros(NB)            # OMZ sub-volume
        self.anoxV = np.zeros(NB)           # anoxic sub-volume
        self.crossArea = np.zeros(NB)       # horizontal cross-section of OMZ
        self.crossArea2 = np.zeros(NB)      # vertical cross-section of OMZ
        self.Flux = np.zeros(NB)            # internal exchange flux oxic<->OMZ
        self.Flux2 = np.zeros(NB)           # internal exchange flux OMZ<->ANOX
        self.VolFraction = np.zeros(NB)     # suboxic volume fraction
        self.CompOMZ = np.zeros((NB, 10))   # tracer concentrations in OMZ: indices [box, tracer]
        self.CompANOX = np.zeros((NB, 10))  # tracer concentrations in ANOX
        self.ComponentTray = np.zeros((NB, 10))  # working tray for remin/denitrification
        self.Production = np.zeros(8)       # production flux (mol P/m2/yr) for each surface box
        self.Redfield_O = np.full(8, -138.0)  # O:P Redfield ratio (negative: O2 consumed)
        self.Redfield_N = np.full(8, 16.0)    # N:P Redfield ratio
        self.OxSat = np.zeros(NB)           # O2 saturation (mol/kg)
        self.lostN = np.zeros(NB)           # N lost to WC denitrification per timestep
        self.lostNsed = np.zeros(NB)        # N lost to sed denitrification per timestep
        self.lostNtotal = np.zeros(NB)      # cumulative WC denitrification
        self.lostNsedtotal = np.zeros(NB)   # cumulative sed denitrification
        self.fixedN = np.zeros(8)           # N fixed per timestep
        self.shelfDenitFract = np.zeros(8)  # fraction of export N denitrified in shelf sediments (per surface box)
        self.fixedNtotal = np.zeros(NB)     # cumulative N fixation
        # Seafloor depth levels for sed-denitrification
        self.SFdepth_levels = np.zeros((NB, 5))    # depth of each level (km)
        self.SFfractarea_levels = np.zeros((NB, 5)) # fractional area at each level
        self.Csolved = Cchem()


#::
#:: --- class ParamQ14 ---
#:: Container for the prescribed atmospheric 14C production history.
class ParamQ14:
    """Radiocarbon production parameters."""
    def __init__(self):
        self.OUT2 = None
        self.Q14Cforcing = np.zeros((367, 4))
        self.Qnode = 0.0
        self.Qnextnode = 0.0
        self.DQ = 0.0
        self.Dt = 0.0
        self.yrstep = 0.0
        self.prod = 0.0
        self.ExNo = 0
        self.row = 0
        self.init_true = 0
        self.OUT = None
        self.OUTrow = 0


#::
#:: --- class DGLforcing ---
#:: Deglacial forcing time-series container.
class DGLforcing:
    """Deglacial forcing."""
    def __init__(self, nforcing=12):
        self.Forcing = np.zeros(nforcing)
        self.ForceTime = np.zeros(nforcing, dtype=int)
        self.node = 0.0
        self.nextnode = 0.0
        self.D = 0.0
        self.Dt = 0
        self.yrstep = 0
        self.value = 0.0
        self.row = 0
        self.init_true = 0


#::
#:: --- class DGLF ---
#:: Deglacial forcing helper.
class DGLF:
    """All deglacial forcings."""
    def __init__(self):
        self.F1 = DGLforcing(16)
        self.F2 = DGLforcing(12)
        self.F3 = DGLforcing(12)
        self.F4 = DGLforcing(12)
        self.init_true = 0
        self.trigerID = 0


#::
#:: --- class OceanS ---
#:: THE OCEAN object -- bundles the tracer arrays, box geometry, rain, circulation matrices,
#:: carbonate tables and all seaweed/intervention target arrays. Passed everywhere as 'ocean'.
class OceanS:
    """Complete ocean state."""
    def __init__(self):
        self.OUTSW = None
        self.OUTSWrow = 0
        self.ArtifUpwellVdot = np.zeros(8)
        self.circulationM = np.zeros((NB, NB))
        self.circM_Sv = np.zeros((NB, NB))
        self.circM_ArtifUpwell = np.zeros((NB, NB))
        self.RainOrg = np.zeros((10, 8))
        self.RainCC = np.zeros((10, 8))
        self.RainSi = np.zeros((10, 8))
        self.RainSW = np.zeros((10, 8))
        self.RainSurfSW = np.zeros((10, 8))
        self.RainArtifUpwellSW = np.zeros((10, 8))
        self.RainSAZSW = np.zeros((10, 8))
        self.stoichSW = np.zeros(2)
        self.stoichSurfSW = np.zeros(2)
        self.stoichArtifUpwellSW = np.zeros(2)
        self.stoichSAZSW = np.zeros(2)
        self.tracer = TracerList()
        self.rain = RainList()
        self.rainSW = RainList()
        self.rainSurfSW = RainList()
        self.rainArtifUpwellSW = RainList()
        self.rainSAZSW = RainList()
        self.box = BoxList()
        self.Ksurf = Ktable()
        self.Ksurf2x = Ktable()
        self.Kdeep = Ktable()
        self.SF = Seafloor()
        self.venttracer = VentTracerList()
        self.Schemes = type('VolChangeSchemes', (), {
            'MDtoDO': np.zeros((NB, NB)),
            'DOtoMD': np.zeros((NB, NB)),
            'MDtoDOviaNA': np.zeros((NB, NB)),
            'DOtoMDviaAApLL': np.zeros((NB, NB)),
            'UOtoDO': np.zeros((NB, NB)),
            'DOtoUO': np.zeros((NB, NB)),
            'NO': np.zeros((NB, NB)),
        })()
        self.VolParams = type('VolChange', (), {
            'scale': 1.0,
            'newvol': np.zeros(NB),
            'reducedvol': np.zeros(NB),
            'DoIt': 0,
            'newtracer': None,
            'newventtracer': None,
            'VolOp': np.zeros((NB, NB)),
        })()


#::
#:: --- class AtmosphereS ---
#:: THE ATMOSPHERE object -- CO2 (ppm), its 13C/14C, and gas-exchange bookkeeping.
class AtmosphereS:
    """Atmosphere state."""
    def __init__(self):
        self.ppm = 300.0
        self.dn13 = 0.0
        self.dn14 = 0.0
        self.oldH = np.full(8, 1e-2)
        self.setCO2pulse = 0.0


#::
#:: --- class GeosphereS ---
#:: THE GEOSPHERE object -- solid-earth/sediment reservoirs and weathering/volcanic terms.
class GeosphereS:
    """Geosphere state."""
    def __init__(self):
        self.ppmCorg = 0.0
        self.flux = 0.0
        self.dn14 = 0.0
        self.dn13 = 0.0
        self.d14Corg = 0.0
        self.NotUsed = False


#::
#:: --- class ParametersS ---
#:: ALL model PARAMETERS and switches: isotope fractionation factors, denitrification
#:: parameters, the OMZ pocket parameters (InternalFlux, OxygenVolumeParam, maxInternalTurnover...),
#:: Martin curve, flags (Ncycle, watercolumndenitrification, sedimentdenitrification...), and the year counter.
class ParametersS:
    """Model parameters."""
    def __init__(self):
        self.OpenSystem = 0
        self.ExOUT = np.zeros((401, 2))
        self.Ex1 = 0
        self.Ex2 = 0
        self.Exflag = 0
        self.TxS = 0.0
        self.VolcX = 0.0
        self.WeathX = 0.0
        self.CaX = 0.0
        self.RivX = 0.0
        self.SetCO2 = 0.0
        self.ORGe = 0.0
        self.flag = 0
        self.Spike = 0.0
        self.SpikeDelta = 0.0
        self.DissolveX = 0.0
        self.alphaSi = 0.0
        self.scalelength = 0.0
        self.PAZiceX = 0.0
        self.PAZsv = 0.0
        self.PAZarea = 0.0
        self.MixFrac = 0.0
        self.year = 0
        self.C14X = 0.0
        self.Q14 = ParamQ14()
        self.DGLFall = DGLF()
        # N-cycle parameters
        self.Ncycle = False
        self.sedimentdenitrification = True
        self.watercolumndenitrification = True
        # --- mechanistic N-flux rebuild (ncycle_nflux_v2); default OFF = legacy path ---
        self.NfluxV2 = False
        self.alphaN15org = 0.995
        self.alphaNO18org = 0.995
        self.fixdelta = -1.0
        self.O2innewNitrate = 0.0
        self.alphaO218 = 0.982
        self.alphadntrN15 = 0.975
        self.alphadntrNO18 = 0.975
        self.denitriparam = 94.4 / 106.0
        self.denitriparam2 = 16.0 / 106.0
        self.scalingShallow = 0.8
        self.scalingDeep = 1.5
        self.MartinB = -0.6767
        self.orgfluxscale = 1.1
        self.fluxangle = 5.0
        # InternalFlux: OMZ<->oxic / OMZ<->ANOX sub-volume exchange coefficient.
        # Pascal's 0.00008 was tuned for the original (small) thermocline boxes.
        # On the multi-layer geometry (NLAYER>1) the larger appended boxes make the
        # raw internal flux exceed the omzV cap (Flux=min(Flux,omzV) saturates), which
        # fully re-mixes the anoxic core with box-mean water every step and pins
        # CompANOX_O2 >= CompOMZ_O2 (k4==1 -> no water-column denitrification).
        self.InternalFlux = 0.00008  # Pascal's geometric exchange coefficient (unchanged)
        # maxInternalTurnover: cap on the suboxic<->surrounding exchange as a fraction of
        # the suboxic sub-volume per year. 1.0 reproduces Pascal's original (full-exchange)
        # cap and is correct for the small original boxes (NLAYER=1). For the multi-layer
        # geometry the larger boxes need a sub-unity turnover so the anoxic core can form;
        # 0.30 matches observed nitrate (~30) and is geometry-independent (same value works
        # across NLAYER). See docs/v26_pascal_recal_FINDINGS.md.
        self.maxInternalTurnover = 1.0 if NLAYER == 1 else 0.30
        self.ResidualOxygen = 10.0
        self.OxygenVolumeParam = 0.02 / 160.0
        self.depthdistribut = 0.3
        self.antarctic = 0.01


#::
#:: --- class Experiment ---
#:: Simple bundle of (ocean, atmosphere, geosphere, param) returned by build().
class Experiment:
    """Complete experiment state."""
    def __init__(self):
        self.ocean = OceanS()
        self.atmosphere = AtmosphereS()
        self.geosphere = GeosphereS()
        self.param = ParametersS()


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

#::
#:: ############################################################################################
#:: ## MODEL SETUP & INPUT -- locate data files, read the base 18-box geometry,
#:: ## circulation, temperatures/salinities and production targets, and build a model instance.
#:: ############################################################################################
#::
#:: --- def get_base_path ---
#:: Return the directory that holds the model's input data files.
def get_base_path():
    """Get the base path for data files (the CYCLOPS-CY2SW_C++ directory)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CYCLOPS-CY2SW_C++')


#::
#:: --- def open_system ---
#:: Toggle an 'open-system' parameter (weathering/burial closure choice).
def open_system(param, value):
    """Set open/closed system."""
    param.OpenSystem = value
    if value == 0:
        param.RivX = 0.0
    else:
        param.RivX = 1.0


#::
#:: --- def load_dgl_forcing ---
#:: Load a deglacial forcing time-series from file.
def load_dgl_forcing(F):
    """Load deglacial forcing files."""
    base = get_base_path()

    data = np.loadtxt(os.path.join(base, 'FORCING', 'EPSL2014_FORCE', 'FORCE_DGL_NAcirc.txt'))
    for row in range(16):
        F.F1.ForceTime[row] = int(data[row, 0])
        F.F1.Forcing[row] = data[row, 1]

    data = np.loadtxt(os.path.join(base, 'FORCING', 'EPSL2014_FORCE', 'FORCE_DGL_SOc.txt'))
    for row in range(12):
        F.F2.ForceTime[row] = int(data[row, 0])
        F.F2.Forcing[row] = data[row, 1]
        F.F3.ForceTime[row] = int(data[row, 2])
        F.F3.Forcing[row] = data[row, 3]

    data = np.loadtxt(os.path.join(base, 'FORCING', 'EPSL2014_FORCE', 'FORCE_HOLO_Vol.txt'))
    for row in range(12):
        F.F4.ForceTime[row] = int(data[row, 0])
        F.F4.Forcing[row] = data[row, 1]


#::
#:: --- def input_model ---
#:: Read the base 18-box input data (geometry, circulation, T/S, box depths, production
#:: targets, seafloor hypsometry) and populate the ocean/atmosphere/geosphere objects.
def input_model(ocean, atmosphere, geosphere):
    """Initialize model from input file (CYCLOPSpp_INPUTunicode_silica.txt)."""
    base = get_base_path()

    ocean.tracer.Sal[:] = 34.7
    ocean.tracer.Temp[:] = 5.0
    ocean.tracer.Si[:] = 92.0
    ocean.tracer.dc30[:] = 0.0

    print(" Initializing model CYCLOPS++ V1 (Python) ...")

    # Read the input file - whitespace separated values
    filepath = os.path.join(base, 'GITCY', 'CYCLOPSpp_INPUTunicode_silica.txt')
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse all numeric values from the file
    tokens = content.replace('\t', ' ').replace('\n', ' ').split()
    vals = []
    for t in tokens:
        t = t.strip().strip('"').strip("'")
        if t == '' or t.startswith('{') or t.startswith('KGPERM') or t.startswith('MOLES') or \
           t.startswith('EXCHANGE') or t.startswith('Rpdb') or t.startswith('Rair') or \
           t.startswith('SECPER') or t.startswith('C14HALF') or t.startswith('EARTH') or \
           t.startswith('NBSC14') or t.startswith('AVOGADRO') or t.startswith('TOTAL') or \
           t.startswith('KDISS') or t.startswith('CAINPUT') or t.startswith('Rnc') or \
           t.startswith('SEDMASS') or t.startswith('IDMIXING') or t.startswith('PAZOVERTURN') or \
           t.startswith('=') or t.startswith(';'):
            continue
        try:
            vals.append(float(t))
        except ValueError:
            continue

    idx = 0

    # Circulation matrix 18x18
    for row in range(18):
        for col in range(18):
            ocean.circulationM[row, col] = vals[idx]; idx += 1

    # Store copy in Sv
    ocean.circM_Sv[:] = ocean.circulationM[:]

    # RainOrg 10x8
    for row in range(10):
        for col in range(8):
            ocean.RainOrg[row, col] = vals[idx]; idx += 1

    # RainCC 10x8
    for row in range(10):
        for col in range(8):
            ocean.RainCC[row, col] = vals[idx]; idx += 1

    ocean.RainSi = ocean.RainCC.copy()

    # Volume 18
    for col in range(18):
        ocean.box.vol[col] = vals[idx]; idx += 1
    ocean.box.vol = ocean.box.vol * 1.35e18 * 0.01  # cubic meters
    ocean.box.CtoN = ocean.box.vol * 1024.0
    ocean.box.NtoC = 1.0 / (ocean.box.vol * 1024.0)

    # Area 8
    for col in range(8):
        ocean.box.Area[col] = vals[idx]; idx += 1
    ocean.box.Area = ocean.box.Area / 100.0 * 3.265e14

    # Phosphate 18
    for col in range(18):
        ocean.tracer.P[col] = vals[idx]; idx += 1
        ocean.tracer.N[col] = 16.0 * ocean.tracer.P[col]

    # setP 8
    for col in range(8):
        ocean.box.setP[col] = vals[idx]; idx += 1

    # setSi 8
    for col in range(8):
        ocean.box.setSi[col] = vals[idx]; idx += 1

    # CaRatio 8
    for col in range(8):
        ocean.box.CaRatio[col] = vals[idx]; idx += 1

    # Salinity 8 (surface boxes)
    for col in range(8):
        ocean.tracer.Sal[col] = vals[idx]; idx += 1

    # Temperature 8 (surface boxes)
    for col in range(8):
        ocean.tracer.Temp[col] = vals[idx]; idx += 1

    # DIC 18
    for col in range(18):
        ocean.tracer.C[col] = vals[idx]; idx += 1

    # Alkalinity 18
    for col in range(18):
        ocean.tracer.Alk[col] = vals[idx]; idx += 1

    # Seafloor data: 18 rows of 13 columns
    for level in range(18):
        ocean.SF.SFdepth[level] = vals[idx]; idx += 1
        ocean.SF.NCW.FArea[level] = vals[idx]; idx += 1
        ocean.SF.NCW.FCa[level] = vals[idx]; idx += 1
        ocean.SF.DSO.FArea[level] = vals[idx]; idx += 1
        ocean.SF.DSO.FCa[level] = vals[idx]; idx += 1
        ocean.SF.Atl.FArea[level] = vals[idx]; idx += 1
        ocean.SF.Atl.FCa[level] = vals[idx]; idx += 1
        ocean.SF.Ind.FArea[level] = vals[idx]; idx += 1
        ocean.SF.Ind.FCa[level] = vals[idx]; idx += 1
        ocean.SF.SPac.FArea[level] = vals[idx]; idx += 1
        ocean.SF.SPac.FCa[level] = vals[idx]; idx += 1
        ocean.SF.NPac.FArea[level] = vals[idx]; idx += 1
        ocean.SF.NPac.FCa[level] = vals[idx]; idx += 1

        ocean.SF.NCW.Hsitu[level] = 1e-2
        ocean.SF.DSO.Hsitu[level] = 1e-2
        ocean.SF.Atl.Hsitu[level] = 1e-2
        ocean.SF.Ind.Hsitu[level] = 1e-2
        ocean.SF.SPac.Hsitu[level] = 1e-2
        ocean.SF.NPac.Hsitu[level] = 1e-2

    geosphere.ppmCorg = 2500 * 1e15 / 12.0 / 1.773e14
    geosphere.flux = 0.01 * geosphere.ppmCorg
    geosphere.dn14 = 0.0
    geosphere.d14Corg = 0.0
    geosphere.dn13 = -30.0 * geosphere.ppmCorg

    atmosphere.ppm = 300.0
    atmosphere.dn13 = 0.0
    atmosphere.dn14 = 1000.0 * atmosphere.ppm
    atmosphere.oldH = np.full(8, 1e-2)
    ocean.box.setProdP[:] = -99.0
    ocean.box.ORGe[:] = 20.0
    ocean.tracer.dc13[:] = 0.0
    ocean.tracer.Preg[:] = 0.0
    ocean.tracer.PregSW[:] = 0.0
    ocean.tracer.PregSurfSW[:] = 0.0
    ocean.tracer.PregArtifUpwellSW[:] = 0.0
    ocean.tracer.PregSAZSW[:] = 0.0
    ocean.tracer.Alkreg[:] = 0.0
    ocean.tracer.dc14 = ocean.tracer.C * 850.0

    ocean.venttracer.trueage[:] = 0.0
    ocean.venttracer.pref14Cage[:] = 0.0
    ocean.venttracer.vent[:] = 0.0
    for b in range(8):
        ocean.venttracer.vent[b, b] = 1.0


#::
#:: ############################################################################################
#:: ## INTERVENTION SETUP -- configure the seaweed-CDR and related forcings
#:: ## (depth-cycled seaweed, surface seaweed, artificial upwelling, SAZ seaweed, CO2 pulse,
#:: ## ocean alkalinity enhancement, iron fertilization). Each fills the per-box 'set*' targets.
#:: ############################################################################################
#::
#:: --- def init_sw ---
#:: Depth-cycled seaweed: set the per-column export target setSW and the seaweed C:P and N:P.
#:: RainSW maps each surface column's export to the interior boxes where it remineralizes (f_deep = deep fraction).
def init_sw(ocean, SWFlux, f_deep, SW_CP, SW_NP):
    """Initialize depth-cycled seaweed parameters."""
    ocean.RainSW = np.zeros((10, 8))
    for i in range(4):
        ocean.RainSW[i, i] = 1.0 - f_deep
        ocean.RainSW[i + 6, i] = f_deep
    ocean.RainSW[4, 4] = 1.0
    ocean.RainSW[5, 5] = 1.0
    ocean.RainSW[5, 6] = 1.0
    ocean.RainSW[5, 7] = 1.0

    surf_vol_sum = ocean.box.vol[0] + ocean.box.vol[1] + ocean.box.vol[2] + ocean.box.vol[3]
    SW_mCDR = (SWFlux * 1e15) * (1.0/12.0) * (1e6 / (1024.0 * SW_CP)) / surf_vol_sum

    ocean.box.setSW[:] = 0.0
    ocean.box.setSW[:4] = SW_mCDR
    ocean.stoichSW[0] = SW_CP
    ocean.stoichSW[1] = SW_NP


#::
#:: --- def init_surf_sw ---
#:: Surface-grown seaweed variant (exports from the surface boxes directly).
def init_surf_sw(ocean, SWFlux, f_deep, SW_CP, SW_NP):
    """Initialize surface-grown seaweed parameters."""
    ocean.RainSurfSW = np.zeros((10, 8))
    for i in range(4):
        ocean.RainSurfSW[i, i] = 1.0 - f_deep
        ocean.RainSurfSW[i + 6, i] = f_deep
    ocean.RainSurfSW[4, 4] = 1.0
    ocean.RainSurfSW[5, 5] = 1.0
    ocean.RainSurfSW[5, 6] = 1.0
    ocean.RainSurfSW[5, 7] = 1.0

    surf_vol_sum = ocean.box.vol[0] + ocean.box.vol[1] + ocean.box.vol[2] + ocean.box.vol[3]
    SW_mCDR = (SWFlux * 1e15) * (1.0/12.0) * (1e6 / (1024.0 * SW_CP)) / surf_vol_sum

    ocean.box.setSurfSW[:] = 0.0
    ocean.box.setSurfSW[:4] = SW_mCDR
    ocean.stoichSurfSW[0] = SW_CP
    ocean.stoichSurfSW[1] = SW_NP


#::
#:: --- def init_artif_upwell_sw ---
#:: Artificial-upwelling seaweed variant setup (interior water pumped to the surface).
def init_artif_upwell_sw(ocean, SWFlux, f_deep, SW_CP, SW_NP):
    """Initialize artificially upwelled seaweed parameters."""
    ocean.RainArtifUpwellSW = np.zeros((10, 8))
    for i in range(4):
        ocean.RainArtifUpwellSW[i, i] = 1.0 - f_deep
        ocean.RainArtifUpwellSW[i + 6, i] = f_deep
    ocean.RainArtifUpwellSW[4, 4] = 1.0
    ocean.RainArtifUpwellSW[5, 5] = 1.0
    ocean.RainArtifUpwellSW[5, 6] = 1.0
    ocean.RainArtifUpwellSW[5, 7] = 1.0

    surf_vol_sum = ocean.box.vol[0] + ocean.box.vol[1] + ocean.box.vol[2] + ocean.box.vol[3]
    SW_mCDR = (SWFlux * 1e15) * (1.0/12.0) * (1e6 / (1024.0 * SW_CP)) / surf_vol_sum

    ocean.box.setArtifUpwellSW[:] = 0.0
    ocean.box.setArtifUpwellSW[:4] = SW_mCDR
    ocean.stoichArtifUpwellSW[0] = SW_CP
    ocean.stoichArtifUpwellSW[1] = SW_NP


#::
#:: --- def init_saz_sw ---
#:: Sub-Antarctic-zone seaweed variant setup (deployment over the SAZ).
def init_saz_sw(ocean, SWFlux, f_deep, SW_CP, SW_NP):
    """Initialize SAZ surface-grown seaweed parameters."""
    ocean.RainSAZSW = np.zeros((10, 8))
    for i in range(4):
        ocean.RainSAZSW[i, i] = 1.0 - f_deep
        ocean.RainSAZSW[i + 6, i] = f_deep
    ocean.RainSAZSW[4, 4] = 1.0
    ocean.RainSAZSW[5, 5] = 1.0
    ocean.RainSAZSW[5, 6] = 1.0
    ocean.RainSAZSW[5, 7] = 1.0

    # For SAZ SW experiment - distribute to SAZ box only
    SW_mCDR = (SWFlux * 1e15) * (1.0/12.0) * (1e6 / (1024.0 * SW_CP)) / ocean.box.vol[6]
    ocean.box.setSAZSW[:] = 0.0
    ocean.box.setSAZSW[6] = SW_mCDR

    ocean.stoichSAZSW[0] = SW_CP
    ocean.stoichSAZSW[1] = SW_NP


#::
#:: --- def init_emissions ---
#:: Set an anthropogenic CO2 emission pulse (ppm/yr) on the atmosphere.
def init_emissions(atmosphere, pulseFlux):
    """Initialize CO2 emissions."""
    atmosphere.setCO2pulse = pulseFlux / 2.12
    print(f"  Emissions pulse: {atmosphere.setCO2pulse:.4f} ppm/yr")


#::
#:: --- def init_oae ---
#:: Ocean alkalinity enhancement: add alkalinity to the surface boxes.
def init_oae(ocean, OAEFlux):
    """Initialize Ocean Alkalinity Enhancement."""
    surf_vol_sum = ocean.box.vol[0] + ocean.box.vol[1] + ocean.box.vol[2] + ocean.box.vol[3]
    OAE_mCDR = (OAEFlux * 1e15) * (1.0/12.0) * (1e6/1024.0) / surf_vol_sum
    ocean.box.setOAE[:] = 0.0
    ocean.box.setOAE[:4] = OAE_mCDR


#::
#:: --- def init_iron_fert ---
#:: Iron fertilization: lower the Southern-Ocean phosphate target so more nutrient is consumed.
def init_iron_fert(ocean, ironFertFrac):
    """Initialize iron fertilization."""
    ocean.box.setIronFertP = ocean.box.setP.copy()
    ocean.box.setIronFertP[6] = ironFertFrac * ocean.box.setIronFertP[6]


#::
#:: ############################################################################################
#:: ## RADIOCARBON & DEGLACIAL FORCING setup.
#:: ############################################################################################
#::
#:: --- def load_q14c ---
#:: Load the prescribed atmospheric 14C production history.
def load_q14c(Q14):
    """Load radiocarbon production data."""
    base = get_base_path()
    if Q14.ExNo == 0:
        filepath = os.path.join(base, '14CPROD', 'QrecFILE', 'Qrec_GLOPIS.txt')
        data = np.loadtxt(filepath)
        Q14.Q14Cforcing[:, :4] = data[:367, :4]
    elif Q14.ExNo > 0:
        filepath = os.path.join(base, '14CPROD', 'QrecFILE', f'Qrec_GLOPIS_SCENARIO_{Q14.ExNo}.txt')
        data = np.loadtxt(filepath)
        Q14.Q14Cforcing[:, :3] = data[:367, :3]


#::
#:: --- def init_dgl_forcing ---
#:: Initialize deglacial (transient) forcing at a given year.
def init_dgl_forcing(F, year):
    """Initialize deglacial forcing."""
    load_dgl_forcing(F)

    F.F1.row = 15
    F.F1.node = F.F1.Forcing[F.F1.row]
    F.F1.nextnode = F.F1.Forcing[F.F1.row - 1]
    F.F1.D = F.F1.nextnode - F.F1.node
    F.F1.Dt = -F.F1.ForceTime[F.F1.row - 1] + F.F1.ForceTime[F.F1.row]
    F.F1.yrstep = F.F1.ForceTime[F.F1.row] - year
    F.F1.init_true = 1

    for attr in ['F2', 'F3']:
        f = getattr(F, attr)
        f.row = 11
        f.node = f.Forcing[f.row]
        f.nextnode = f.Forcing[f.row - 1]
        f.D = f.nextnode - f.node
        f.Dt = -f.ForceTime[f.row - 1] + f.ForceTime[f.row]
        f.yrstep = f.ForceTime[f.row] - year
        f.init_true = 1

    F.F4.row = 11
    F.F4.node = F.F4.Forcing[F.F3.row]  # note: uses F3.row in C++ (line 456)
    F.F4.nextnode = F.F4.Forcing[F.F4.row - 1]
    F.F4.D = F.F4.nextnode - F.F4.node
    F.F4.Dt = -F.F4.ForceTime[F.F4.row - 1] + F.F4.ForceTime[F.F4.row]
    F.F4.yrstep = F.F4.ForceTime[F.F4.row] - year
    F.F4.init_true = 1
    F.init_true = 1


#::
#:: --- def update_dgl_forcing ---
#:: Advance the deglacial forcing to the current year (interpolates the time-series).
def update_dgl_forcing(F, year):
    """Update deglacial forcing for current year."""
    # FORCING 1 (step change)
    if F.F1.yrstep == F.F1.Dt:
        F.F1.row -= 1
        if F.F1.row >= 1:
            F.F1.node = F.F1.nextnode
            F.F1.nextnode = F.F1.Forcing[F.F1.row - 1]
            F.F1.D = F.F1.nextnode - F.F1.node
            F.F1.Dt = -F.F1.ForceTime[F.F1.row - 1] + F.F1.ForceTime[F.F1.row]
            F.F1.yrstep = 0
        elif F.F1.row == 0:
            F.F1.node = F.F1.nextnode
            F.F1.D = 0
            F.F1.Dt = -1
            F.F1.yrstep = 0
    F.F1.value = -77
    if F.F1.yrstep == 0:
        F.F1.value = F.F1.node
    F.F1.yrstep += 1

    # FORCING 2 (linear interpolation)
    if F.F2.yrstep == F.F2.Dt:
        F.F2.row -= 1
        if F.F2.row >= 1:
            F.F2.node = F.F2.nextnode
            F.F2.nextnode = F.F2.Forcing[F.F2.row - 1]
            F.F2.D = F.F2.nextnode - F.F2.node
            F.F2.Dt = -F.F2.ForceTime[F.F2.row - 1] + F.F2.ForceTime[F.F2.row]
            F.F2.yrstep = 0
        elif F.F2.row == 0:
            F.F2.node = F.F2.nextnode
            F.F2.D = 0
            F.F2.Dt = -1
            F.F2.yrstep = 0
    Frac = F.F2.yrstep / F.F2.Dt if F.F2.Dt != 0 else 0
    F.F2.value = F.F2.node + F.F2.D * Frac
    F.F2.yrstep += 1

    # FORCING 3
    if F.F3.yrstep == F.F3.Dt:
        F.F3.row -= 1
        if F.F3.row >= 1:
            F.F3.node = F.F3.nextnode
            F.F3.nextnode = F.F3.Forcing[F.F3.row - 1]
            F.F3.D = F.F3.nextnode - F.F3.node
            F.F3.Dt = -F.F3.ForceTime[F.F3.row - 1] + F.F3.ForceTime[F.F3.row]
            F.F3.yrstep = 0
        elif F.F3.row == 0:
            F.F3.node = F.F3.nextnode
            F.F3.D = 0
            F.F3.Dt = -1
            F.F3.yrstep = 0
    Frac = F.F3.yrstep / F.F3.Dt if F.F3.Dt != 0 else 0
    F.F3.value = F.F3.node + F.F3.D * Frac
    F.F3.yrstep += 1

    # FORCING 4 (step change)
    if F.F4.yrstep == F.F4.Dt:
        F.F4.row -= 1
        if F.F4.row >= 1:
            F.F4.node = F.F4.nextnode
            F.F4.nextnode = F.F4.Forcing[F.F4.row - 1]
            F.F4.D = F.F4.nextnode - F.F4.node
            F.F4.Dt = -F.F4.ForceTime[F.F4.row - 1] + F.F4.ForceTime[F.F4.row]
            F.F4.yrstep = 0
        elif F.F4.row == 0:
            F.F4.node = F.F4.nextnode
            F.F4.D = 0
            F.F4.Dt = -1
            F.F4.yrstep = 0
    if F.F4.yrstep == 0:
        F.F4.value = F.F4.node
    F.F4.yrstep += 1


#::
#:: ############################################################################################
#:: ## CIRCULATION & VOLUME SCHEMES -- build/modify the transport matrix, set up the
#:: ## Southern-Ocean mixing, and (for the N-layer model) the sub-volume/vertical-exchange scheme.
#:: ############################################################################################
#::
#:: --- def change_circ ---
#:: Switch/scale the circulation matrix to a named configuration (e.g. weaker overturning).
def change_circ(circulationM, circID, ex5=0, ex6=0, ex7=0):
    """Load a circulation matrix."""
    base = get_base_path()
    if circID == 0:
        filepath = os.path.join(base, 'CIRCULATIONS', 'NADW_HainGBC2010_MYNADW2.txt')
    elif circID == 1:
        filepath = os.path.join(base, 'CIRCULATIONS', 'GNAIW_HainGBC2010_MYGNAIW5normIDmix.txt')
    elif circID == 98:
        filepath = os.path.join(base, 'CIRCULATIONS', 'GNAIWslow', f'GNAIWslow_{ex5}0.txt')
    elif circID == 99:
        filepath = os.path.join(base, 'CIRCULATIONS', 'GNAIWmod', f'GNAIWmod_{ex5}_{ex6}_{ex7}.txt')
    else:
        print(f"XXX CIRC-WARNING: ILLEGITIMATE CHOICE - default to NADW")
        filepath = os.path.join(base, 'CIRCULATIONS', 'NADW_HainGBC2010_MYNADW2.txt')

    data = np.loadtxt(filepath)
    circulationM[:] = data.reshape(18, 18)


#::
#:: --- def init_vol_schemes ---
#:: Set up the vertical-exchange / sub-volume schemes used by the N-layer thermocline.
def init_vol_schemes(Schemes):
    """Initialize volume change schemes."""
    MDtoDO = np.zeros((NB, NB))
    MDtoDO[14, 8] = 0.2567
    MDtoDO[15, 9] = 0.206
    MDtoDO[16, 10] = 0.2388
    MDtoDO[17, 11] = 0.2985
    Schemes.MDtoDO = MDtoDO * 1e6 * SECPERYEAR

    DOtoMD = np.zeros((NB, NB))
    DOtoMD[8, 14] = 0.2567
    DOtoMD[9, 15] = 0.206
    DOtoMD[10, 16] = 0.2388
    DOtoMD[11, 17] = 0.2985
    Schemes.DOtoMD = DOtoMD * 1e6 * SECPERYEAR

    DOtoMDviaAApLL = np.zeros((NB, NB))
    DOtoMDviaAApLL[13, 14] = 0.2567
    DOtoMDviaAApLL[13, 15] = 0.206
    DOtoMDviaAApLL[13, 16] = 0.2388 + 0.2985
    DOtoMDviaAApLL[16, 17] = 0.2985
    DOtoMDviaAApLL[5, 13] = 1.0
    DOtoMDviaAApLL[6, 5] = 1.0
    DOtoMDviaAApLL[8, 6] = 0.2567
    DOtoMDviaAApLL[9, 6] = 0.206
    DOtoMDviaAApLL[10, 6] = 0.2388 + 0.2985
    DOtoMDviaAApLL[11, 10] = 0.2985
    DOtoMDviaAApLL[8, 0] = 0.2567
    DOtoMDviaAApLL[9, 1] = 0.206
    DOtoMDviaAApLL[10, 2] = 0.2388
    DOtoMDviaAApLL[11, 3] = 0.2985
    DOtoMDviaAApLL[0, 8] = 0.2567
    DOtoMDviaAApLL[1, 9] = 0.206
    DOtoMDviaAApLL[2, 10] = 0.2388
    DOtoMDviaAApLL[3, 11] = 0.2985
    Schemes.DOtoMDviaAApLL = DOtoMDviaAApLL * 1e6 * SECPERYEAR

    MDtoDOviaNA = np.zeros((NB, NB))
    MDtoDOviaNA[8, 9] = 0.206
    MDtoDOviaNA[8, 10] = 0.2388 + 0.2985
    MDtoDOviaNA[10, 11] = 0.2985
    MDtoDOviaNA[12, 8] = 1.0
    MDtoDOviaNA[14, 12] = 1.0
    MDtoDOviaNA[15, 14] = 0.2388 + 0.2985 + 0.206
    MDtoDOviaNA[16, 15] = 0.2388 + 0.2985
    MDtoDOviaNA[17, 16] = 0.2985
    Schemes.MDtoDOviaNA = MDtoDOviaNA * 1e6 * SECPERYEAR

    UOtoDO = np.zeros((NB, NB))
    UOtoDO[14, 8] = 0.2567;  UOtoDO[8, 0] = 0.2567/2
    UOtoDO[15, 9] = 0.206;   UOtoDO[9, 1] = 0.206/2
    UOtoDO[16, 10] = 0.2388; UOtoDO[10, 2] = 0.2388/2
    UOtoDO[17, 11] = 0.2985; UOtoDO[11, 3] = 0.2985/2
    Schemes.UOtoDO = UOtoDO * 1e6 * SECPERYEAR

    DOtoUO = np.zeros((NB, NB))
    DOtoUO[8, 14] = 0.2567;  DOtoUO[0, 8] = 0.2567/2
    DOtoUO[9, 15] = 0.206;   DOtoUO[1, 9] = 0.206/2
    DOtoUO[10, 16] = 0.2388; DOtoUO[2, 10] = 0.2388/2
    DOtoUO[11, 17] = 0.2985; DOtoUO[3, 11] = 0.2985/2
    Schemes.DOtoUO = DOtoUO * 1e6 * SECPERYEAR

    Schemes.NO = np.zeros((NB, NB))


#::
#:: --- def oc_av_conc ---
#:: Volume-weighted mean concentration of a tracer over a set of boxes (diagnostic).
def oc_av_conc(tracer, vol, top, N):
    """Volume-weighted average concentration."""
    seg_t = tracer[top:top+N]
    seg_v = vol[top:top+N]
    return np.sum(seg_t * seg_v) / np.sum(seg_v)


#::
#:: --- def oc_av_delta ---
#:: Volume-weighted mean delta-value over a set of boxes (diagnostic).
def oc_av_delta(delta, tracer, vol, top, N):
    """Volume-weighted average delta (for isotopes)."""
    seg_d = delta[top:top+N]
    seg_t = tracer[top:top+N]
    seg_v = vol[top:top+N]
    return np.sum(seg_d * seg_t * seg_v) / np.sum(seg_t * seg_v)


#::
#:: --- def oc_av_dc ---
#:: Volume-weighted mean of a carbon delta over a set of boxes (diagnostic).
def oc_av_dc(delta, tracer, vol, top, N):
    """Volume-weighted average dc/C."""
    seg_d = delta[top:top+N]
    seg_t = tracer[top:top+N]
    seg_v = vol[top:top+N]
    return np.sum(seg_d * seg_v) / np.sum(seg_t * seg_v)


#::
#:: --- def change_vol ---
#:: Recompute box volumes and the mass<->concentration factors when the sub-volume scheme changes.
def change_vol(SchemeUsed, CircOp, tracer, venttracer, vol, vol_inv, CtoN, NtoC, params):
    """Apply volume change scheme."""
    params.reducedvol = vol.copy()
    for fr in range(NB):
        for to in range(NB):
            params.reducedvol[fr] -= params.scale * SchemeUsed[to, fr]

    params.VolOp = params.scale * SchemeUsed.copy()
    np.fill_diagonal(params.VolOp, params.reducedvol)
    params.newvol = params.VolOp @ np.ones(NB)
    for row in range(NB):
        params.VolOp[row, :] = params.VolOp[row, :] / params.newvol[row]

    for row in range(NB):
        CircOp[row, :] = CircOp[row, :] * vol[row]
    CircOp[np.diag_indices(18)] += (params.newvol - vol)
    for row in range(NB):
        CircOp[row, :] = CircOp[row, :] / params.newvol[row]

    vol[:] = params.newvol
    vol_inv[:] = 1.0 / vol
    CtoN[:] = vol * 1024
    NtoC[:] = 1.0 / (vol * 1024)

    tracer.P = params.VolOp @ tracer.P
    tracer.Preg = params.VolOp @ tracer.Preg
    tracer.PregSW = params.VolOp @ tracer.PregSW
    tracer.PregSurfSW = params.VolOp @ tracer.PregSurfSW
    tracer.PregArtifUpwellSW = params.VolOp @ tracer.PregArtifUpwellSW
    tracer.C = params.VolOp @ tracer.C
    tracer.dc13 = params.VolOp @ tracer.dc13
    tracer.Alk = params.VolOp @ tracer.Alk
    tracer.Alkreg = params.VolOp @ tracer.Alkreg
    tracer.N = params.VolOp @ tracer.N
    tracer.N15 = params.VolOp @ tracer.N15
    tracer.NO18 = params.VolOp @ tracer.NO18
    tracer.O2 = params.VolOp @ tracer.O2
    tracer.O218 = params.VolOp @ tracer.O218
    tracer.Sal = params.VolOp @ tracer.Sal
    tracer.Temp = params.VolOp @ tracer.Temp
    tracer.Si = params.VolOp @ tracer.Si
    tracer.dc30 = params.VolOp @ tracer.dc30
    tracer.dc14 = params.VolOp @ tracer.dc14

    venttracer.vent[8:, :] = params.VolOp[8:, :] @ venttracer.vent
    venttracer.trueage[8:] = params.VolOp[8:, :] @ venttracer.trueage
    venttracer.pref14Cage[8:] = params.VolOp[8:, :] @ venttracer.pref14Cage


#::
#:: --- def init_circ ---
#:: Normalize/initialize the circulation operator from the Sv transport matrix and box volumes.
def init_circ(circulationM, vol, vol_inv):
    """Convert circulation from Sv to 1/yr operator."""
    circulationM[:] = circulationM * 1e6 * SECPERYEAR  # m^3/yr

    selfval = vol.copy()
    temp = circulationM.copy()
    for row in range(NB):
        selfval[row] -= np.sum(temp[row, :])
    for row in range(NB):
        temp[row, row] = selfval[row]

    for row in range(NB):
        circulationM[row, :] = temp[row, :] / vol[row]

    vol_inv[:] = 1.0 / vol


#::
#:: --- def paz_mix ---
#:: Add polar-Antarctic-zone vertical mixing to the circulation.
def paz_mix(circ, vol, SVmix):
    """Modify PAZ mixing."""
    SVmix = SVmix * 1e6 * SECPERYEAR
    circ[7, 7] += circ[7, 13] - SVmix / vol[7]
    circ[13, 13] += circ[13, 7] - SVmix / vol[13]
    circ[7, 13] = SVmix / vol[7]
    circ[13, 7] = SVmix / vol[13]


#::
#:: --- def pfz_mix ---
#:: Add polar-frontal-zone mixing to the circulation.
def pfz_mix(circ, vol, SVmix):
    """Modify PFZ mixing."""
    SVmix = SVmix * 1e6 * SECPERYEAR
    circ[8, 8] += circ[8, 13] - SVmix / vol[8]
    circ[9, 9] += circ[9, 13] - SVmix / vol[9]
    circ[10, 10] += circ[10, 13] - SVmix / vol[10]
    circ[13, 13] += circ[13, 8] + circ[13, 9] + circ[13, 10] - 3 * SVmix / vol[13]
    circ[8, 13] = SVmix / vol[8]
    circ[13, 8] = SVmix / vol[13]
    circ[9, 13] = SVmix / vol[9]
    circ[13, 9] = SVmix / vol[13]
    circ[10, 13] = SVmix / vol[10]
    circ[13, 10] = SVmix / vol[13]


#::
#:: --- def oaa_mix ---
#:: Add open-Antarctic mixing to the circulation.
def oaa_mix(circ, vol, SVmix):
    """Modify open Antarctic mixing."""
    SVmix = SVmix * 1e6 * SECPERYEAR
    circ[5, 5] += circ[5, 13] - SVmix / vol[5]
    circ[13, 13] += circ[13, 5] - SVmix / vol[13]
    circ[5, 13] = SVmix / vol[5]
    circ[13, 5] = SVmix / vol[13]


#::
#:: --- def D14Ccalc ---
#:: Compute D14C (age-corrected radiocarbon) from C, d13C, d14C.
def D14Ccalc(C, dc13, dc14):
    """Calculate Delta14C from raw tracers."""
    return 1000.0 * ((1.0 + (dc14/C - 1000.0) / 1000.0) * 0.950625 / (1.0 + (dc13/C / 1000.0))**2 - 1.0)


#::
#:: ############################################################################################
#:: ## PHYSICS: ADVECTION -- move every tracer between boxes along the fixed
#:: ## transport matrix once per year. This is the ventilation step that brings deep water up
#:: ## and carries surface water down. OMZ-aware: advects the whole-box mean so sub-grid pockets
#:: ## never break mass conservation.
#:: ############################################################################################
#::
#:: --- def circ_advect ---
#:: Advect all tracers one year along the transport matrix. With the N-cycle active it works
#:: on the whole-box mean (oxic + OMZ + anoxic sub-volumes) so the sub-grid pockets stay mass-conserving.
def circ_advect(circ, tracer, Ncycle=False, box=None):
    """Advect tracers with circulation matrix.

    When box is provided and Ncycle=True, ComponentTray tracers (C, O2, P, Alk,
    dc13, dc14, N, N15, NO18, O218) are advected using whole-box average
    concentrations to ensure mass conservation.  The circulation matrix was built
    with completeVolume via init_circ, so when OMZ sub-volumes reduce box.vol
    below completeVolume, advecting the oxic concentration alone breaks mass
    conservation.  The fix: (1) compute whole-box average = (oxic*vol + OMZ*omzV
    + ANOX*anoxV) / completeVolume, (2) advect that average, (3) distribute the
    advection-induced delta equally to all sub-volumes.
    """
    _omz_active = (box is not None and Ncycle and np.any(box.dV > 0))

    if _omz_active:
        # --- OMZ-aware advection for ComponentTray tracers ---
        # Map: (tracer_attr_name, ComponentTray_index)
        _tray_names = [
            ('C', _iC), ('P', _iP), ('Alk', _iAlk),
            ('dc13', _iC13), ('dc14', _iC14),
            ('N', _iN),
        ]
        if Ncycle:
            _tray_names += [
                ('N15', _iN15), ('NO18', _iNO18),
                ('O2', _iO), ('O218', _iO218),
            ]

        for attr, idx in _tray_names:
            arr = getattr(tracer, attr)          # reference to numpy array
            oxic_old = arr.copy()
            wb_old = arr.copy()

            for i in range(NB):
                if box.dV[i] > 0:
                    mass = (arr[i] * box.vol[i]
                            + box.CompOMZ[i, idx] * box.omzV[i]
                            + box.CompANOX[i, idx] * box.anoxV[i])
                    wb_old[i] = mass / box.completeVolume[i]

            wb_new = circ @ wb_old

            for i in range(NB):
                if box.dV[i] > 0:
                    delta = wb_new[i] - wb_old[i]
                    arr[i] = oxic_old[i] + delta
                    box.CompOMZ[i, idx] += delta
                    box.CompANOX[i, idx] += delta
                else:
                    arr[i] = wb_new[i]

        # Non-ComponentTray tracers: represent whole-box concentrations,
        # so circ (built with completeVolume) is already mass-conserving.
        tracer.Si = circ @ tracer.Si
        tracer.dc30 = circ @ tracer.dc30
    else:
        # Standard advection (no OMZ sub-volumes)
        tracer.P = circ @ tracer.P
        tracer.C = circ @ tracer.C
        tracer.dc13 = circ @ tracer.dc13
        tracer.Alk = circ @ tracer.Alk
        tracer.N = circ @ tracer.N
        tracer.Si = circ @ tracer.Si
        tracer.dc30 = circ @ tracer.dc30
        tracer.dc14 = circ @ tracer.dc14
        if Ncycle:
            tracer.N15 = circ @ tracer.N15
            tracer.NO18 = circ @ tracer.NO18
            tracer.O2 = circ @ tracer.O2
            tracer.O218 = circ @ tracer.O218

    # Surface T/S are prescribed; only deep boxes advected
    tracer.Temp[8:] = circ[8:, :] @ tracer.Temp
    tracer.Sal[8:] = circ[8:, :] @ tracer.Sal
    tracer.H2Od18O[8:] = circ[8:, :] @ tracer.H2Od18O
    tracer.Preg[8:] = circ[8:, :] @ tracer.Preg
    tracer.PregSW[8:] = circ[8:, :] @ tracer.PregSW
    tracer.PregSurfSW[8:] = circ[8:, :] @ tracer.PregSurfSW
    tracer.PregArtifUpwellSW[8:] = circ[8:, :] @ tracer.PregArtifUpwellSW
    tracer.PregSAZSW[8:] = circ[8:, :] @ tracer.PregSAZSW
    tracer.Alkreg[8:] = circ[8:, :] @ tracer.Alkreg


#::
#:: --- def vent_track ---
#:: Update the idealized ventilation/age tracers.
def vent_track(circ, venttracer, atm, C8, dc13_8, dc14_8):
    """Track ventilation."""
    venttracer.vent[8:, :] = circ[8:, :] @ venttracer.vent
    venttracer.trueage[8:] = circ[8:, :] @ venttracer.trueage
    venttracer.trueage[8:] += 1.0

    atmD14C = D14Ccalc(atm.ppm, atm.dn13, atm.dn14)
    # Calculate preformed 14C age for surface boxes
    C = dc14_8 / C8
    d13 = dc13_8 / C8
    D14C_surf = 1000.0 * ((1.0 + (C - 1000.0) / 1000.0) * 0.950625 / (1.0 + d13 / 1000.0)**2 - 1.0)
    venttracer.pref14Cage[:8] = -5730.0 * 0.69314718056 * np.log((1000.0 + D14C_surf) / (1000.0 + atmD14C))
    venttracer.pref14Cage[8:] = circ[8:, :] @ venttracer.pref14Cage


#::
#:: ############################################################################################
#:: ## BIOLOGY (carbon core) -- surface production strips phosphate to its target and
#:: ## exports organic + CaCO3 rain; remineralization returns it at depth. Byte-identical to the
#:: ## validated 18-box carbon model apart from the box-count generalization.
#:: ############################################################################################
#::
#:: --- def prod ---
#:: Surface production: restore phosphate toward setP; the removed P becomes sinking organic + CaCO3 rain.
#:: Also applies the seaweed/OAE/iron-fertilization interventions when their flags are on. (Carbon-cycle side;
#:: the nitrogen strip and N2 fixation are done next by ncycle_production.)
def prod(tracer, setP, setProdP, setSi, setSW, setSurfSW, setOAE, setIronFertP,
         setArtifUpwellSW, setSAZSW, CaRatio, rain, rainSW, stoichSW,
         rainSurfSW, stoichSurfSW, rainArtifUpwellSW, stoichArtifUpwellSW,
         rainSAZSW, stoichSAZSW, alphaSi, CtoN, NtoC, ORGe,
         SWflag, SurfSWflag, OAEflag, IronFertflag, ArtifUpwellflag, SAZSWflag,
         Ncycle=False):
    """Calculate biological production, surface nutrient uptake, and seaweed interventions."""

    rain.d13Ccc = tracer.dc13[:8] / tracer.C[:8]
    rain.d13Corg = rain.d13Ccc + ORGe
    rain.d14Ccc = tracer.dc14[:8] / tracer.C[:8]
    rain.d14Corg = rain.d14Ccc + 2.0 * ORGe
    rainSW.d13Corg = rain.d13Corg.copy()
    rainSW.d14Corg = rain.d14Corg.copy()

    # Reset SW rain
    rainSW.P[:] = 0.0
    rainSurfSW.P[:] = 0.0
    rainArtifUpwellSW.P[:] = 0.0
    rainSAZSW.P[:] = 0.0

    # Depth-cycled seaweed
    if SWflag:
        rainSW.P = setSW * CtoN[:8]
        for IntBox in INTERMEDIATE_IDX:
            tracer.P[IntBox] = (tracer.P[IntBox] * CtoN[IntBox] - rainSW.P[IntBox - 8]) * NtoC[IntBox]
            tracer.N[IntBox] = (tracer.N[IntBox] * CtoN[IntBox] - stoichSW[1] * rainSW.P[IntBox - 8]) * NtoC[IntBox]
            tracer.Alk[IntBox] = (tracer.Alk[IntBox] * CtoN[IntBox] + stoichSW[1] * rainSW.P[IntBox - 8]) * NtoC[IntBox]

        for SurfBox in range(4):
            tracer.C[SurfBox] = (tracer.C[SurfBox] * CtoN[SurfBox] - stoichSW[0] * rainSW.P[SurfBox]) * NtoC[SurfBox]
            tracer.dc13[SurfBox] = (tracer.dc13[SurfBox] * CtoN[SurfBox] - stoichSW[0] * rainSW.P[SurfBox] * rainSW.d13Corg[SurfBox]) * NtoC[SurfBox]
            tracer.dc14[SurfBox] = (tracer.dc14[SurfBox] * CtoN[SurfBox] - stoichSW[0] * rainSW.P[SurfBox] * rainSW.d14Corg[SurfBox]) * NtoC[SurfBox]

    # OAE
    if OAEflag:
        tracer.Alk[:8] += setOAE

    # Surface-grown seaweed
    if SurfSWflag:
        rainSurfSW.P = setSurfSW.copy()
        tracer.P[:8] -= rainSurfSW.P
        for SurfBox in range(8):
            tracer.C[SurfBox] -= stoichSurfSW[0] * rainSurfSW.P[SurfBox]
            tracer.Alk[SurfBox] += stoichSurfSW[1] * rainSurfSW.P[SurfBox]
            tracer.N[SurfBox] -= stoichSurfSW[1] * rainSurfSW.P[SurfBox]
            tracer.dc13[SurfBox] -= stoichSurfSW[0] * rainSurfSW.P[SurfBox] * rainSurfSW.d13Corg[SurfBox]
            tracer.dc14[SurfBox] -= stoichSurfSW[0] * rainSurfSW.P[SurfBox] * rainSurfSW.d14Corg[SurfBox]
        rainSurfSW.P = rainSurfSW.P * CtoN[:8]

    # Artificial upwelling
    if ArtifUpwellflag:
        rainArtifUpwellSW.P = setArtifUpwellSW.copy()
        tracer.P[:8] -= rainArtifUpwellSW.P
        for SurfBox in range(8):
            tracer.C[SurfBox] -= stoichArtifUpwellSW[0] * rainArtifUpwellSW.P[SurfBox]
            tracer.Alk[SurfBox] += stoichArtifUpwellSW[1] * rainArtifUpwellSW.P[SurfBox]
            tracer.N[SurfBox] -= stoichArtifUpwellSW[1] * rainArtifUpwellSW.P[SurfBox]
            tracer.dc13[SurfBox] -= stoichArtifUpwellSW[0] * rainArtifUpwellSW.P[SurfBox] * rainArtifUpwellSW.d13Corg[SurfBox]
            tracer.dc14[SurfBox] -= stoichArtifUpwellSW[0] * rainArtifUpwellSW.P[SurfBox] * rainArtifUpwellSW.d14Corg[SurfBox]
        rainArtifUpwellSW.P = rainArtifUpwellSW.P * CtoN[:8]

    # SAZ surface-grown seaweed
    if SAZSWflag:
        rainSAZSW.P = setSAZSW.copy()
        tracer.P[:8] -= rainSAZSW.P
        for SurfBox in range(8):
            tracer.C[SurfBox] -= stoichSurfSW[0] * rainSAZSW.P[SurfBox]
            tracer.Alk[SurfBox] += stoichSurfSW[1] * rainSAZSW.P[SurfBox]
            tracer.N[SurfBox] -= stoichSurfSW[1] * rainSAZSW.P[SurfBox]
            tracer.dc13[SurfBox] -= stoichSurfSW[0] * rainSAZSW.P[SurfBox] * rainSAZSW.d13Corg[SurfBox]
            tracer.dc14[SurfBox] -= stoichSurfSW[0] * rainSAZSW.P[SurfBox] * rainSAZSW.d14Corg[SurfBox]
        rainSAZSW.P = rainSAZSW.P * CtoN[:8]

    # Phytoplankton production
    for SurfBox in range(8):
        if SurfBox < 5:
            # Set concentration rule for first 5 surface boxes
            if setP[SurfBox] < tracer.P[SurfBox]:
                rain.P[SurfBox] = tracer.P[SurfBox] - setP[SurfBox]
                tracer.P[SurfBox] = setP[SurfBox]
            else:
                rain.P[SurfBox] = 0.0

        elif SurfBox == 5:  # OAZ
            if setProdP[SurfBox] > 0:
                if tracer.P[SurfBox] > setProdP[SurfBox]:
                    rain.P[SurfBox] = setProdP[SurfBox]
                    tracer.P[SurfBox] -= setProdP[SurfBox]
                else:
                    rain.P[SurfBox] = 0.0
            elif setP[SurfBox] < tracer.P[SurfBox]:
                rain.P[SurfBox] = tracer.P[SurfBox] - setP[SurfBox]
                tracer.P[SurfBox] = setP[SurfBox]
                setProdP[SurfBox] = -rain.P[SurfBox]
            else:
                rain.P[SurfBox] = 0.0

        elif SurfBox == 6:  # SAZ
            if IronFertflag:
                if setIronFertP[SurfBox] < tracer.P[SurfBox]:
                    rain.P[SurfBox] = tracer.P[SurfBox] - setIronFertP[SurfBox]
                    tracer.P[SurfBox] = setIronFertP[SurfBox]
                else:
                    rain.P[SurfBox] = 0.0
            else:
                if setProdP[SurfBox] > 0:
                    if tracer.P[SurfBox] > setProdP[SurfBox]:
                        rain.P[SurfBox] = setProdP[SurfBox]
                        tracer.P[SurfBox] -= setProdP[SurfBox]
                    else:
                        rain.P[SurfBox] = 0.0
                elif setP[SurfBox] < tracer.P[SurfBox]:
                    rain.P[SurfBox] = tracer.P[SurfBox] - setP[SurfBox]
                    tracer.P[SurfBox] = setP[SurfBox]
                    setProdP[SurfBox] = -rain.P[SurfBox]
                else:
                    rain.P[SurfBox] = 0.0

        elif SurfBox == 7:  # PAZ
            if setProdP[SurfBox] > 0:
                if tracer.P[SurfBox] > setProdP[SurfBox]:
                    rain.P[SurfBox] = setProdP[SurfBox]
                    tracer.P[SurfBox] -= setProdP[SurfBox]
                else:
                    rain.P[SurfBox] = 0.0
            elif setP[SurfBox] < tracer.P[SurfBox]:
                rain.P[SurfBox] = tracer.P[SurfBox] - setP[SurfBox]
                tracer.P[SurfBox] = setP[SurfBox]
                setProdP[SurfBox] = -rain.P[SurfBox]
            else:
                rain.P[SurfBox] = 0.0

    # Phytoplankton C/ALK/etc uptake (Redfield: C:N:P = 106:16:1)
    for SurfBox in range(8):
        tracer.C[SurfBox] -= 106.0 * rain.P[SurfBox]
        tracer.Alk[SurfBox] -= -16.0 * rain.P[SurfBox]
        if not Ncycle:
            tracer.N[SurfBox] -= 16.0 * rain.P[SurfBox]  # N stripping handled by ncycle_production when Ncycle=True
        tracer.dc13[SurfBox] -= 106.0 * rain.P[SurfBox] * rain.d13Corg[SurfBox]
        tracer.dc14[SurfBox] -= 106.0 * rain.P[SurfBox] * rain.d14Corg[SurfBox]

    # CaCO3 rain
    rain.Ca = 106.0 * rain.P * CaRatio
    for SurfBox in range(8):
        tracer.dc13[SurfBox] -= rain.Ca[SurfBox] * rain.d13Ccc[SurfBox]
        tracer.dc14[SurfBox] -= rain.Ca[SurfBox] * rain.d14Ccc[SurfBox]
        tracer.C[SurfBox] -= rain.Ca[SurfBox]
        tracer.Alk[SurfBox] -= 2.0 * rain.Ca[SurfBox]

    rain.P = rain.P * CtoN[:8]
    rain.Ca = rain.Ca * CtoN[:8]

    # Silicon uptake
    for SurfBox in range(8):
        if setSi[SurfBox] < tracer.Si[SurfBox]:
            newdelta = (tracer.dc30[SurfBox] / tracer.Si[SurfBox]) + 1000.0 * (alphaSi - 1.0) * np.log(setSi[SurfBox] / tracer.Si[SurfBox])
            if setSi[SurfBox] < 0.0001:
                newdelta = 0.0
            rain.d30Si[SurfBox] = (tracer.dc30[SurfBox] - newdelta * setSi[SurfBox]) / (tracer.Si[SurfBox] - setSi[SurfBox])
            tracer.dc30[SurfBox] = newdelta * setSi[SurfBox]
            rain.Si[SurfBox] = tracer.Si[SurfBox] - setSi[SurfBox]
            tracer.Si[SurfBox] = setSi[SurfBox]
        else:
            rain.Si[SurfBox] = 0.0
    rain.Si = rain.Si * CtoN[:8]


#::
#:: --- def remin ---
#:: Standard remineralization of the sinking rain in the interior boxes (used when the N-cycle/OMZ
#:: manager is NOT active; when it is, ncycle_manager does the interior remineralization instead).
def remin(tracer, RainOrg, RainSi, RainSW, RainSurfSW, RainArtifUpwellSW, RainSAZSW,
          rain, rainSW, stoichSW, rainSurfSW, stoichSurfSW,
          rainArtifUpwellSW, stoichArtifUpwellSW, rainSAZSW, stoichSAZSW,
          NtoC, SWflag, SurfSWflag, ArtifUpwellflag, SAZSWflag):
    """Remineralize sinking organic matter in subsurface boxes."""

    # Depth-cycled seaweed remineralization
    if SWflag:
        addSW = RainSW @ rainSW.P
        for Box in INTERIOR_IDX:
            tracer.P[Box] += addSW[Box - 8] * NtoC[Box]
            tracer.PregSW[Box] += addSW[Box - 8] * NtoC[Box]
            tracer.C[Box] += stoichSW[0] * addSW[Box - 8] * NtoC[Box]
            tracer.Alk[Box] += -stoichSW[1] * addSW[Box - 8] * NtoC[Box]
            tracer.N[Box] += stoichSW[1] * addSW[Box - 8] * NtoC[Box]

        addSW = RainSW @ (rainSW.P * rainSW.d13Corg)
        for Box in INTERIOR_IDX:
            tracer.dc13[Box] += stoichSW[0] * addSW[Box - 8] * NtoC[Box]

        addSW = RainSW @ (rainSW.P * rainSW.d14Corg)
        for Box in INTERIOR_IDX:
            tracer.dc14[Box] += stoichSW[0] * addSW[Box - 8] * NtoC[Box]

    # Surface seaweed remineralization
    if SurfSWflag:
        addSurfSW = RainSurfSW @ rainSurfSW.P
        for Box in INTERIOR_IDX:
            tracer.P[Box] += addSurfSW[Box - 8] * NtoC[Box]
            tracer.PregSurfSW[Box] += addSurfSW[Box - 8] * NtoC[Box]
            tracer.C[Box] += stoichSurfSW[0] * addSurfSW[Box - 8] * NtoC[Box]
            tracer.Alk[Box] += -stoichSurfSW[1] * addSurfSW[Box - 8] * NtoC[Box]
            tracer.N[Box] += stoichSurfSW[1] * addSurfSW[Box - 8] * NtoC[Box]
        addSurfSW = RainSurfSW @ (rainSurfSW.P * rainSurfSW.d13Corg)
        for Box in INTERIOR_IDX:
            tracer.dc13[Box] += stoichSurfSW[0] * addSurfSW[Box - 8] * NtoC[Box]
        addSurfSW = RainSurfSW @ (rainSurfSW.P * rainSurfSW.d14Corg)
        for Box in INTERIOR_IDX:
            tracer.dc14[Box] += stoichSurfSW[0] * addSurfSW[Box - 8] * NtoC[Box]

    # Artificially upwelled seaweed remineralization
    if ArtifUpwellflag:
        addAU = RainArtifUpwellSW @ rainArtifUpwellSW.P
        for Box in INTERIOR_IDX:
            tracer.P[Box] += addAU[Box - 8] * NtoC[Box]
            tracer.PregArtifUpwellSW[Box] += addAU[Box - 8] * NtoC[Box]
            tracer.C[Box] += stoichArtifUpwellSW[0] * addAU[Box - 8] * NtoC[Box]
            tracer.Alk[Box] += -stoichArtifUpwellSW[1] * addAU[Box - 8] * NtoC[Box]
            tracer.N[Box] += stoichArtifUpwellSW[1] * addAU[Box - 8] * NtoC[Box]
        addAU = RainArtifUpwellSW @ (rainArtifUpwellSW.P * rainArtifUpwellSW.d13Corg)
        for Box in INTERIOR_IDX:
            tracer.dc13[Box] += stoichArtifUpwellSW[0] * addAU[Box - 8] * NtoC[Box]
        addAU = RainArtifUpwellSW @ (rainArtifUpwellSW.P * rainArtifUpwellSW.d14Corg)
        for Box in INTERIOR_IDX:
            tracer.dc14[Box] += stoichArtifUpwellSW[0] * addAU[Box - 8] * NtoC[Box]

    # SAZ seaweed remineralization
    if SAZSWflag:
        addSAZ = RainSAZSW @ rainSAZSW.P
        for Box in INTERIOR_IDX:
            tracer.P[Box] += addSAZ[Box - 8] * NtoC[Box]
            tracer.PregSAZSW[Box] += addSAZ[Box - 8] * NtoC[Box]
            tracer.C[Box] += stoichSAZSW[0] * addSAZ[Box - 8] * NtoC[Box]
            tracer.Alk[Box] += -stoichSAZSW[1] * addSAZ[Box - 8] * NtoC[Box]
            tracer.N[Box] += stoichSAZSW[1] * addSAZ[Box - 8] * NtoC[Box]
        addSAZ = RainSAZSW @ (rainSAZSW.P * rainSAZSW.d13Corg)
        for Box in INTERIOR_IDX:
            tracer.dc13[Box] += stoichSAZSW[0] * addSAZ[Box - 8] * NtoC[Box]
        addSAZ = RainSAZSW @ (rainSAZSW.P * rainSAZSW.d14Corg)
        for Box in INTERIOR_IDX:
            tracer.dc14[Box] += stoichSAZSW[0] * addSAZ[Box - 8] * NtoC[Box]

    # Standard phytoplankton organic matter remineralization (Redfield)
    addOrg = RainOrg @ rain.P
    for Box in INTERIOR_IDX:
        tracer.P[Box] += addOrg[Box - 8] * NtoC[Box]
        tracer.Preg[Box] += addOrg[Box - 8] * NtoC[Box]
        tracer.C[Box] += 106.0 * addOrg[Box - 8] * NtoC[Box]
        tracer.Alk[Box] += -16.0 * addOrg[Box - 8] * NtoC[Box]
        tracer.N[Box] += 16.0 * addOrg[Box - 8] * NtoC[Box]

    addOrg = RainOrg @ (rain.P * rain.d13Corg)
    for Box in INTERIOR_IDX:
        tracer.dc13[Box] += 106.0 * addOrg[Box - 8] * NtoC[Box]

    addOrg = RainOrg @ (rain.P * rain.d14Corg)
    for Box in INTERIOR_IDX:
        tracer.dc14[Box] += 106.0 * addOrg[Box - 8] * NtoC[Box]

    # Silicon remineralization
    addOrg = RainSi @ rain.Si
    for Box in INTERIOR_IDX:
        tracer.Si[Box] += addOrg[Box - 8] * NtoC[Box]

    addOrg = RainSi @ (rain.Si * rain.d30Si)
    for Box in INTERIOR_IDX:
        tracer.dc30[Box] += addOrg[Box - 8] * NtoC[Box]


#::
#:: ############################################################################################
#:: ## CARBONATE CHEMISTRY, GAS EXCHANGE & EXTERNAL SOURCES -- riverine/weathering input,
#:: ## CaCO3 dissolution, the seawater carbonate solver, air-sea CO2 and O2 exchange, volcanic CO2,
#:: ## and radiocarbon production/decay.
#:: ############################################################################################
#::
#:: --- def river ---
#:: Riverine + weathering input of DIC/alkalinity/isotopes to the surface.
def river(C, dc13, dc14, A, NtoC4, Appm, dn13, dn14, WeathX, RivX, SetCO2):
    """River input of carbon and alkalinity."""
    CCflux = 1.6 * 1e19 * RivX
    SWflux = 0.2 * 1e19 * (Appm[0] / 250.0) * WeathX  # Appm is passed as list for mutability

    dn13[0] = dn13[0] / Appm[0] * (Appm[0] - (2.0 * SWflux) / 1.773e20)
    dn14[0] = dn14[0] / Appm[0] * (Appm[0] - (2.0 * SWflux) / 1.773e20)
    Appm[0] -= (2.0 * SWflux) / 1.773e20

    C[:4] += 0.25 * (CCflux + 2.0 * SWflux) * NtoC4
    A[:4] += 0.25 * 2.0 * (CCflux + SWflux) * NtoC4
    dc13[:4] += 0.25 * dn13[0] / Appm[0] * (2.0 * SWflux) * NtoC4
    dc14[:4] += 0.25 * dn14[0] / Appm[0] * (2.0 * SWflux) * NtoC4


#::
#:: --- def dissolve ---
#:: Dissolve sinking CaCO3 in the interior according to seafloor carbonate saturation, returning DIC + alkalinity.
def dissolve(tracer, RainCC, rainCa, d13Ccc, d14Ccc, NtoC, Fdiss):
    """CaCO3 dissolution on seafloor."""
    addCC = RainCC @ rainCa
    addCC[6] *= Fdiss[0]; addCC[7] *= Fdiss[1]; addCC[8] *= Fdiss[2]; addCC[9] *= Fdiss[3]
    for box in INTERIOR_IDX:
        tracer.C[box] += addCC[box - 8] * NtoC[box]
        tracer.Alk[box] += 2.0 * addCC[box - 8] * NtoC[box]
        tracer.Alkreg[box] += 2.0 * addCC[box - 8] * NtoC[box]

    addCC = RainCC @ (rainCa * d13Ccc)
    addCC[6] *= Fdiss[0]; addCC[7] *= Fdiss[1]; addCC[8] *= Fdiss[2]; addCC[9] *= Fdiss[3]
    for box in INTERIOR_IDX:
        tracer.dc13[box] += addCC[box - 8] * NtoC[box]

    addCC = RainCC @ (rainCa * d14Ccc)
    addCC[6] *= Fdiss[0]; addCC[7] *= Fdiss[1]; addCC[8] *= Fdiss[2]; addCC[9] *= Fdiss[3]
    for box in INTERIOR_IDX:
        tracer.dc14[box] += addCC[box - 8] * NtoC[box]


#::
#:: --- def kcalc ---
#:: Compute the temperature/salinity-dependent carbonate equilibrium constants (K0,K1,K2,Kb,Ks) per surface box.
def kcalc(Ksurf, TempV, Sal, top, bot, SF):
    """Calculate equilibrium constants."""
    T = TempV + 273.15
    Temp = TempV.copy()
    Tinv = 1.0 / T
    S = Sal.copy()
    Srt = np.sqrt(S)

    Ksurf.K0 = np.exp(-60.2409 + 9345.17 * Tinv + 23.3585 * np.log(T / 100.0) + S * (0.023517 - 2.3656e-4 * T + 4.7036e-7 * T * T))
    Ksurf.K1 = 1e6 * np.exp((62.008 - 3670.7 * Tinv - 9.7944 * np.log(T) + 0.0118 * S - 1.16e-4 * S * S) * np.log(10.0))
    Ksurf.K2 = 1e6 * np.exp((-4.777 - 1394.7 * Tinv + 0.0184 * S - 1.18e-4 * S * S) * np.log(10.0))
    Ksurf.Kb = 1e6 * np.exp(Tinv * (-8966.9 - 2890.53 * Srt - 77.942 * S + 1.728 * S * Srt - 0.0996 * S * S) + 148.0248 + 137.1942 * Srt + 1.62142 * S + 0.053105 * T * Srt + np.log(T) * (-24.4344 - 25.085 * Srt - 0.2474 * S))
    Ksurf.Ks = 1e12 * np.exp(-395.8293 + 6537.773 * Tinv + 71.595 * np.log(T) - 0.17959 * T + Srt * (-1.78938 + 410.64 * Tinv + 0.0065453 * T) - 0.17755 * S + 0.0094979 * S * Srt)

    # Seafloor K corrections for deep boxes
    for b, bidx in [('NCW', 12), ('DSO', 13), ('Atl', 14), ('Ind', 15), ('SPac', 16), ('NPac', 17)]:
        basin = getattr(SF, b)
        basin.K1[:18] = Ksurf.K1[bidx] * np.exp(((2420.0 - 8500.0 * Temp[bidx]) * SF.SFdepth) / (166286.0 * T[bidx]))
        basin.K2[:18] = Ksurf.K2[bidx] * np.exp(((1640.0 - 4000.0 * Temp[bidx]) * SF.SFdepth) / (166286.0 * T[bidx]))
        basin.Kb[:18] = Ksurf.Kb[bidx] * np.exp(((2750.0 - 9500.0 * Temp[bidx]) * SF.SFdepth) / (166286.0 * T[bidx]))

    dV = -65.28 - 0.397 * Temp - 5.155e-3 * Temp**2 + (19.816 - 4.41e-2 * Temp - 1.7e-4 * Temp**2) * np.sqrt(S / 35.0)
    dk = 1.847e-2 + 1.956e-4 * Temp - 2.212e-6 * Temp**2 + (-3.217e-2 - 7.11e-5 * Temp + 2.212e-6 * Temp**2) * np.sqrt(S / 35.0)

    for b, bidx in [('NCW', 12), ('DSO', 13), ('Atl', 14), ('Ind', 15), ('SPac', 16), ('NPac', 17)]:
        basin = getattr(SF, b)
        P = 1.0 + SF.SFdepth * 100.0
        basin.Ks[:18] = Ksurf.Ks[bidx] * np.exp(-dV[bidx] * 1.202747e-2 * Tinv[bidx] * P + 0.5 * dk[bidx] * 1.202747e-2 * Tinv[bidx] * P * P)


#::
#:: --- def sf_ph_calc_kernel ---
#:: Inner kernel of the seafloor pH/saturation calculation.
def sf_ph_calc_kernel(Sal, C, A, Ca, K1, K2, Kb, Ks, Hs, CO3situ, omega):
    """Iterative pH solver for seafloor."""
    Hx = Hs * 1.01
    K1inv = 1.0 / K1

    tmpA1 = C / (Hs * K1inv + 1.0 + K2 / Hs) + 2.0 * K2 * C / (Hs * Hs * K1inv + Hs + K2) + Kb * (12.12255 * Sal) / (Hs + Kb) - Hs + 1e-2 / Hs
    tmpA2 = C / (Hx * K1inv + 1.0 + K2 / Hx) + 2.0 * K2 * C / (Hx * Hx * K1inv + Hx + K2) + Kb * (12.12255 * Sal) / (Hx + Kb) - Hx + 1e-2 / Hx

    Hs[:] = Hs * np.exp(0.01 * (tmpA1 - A) / (tmpA1 - tmpA2))
    CO3situ[:] = K2 * C / (Hs * Hs * K1inv + Hs + K2)
    omega[:] = Ca * CO3situ / Ks


#::
#:: --- def my_sqrt ---
#:: Guarded square root (avoids NaN on tiny negatives in the pH solver).
def my_sqrt(X):
    """Fast iterative square root approximation."""
    Y = (X + 1.0) * 0.5
    Y = (Y + X / Y) * 0.5
    Y = (Y + X / Y) * 0.5
    Y = (Y + X / Y) * 0.5
    return Y


#::
#:: --- def sf_ph_calc ---
#:: Compute seafloor carbonate saturation and the CaCO3 dissolution fraction (run every ~50 yr).
def sf_ph_calc(S, C, A, SF, CaX, DissolveX):
    """Calculate seafloor pH and CaCO3 dissolution."""
    Ca = CaX * 10600.0

    for b in ['Atl', 'Ind', 'SPac', 'NPac']:
        basin = getattr(SF, b)
        sf_ph_calc_kernel(S[{'Atl': 14, 'Ind': 15, 'SPac': 16, 'NPac': 17}[b]], C[{'Atl': 14, 'Ind': 15, 'SPac': 16, 'NPac': 17}[b]], A[{'Atl': 14, 'Ind': 15, 'SPac': 16, 'NPac': 17}[b]], Ca, basin.K1[:18], basin.K2[:18], basin.Kb[:18], basin.Ks[:18], basin.Hsitu[:18], basin.CO3situ[:18], basin.omega[:18])

    # Calculate CSH for each basin
    for idx, b in enumerate(['Atl', 'Ind', 'SPac', 'NPac']):
        basin = getattr(SF, b)
        top = -1
        for ly in range(18):
            if basin.omega[ly] > 1.0:
                top = ly
        if top == -1:
            SF.CSH[idx] = 1.999 if idx > 0 else 1.998
        elif top == 17:
            SF.CSH[idx] = 6.501
        else:
            SF.CSH[idx] = SF.SFdepth[top + 1] + (1.0 - basin.omega[top + 1]) * (SF.SFdepth[top + 1] - SF.SFdepth[top]) / (basin.omega[top] - basin.omega[top + 1])

    # Dissolution calculation (the "true" branch in C++)
    for idx, b in enumerate(['Atl', 'Ind', 'SPac', 'NPac']):
        basin = getattr(SF, b)
        omega_capped = basin.omega[:18].copy()
        omega_capped[omega_capped > 1.0] = 1.0

        # Use cross-basin FCa for sqrt (matches C++ code)
        # Note: C++ uses Ind.FCa for Atl, SPac.FCa for Ind, NPac.FCa for SPac, Atl.FCa for NPac
        cross_basins = ['Ind', 'SPac', 'NPac', 'Atl']
        cross_basin = getattr(SF, cross_basins[idx])
        rootFCa = my_sqrt(cross_basin.FCa[:18])

        Fdiss = 300.0 * DissolveX * rootFCa * (1.0 - omega_capped)
        Fdiss = np.clip(Fdiss, 0.0, 1.0)
        SF.Fdiss[idx] = np.sum(basin.FArea[:18] * Fdiss)
        basin.FCa[:18] = 0.999 * basin.FCa[:18] + 0.001 * (0.9 * (1.0 - Fdiss) / (1.0 - 0.9 * Fdiss))


#::
#:: --- def co2_find ---
#:: Solve the surface carbonate system for [H+]/pCO2 given DIC and alkalinity.
def co2_find(C, A, Sal, K0, K1, K2, Kb, Hs, H2CO3):
    """Iterative CO2 solver for surface ocean."""
    Hx = Hs * 1.02
    K1inv = 1.0 / K1

    tmpA1 = C / (Hs * K1inv + 1.0 + K2 / Hs) + 2.0 * K2 * C / (Hs * Hs * K1inv + Hs + K2) + Kb * (12.12255 * Sal) / (Hs + Kb) - Hs + 1e-2 / Hs
    tmpA2 = C / (Hx * K1inv + 1.0 + K2 / Hx) + 2.0 * K2 * C / (Hx * Hx * K1inv + Hx + K2) + Kb * (12.12255 * Sal) / (Hx + Kb) - Hx + 1e-2 / Hx

    Hs[:] = Hs * np.exp(0.02 * (tmpA1 - A) / (tmpA1 - tmpA2))
    H2CO3[:] = C * Hs * Hs / (Hs * Hs + K1 * Hs + K1 * K2)


#::
#:: --- def final_c_solve ---
#:: Final surface carbonate solve each year to set pCO2 consistent with DIC/alkalinity.
def final_c_solve(C, A, Sal, K0, K1, K2, Kb, Ks, CaX, Csolved):
    """Full carbonate chemistry solution for all 18 boxes."""
    Ca = CaX * 10600.0
    K1inv = 1.0 / K1
    Hs = np.full(NB, 0.005)

    # Multiple Newton iterations with decreasing step sizes
    for step in [0.3, 0.1, 0.03, 0.01]:
        Hx = Hs * (1.0 + step)
        tmpA1 = C / (Hs * K1inv + 1.0 + K2 / Hs) + 2.0 * K2 * C / (Hs * Hs * K1inv + Hs + K2) + Kb * (12.12255 * Sal) / (Hs + Kb) - Hs + 1e-2 / Hs
        tmpA2 = C / (Hx * K1inv + 1.0 + K2 / Hx) + 2.0 * K2 * C / (Hx * Hx * K1inv + Hx + K2) + Kb * (12.12255 * Sal) / (Hx + Kb) - Hx + 1e-2 / Hx
        Hs = Hs * np.exp(step * (tmpA1 - A) / (tmpA1 - tmpA2))

    Csolved.H = Hs.copy()
    Csolved.H2CO3 = C * Hs * Hs / (Hs * Hs + K1 * Hs + K1 * K2)
    Csolved.pCO2 = Csolved.H2CO3 / K0
    Csolved.HCO3 = C / (Hs * K1inv + 1.0 + K2 / Hs)
    Csolved.CO3 = C / (Hs * Hs * K1inv / K2 + Hs / K2 + 1.0)
    Csolved.omega = Csolved.CO3 * Ca / Ks
    Csolved.BOH4 = Kb * (12.12255 * Sal) / (Hs + Kb)

    # Beta calculation (d(H)/d(Alk))
    Hs_save = Hs.copy()
    for step in [0.02, 0.01]:
        Hx = Hs * (1.0 + step)
        tmpA1 = C / (Hs * K1inv + 1.0 + K2 / Hs) + 2.0 * K2 * C / (Hs * Hs * K1inv + Hs + K2) + Kb * (12.12255 * Sal) / (Hs + Kb) - Hs + 1e-2 / Hs
        tmpA2 = C / (Hx * K1inv + 1.0 + K2 / Hx) + 2.0 * K2 * C / (Hx * Hx * K1inv + Hx + K2) + Kb * (12.12255 * Sal) / (Hx + Kb) - Hx + 1e-2 / Hx
        Hs = Hs * np.exp(step * (tmpA1 - (A - 1.0)) / (tmpA1 - tmpA2))

    Csolved.beta = Hs - Hs_save


#::
#:: --- def gas_ex ---
#:: Air-sea CO2 exchange: relax surface DIC toward equilibrium with atmospheric pCO2 (and carry 13C/14C).
def gas_ex(atm, S, T, ORGe, DIC, dc13, dc14, A, NtoC, Ksurf, Area):
    """Air-sea gas exchange for CO2 and isotopes."""
    H2CO3 = np.zeros(8)
    C = DIC[:8].copy()

    co2_find(C, A, S, Ksurf.K0[:8], Ksurf.K1[:8], Ksurf.K2[:8], Ksurf.Kb[:8], atm.oldH, H2CO3)
    co2_find(C, A, S, Ksurf.K0[:8], Ksurf.K1[:8], Ksurf.K2[:8], Ksurf.Kb[:8], atm.oldH, H2CO3)

    N = 8
    for dt in range(1, N + 1):
        co2_find(C, A, S, Ksurf.K0[:8], Ksurf.K1[:8], Ksurf.K2[:8], Ksurf.Kb[:8], atm.oldH, H2CO3)

        gasex_rate = 1536000 // (((N + 1) // 2) * N)  # C++ integer division: (N+1)/2 truncates
        AirtoSea = Ksurf.K0[:8] * atm.ppm * Area * gasex_rate * dt
        SeatoAir = H2CO3 * Area * gasex_rate * dt

        d13Fsa = (dc13[:8] / C + (0.107 * T - 10.53 - 0.875)) * SeatoAir
        d13Fas = (atm.dn13 / atm.ppm - 0.875) * AirtoSea
        d14Fsa = (dc14[:8] / C + 2.0 * (0.107 * T - 10.53 - 0.875)) * SeatoAir
        d14Fas = (atm.dn14 / atm.ppm - 2.0 * 0.875) * AirtoSea

        dc13[:8] += (d13Fas - d13Fsa) * NtoC
        atm.dn13 += (np.sum(d13Fsa) - np.sum(d13Fas)) / 1.773e20
        dc14[:8] += (d14Fas - d14Fsa) * NtoC
        atm.dn14 += (np.sum(d14Fsa) - np.sum(d14Fas)) / 1.773e20

        C += (AirtoSea - SeatoAir) * NtoC
        atm.ppm -= (np.sum(AirtoSea) - np.sum(SeatoAir)) / 1.773e20

    co2_find(C, A, S, Ksurf.K0[:8], Ksurf.K1[:8], Ksurf.K2[:8], Ksurf.Kb[:8], atm.oldH, H2CO3)
    ORGe[:] = -(25.3 - (182.0 / H2CO3 * 0.8)) + 1.0 + ((-9866.0 / (T + 273.15)) + 24.12)

    DIC[:8] = C


#::
#:: --- def volcano ---
#:: Volcanic CO2 outgassing input to the atmosphere.
def volcano(Appm_ref, dn13_ref, VolcX):
    """Volcanic CO2 input."""
    Appm_ref[0] += 0.2 * 1e19 * VolcX / 1.773e20
    dn13_ref[0] += 10.5 * 0.2 * 1e19 * VolcX / 1.773e20


#::
#:: --- def reset_atm ---
#:: Hold/reset the atmosphere to a prescribed CO2 (used in fixed-CO2 experiments).
def reset_atm(Appm_ref, dn13_ref, SetCO2):
    """Reset atmosphere to target CO2."""
    dn13_ref[0] += 1.0 * (SetCO2 - Appm_ref[0])
    Appm_ref[0] = SetCO2


#::
#:: --- def handle_14c ---
#:: Radiocarbon production (cosmogenic) and decay bookkeeping for atmosphere and ocean.
def handle_14c(atm, dc14, param, geo):
    """Handle radiocarbon production and decay."""
    param.Q14.prod = 1.704

    if param.year >= 0:
        if param.Q14.init_true < 1:
            load_q14c(param.Q14)
            param.Q14.row = 366
            param.Q14.Qnode = param.Q14.Q14Cforcing[param.Q14.row, 2]
            param.Q14.Qnextnode = param.Q14.Q14Cforcing[param.Q14.row - 1, 2]
            param.Q14.DQ = param.Q14.Qnextnode - param.Q14.Qnode
            param.Q14.Dt = -param.Q14.Q14Cforcing[param.Q14.row - 1, 0] + param.Q14.Q14Cforcing[param.Q14.row, 0]
            param.Q14.yrstep = param.Q14.Q14Cforcing[param.Q14.row, 0] - param.year
            param.Q14.init_true = 1

        if param.Q14.yrstep == param.Q14.Dt:
            param.Q14.row -= 1
            if param.Q14.row >= 1:
                param.Q14.Qnode = param.Q14.Qnextnode
                param.Q14.Qnextnode = param.Q14.Q14Cforcing[param.Q14.row - 1, 2]
                param.Q14.DQ = param.Q14.Qnextnode - param.Q14.Qnode
                param.Q14.Dt = -param.Q14.Q14Cforcing[param.Q14.row - 1, 0] + param.Q14.Q14Cforcing[param.Q14.row, 0]
                param.Q14.yrstep = 0
            elif param.Q14.row == 0:
                param.Q14.Qnode = param.Q14.Qnextnode
                param.Q14.DQ = 0
                param.Q14.Dt = -1
                param.Q14.yrstep = 0

        param.Q14.prod = param.Q14.Qnode + param.Q14.DQ * (param.Q14.yrstep / param.Q14.Dt if param.Q14.Dt != 0 else 0.0)
        param.Q14.yrstep += 1

    # C14 production
    atm.dn14 += 1000.0 * (param.C14X * param.Q14.prod * 1e10 * 510072000.0 * 31556736.0 / 6.02214179e23) / (1.773e20 * 1e-6) / (0.95 * 1.25e-12)

    # C14 decay
    decay = 0.999879039  # 0.5^(1/5730)
    atm.dn14 *= decay
    dc14[:] *= decay
    geo.dn14 *= decay
    geo.d14Corg *= decay

    # Geosphere exchange
    tmp = geo.d14Corg
    geo.d14Corg = 0.99 * geo.d14Corg + 0.01 * (atm.dn14 / atm.ppm)
    atm.dn14 = 14.1 * tmp + (atm.ppm - 14.1) * (atm.dn14 / atm.ppm)  # 3000PgC


# ============================================================================
# N-CYCLE FUNCTIONS
# ============================================================================

# Tracer indices for ComponentTray / CompOMZ / CompANOX arrays
_iC = 0; _iO = 1; _iP = 2; _iAlk = 3; _iC13 = 4; _iC14 = 5
_iN = 6; _iN15 = 7; _iNO18 = 8; _iO218 = 9


#::
#:: --- def ox_exchange ---
#:: Reset surface-box oxygen to saturation (the surface is always in contact with the air).
def ox_exchange(tracer, box):
    """Air-sea O2 exchange: reset surface O2 to saturation (Garcia & Gordon 1992)."""
    A0 = 5.80818; A1 = 3.20684; A2 = 4.11890; A3 = 4.93845
    A4 = 1.01567; A5 = 1.41575
    B0 = -7.01211e-03; B1 = -7.25958e-03; B2 = -7.93334e-03; B3 = -5.54491e-03
    C0 = -1.32412e-07

    for i in range(NB):
        T = tracer.Temp[i]
        S = tracer.Sal[i]
        Ts = np.log((298.15 - T) / (273.15 + T))
        if Ts > 0:
            V1 = (A0 + A1*Ts + A2*Ts**2 + A3*Ts**3
                  + A4*Ts**4 + A5*Ts**5
                  + S*(B0 + B1*Ts + B2*Ts**2 + B3*Ts**3) + C0*S**2)
        else:
            aTs = abs(Ts)
            # Same polynomial as above, it just negates odd powers to keep signs in line 
            V1 = (A0 - A1*aTs + A2*aTs**2 - A3*aTs**3
                  + A4*aTs**4 - A5*aTs**5
                  + S*(B0 - B1*aTs + B2*aTs**2 - B3*aTs**3) + C0*S**2)
        box.OxSat[i] = np.exp(V1)  # µmol/kg (CY2SW units)

        # Surface boxes: reset to saturation
        if box.top[i] < 1.0:
            tracer.O2[i] = box.OxSat[i]
            tracer.O218[i] = tracer.O2[i] * IsoConcPDB(24.7)



#::
#:: ############################################################################################
#:: ## NITROGEN CYCLE -- the four N mechanisms ported parameter-for-parameter from
#:: ## the Sigman-2009 Pascal model: threshold N2 fixation (in production), water-column
#:: ## denitrification, deep sediment denitrification, and shallow/shelf sediment denitrification.
#:: ############################################################################################
#::
#:: --- def ncycle_production ---
#:: Nitrogen side of production: strip nitrate at the Redfield ratio using 21 Rayleigh sub-steps
#:: (genuine fractionation that enriches residual nitrate), and where upwelled nitrate falls short of the
#:: threshold, fire N2 FIXATION (adds light 15N, delta ~ -1 per mil). Also runs the shallow/shelf sediment
#:: denitrification term (isotope-only by default).
def ncycle_production(tracer, box, param, rain):
    """Production with N isotope sub-stepping, N2 fixation, and shallow sed-denitrification.
    This modifies tracer.N, tracer.N15, tracer.NO18 during surface production
    and sets rain.d15Norg and rain.d18ONorg for use in remineralization.

    Must be called AFTER standard prod() has set rain.P (production in mol P per box-volume unit).
    rain.P at this point is in CtoN-scaled units; Production (mol P/m2/yr) = rain.P / CtoN * vol / Area.
    """
    PRODUCTIONTIMESTEPS = 21

    for i in range(8):
        if box.top[i] >= 1.0:
            continue

        # Compute Production in mol/m2/yr from the rain.P that was already computed
        # rain.P[i] is in CtoN-scaled units: actual mol P stripped = rain.P[i] * NtoC[i]
        # But rain.P has already been multiplied by CtoN at end of prod(),
        # so actual_molP_stripped = rain.P[i] * box.NtoC[i]
        # But we need Production = actual_molP / (Area * timestep) for consistency
        # Actually, the Python model already stripped nutrients in prod().
        # For the N isotope sub-stepping, we need to know the production rate.
        # Let's store it from the rain.P before CtoN scaling.
        # Production is stored in box.Production[i] set earlier.
        Production_i = box.Production[i]  # mol P/m2/yr, set during N-cycle init in run_ex

        if Production_i <= 0:
            rain.d15Norg[i] = 0.0
            rain.d18ONorg[i] = 0.0
            continue

        Vol_i = box.vol[i]
        Area_i = box.Area[i]

        # Save old N isotope concentrations (before sub-stepping strips N)
        oldN = tracer.N[i]
        oldN15 = tracer.N15[i]
        oldNO18 = tracer.NO18[i]

        # Redfield N:P ratio
        RedN = box.Redfield_N[i]

        # Sub-stepping for isotope fractionation during production
        for q in range(PRODUCTIONTIMESTEPS):
            # Compute current isotope composition of organic matter
            if tracer.N[i] > 1e-30:
                F15 = tracer.N15[i] / tracer.N[i]
                R15org = param.alphaN15org * F15 / (1.0 - F15)
                d15Norg = ((R15org - RAIR) / RAIR) * 1000.0
                RedN15 = RedN * IsoConcN(d15Norg)

                F18 = tracer.NO18[i] / tracer.N[i]
                R18org = param.alphaNO18org * F18 / (1.0 - F18)
                d18ONorg = ((R18org - RPDB) / RPDB) * 1000.0
                RedNO18 = RedN * IsoConcPDB(d18ONorg)
            else:
                RedN15 = 0.0
                RedNO18 = 0.0

            dN = Production_i * Area_i * RedN / (PRODUCTIONTIMESTEPS) / (Vol_i * KGPERM)
            dN15 = Production_i * Area_i * RedN15 / (PRODUCTIONTIMESTEPS) / (Vol_i * KGPERM)
            dNO18 = Production_i * Area_i * RedNO18 / (PRODUCTIONTIMESTEPS) / (Vol_i * KGPERM)
            tracer.N[i] -= dN
            tracer.N15[i] -= dN15
            tracer.NO18[i] -= dNO18

        # Compute integrated del15N of organic matter produced
        strippedN = oldN - tracer.N[i]
        strippedN15 = oldN15 - tracer.N15[i]
        strippedNO18 = oldNO18 - tracer.NO18[i]
        if strippedN > 1e-30:
            rain.d15Norg[i] = IsoDelN(strippedN15, strippedN)
            rain.d18ONorg[i] = IsoDelPDB(strippedNO18, strippedN)
        else:
            rain.d15Norg[i] = 0.0
            rain.d18ONorg[i] = 0.0

        # ---- N2 fixation (legacy in-production; disabled only when v2 fixation is on) ----
        _v2fix = getattr(param, "NfluxV2", False) and getattr(param, "NfixV2", False)
        _fixthr = getattr(param, "fixThreshold", 0.016)
        if tracer.N[i] < _fixthr and Production_i > 0 and not _v2fix:
            fixedN = (-tracer.N[i] + _fixthr) * Vol_i * KGPERM  # µmoles
            fixedN15 = IsoConcN(param.fixdelta) * fixedN
            # d18O of newly fixed nitrate: from water d18O
            fixedNO18_F = ((1.0 - param.O2innewNitrate) * IsoConcPDB(tracer.H2Od18O[i])
                           + param.O2innewNitrate * IsoConcPDB(IsoDelPDB(tracer.O218[i], tracer.O2[i])))
            fixedNO18 = fixedN * fixedNO18_F
            box.fixedNtotal[i] += fixedN

            # Restore pre-production N concentrations and add fixed N
            tracer.NO18[i] = oldNO18
            tracer.N[i] = oldN
            tracer.N15[i] = oldN15
            tracer.NO18[i] += fixedNO18 / (Vol_i * KGPERM)
            tracer.N[i] += fixedN / (Vol_i * KGPERM)
            tracer.N15[i] += fixedN15 / (Vol_i * KGPERM)

            # Re-do production sub-stepping including the fixed N
            oldN2 = tracer.N[i]
            oldN152 = tracer.N15[i]
            oldNO182 = tracer.NO18[i]
            for q in range(PRODUCTIONTIMESTEPS):
                if tracer.N[i] > 1e-30:
                    F15 = tracer.N15[i] / tracer.N[i]
                    R15org = param.alphaN15org * F15 / (1.0 - F15)
                    d15Norg = ((R15org - RAIR) / RAIR) * 1000.0
                    RedN15 = RedN * IsoConcN(d15Norg)
                    F18 = tracer.NO18[i] / tracer.N[i]
                    R18org = param.alphaNO18org * F18 / (1.0 - F18)
                    d18ONorg = ((R18org - RPDB) / RPDB) * 1000.0
                    RedNO18 = RedN * IsoConcPDB(d18ONorg)
                else:
                    RedN15 = 0.0
                    RedNO18 = 0.0
                dN = Production_i * Area_i * RedN / PRODUCTIONTIMESTEPS / (Vol_i * KGPERM)
                tracer.N[i] -= dN
                tracer.N15[i] -= Production_i * Area_i * RedN15 / PRODUCTIONTIMESTEPS / (Vol_i * KGPERM)
                tracer.NO18[i] -= Production_i * Area_i * RedNO18 / PRODUCTIONTIMESTEPS / (Vol_i * KGPERM)

            strippedN = oldN2 - tracer.N[i]
            strippedN15 = oldN152 - tracer.N15[i]
            strippedNO18 = oldNO182 - tracer.NO18[i]
            if strippedN > 1e-30:
                rain.d15Norg[i] = IsoDelN(strippedN15, strippedN)
                rain.d18ONorg[i] = IsoDelPDB(strippedNO18, strippedN)

        # ---- Shallow sediment denitrification ----
        # Threshold on residual surface phosphate below which the box is treated as a
        # productive margin with shallow-sediment denitrification. The reference
        # Sigman2009b/Pascal value (0.1e-6) sits below the low-latitude P restoring floor
        # (~0.001), so this term is DORMANT by default -- sediment denitrification is
        # carried in the interior boxes, exactly as in the validated 18-box reference.
        # Raise param.shallowSedPThreshold (e.g. to 0.1) to activate margin denitrification.
        if (Production_i > 0 and tracer.P[i] < getattr(param, 'shallowSedPThreshold', 0.1e-6)
                and param.sedimentdenitrification):
            RedC = 106.0  # C:P Redfield
            F100 = Production_i * RedC / (1e4 * 365)  # µmol C/(cm2*d) at 100m
            DenitriCarbon = 0.0
            for L in range(TOTALDEPTHLEVELS):
                sf_depth = box.SFdepth_levels[i, L]
                sf_fract = box.SFfractarea_levels[i, L]
                if sf_depth <= 0:
                    continue
                if sf_depth * 1000 > box.top[i] and sf_depth * 1000 <= box.bottom[i]:
                    F = (1.0 + param.scalingShallow * ((-sf_depth + 3.5) / 3.5)) * F100 * (sf_depth * 1000 / 100) ** param.MartinB
                    if F > 0:
                        loose = -0.9543 + 0.7662 * np.log(F) - 0.235 * np.log(F)**2
                        loose = np.exp(loose)
                        DenitriCarbon += loose * sf_fract * Area_i * 1e4 * 365

            box.shelfDenitFract[i] = 0.0
            if DenitriCarbon > 0:
                lostNsed = DenitriCarbon * 110.4 / 106.0
                box.fixedN[i] = getattr(box, 'fixedN', np.zeros(8))[i]
                box.fixedN[i] += lostNsed
                lostFract = lostNsed / (Production_i * RedN * Area_i)
                # Isotope-only mode (legacy): mix the organic-rain d15N with the fixation
                # signature (-1‰). Skipped when the real sink is active (the real removal
                # below handles the N and its isotopes, with no shelf-denit fractionation).
                if not getattr(param, 'shelfDenitReal', False):
                    rain.d15Norg[i] = (1.0 - lostFract) * rain.d15Norg[i] + lostFract * (-1.0)
                # REAL shelf-sediment N sink (param.shelfDenitReal): the denitrified N is
                # removed from the exported organic matter so it never remineralizes in the
                # interior, and is counted in the denitrification budget. Balanced at steady
                # state by the in-production N2 fixation. (Default off -> isotope-only, as before.)
                if getattr(param, 'shelfDenitReal', False):
                    box.shelfDenitFract[i] = min(max(lostFract, 0.0), 0.95)
                    box.lostNsedtotal[i] += lostNsed

        # O2 consumption during production (Redfield O is negative = consumption)
        tracer.O2[i] -= Production_i * Area_i * box.Redfield_O[i] / (Vol_i * KGPERM)


#::
#:: --- def ncycle_remin_box ---
#:: Remineralize the sinking organic rain reaching one interior box (or one sub-compartment of it):
#:: adds C/P/N/alkalinity/isotopes back and consumes O2. Called by the OMZ manager once per compartment.
def ncycle_remin_box(i, ocean, usedFract, supplyTray, param):
    """Remineralization for a single subsurface box using ComponentTray approach.
    Adds organic matter (C, P, Alk, C13, C14, N, N15, NO18) and dissolves carbonate.
    Also handles O2 consumption with sub-stepping for O218 fractionation.

    ComponentTray[i] must be initialized with box contents * volume * KGPERM before calling.
    """
    tracer = ocean.tracer
    box = ocean.box
    tray = box.ComponentTray

    REMINTIMESTEPS = 20
    TIMESTEP = 1.0

    # Accumulate organic rain from all surface boxes
    AddedC = 0.0
    AddedP = 0.0
    AddedAlk = 0.0
    AddedN = 0.0
    AddedN15 = 0.0
    AddedNO18 = 0.0
    AddedC13 = 0.0
    AddedC14 = 0.0
    AddedO_total = 0.0

    for j in range(8):
        Prod_j = box.Production[j]
        if Prod_j <= 0:
            continue
        Area_j = box.Area[j]
        Ox_ij = ocean.RainOrg[i - 8, j] if i >= 8 else 0.0  # oxidation fraction
        if Ox_ij <= 0:
            continue
        RedC = 106.0
        RedP = 1.0
        RedN = box.Redfield_N[j]
        # Real shelf-sediment denitrification: a fraction of this surface box's exported N
        # was denitrified on the margin and never reaches the interior (param.shelfDenitReal).
        if getattr(param, 'shelfDenitReal', False):
            RedN = RedN * (1.0 - box.shelfDenitFract[j])
        RedO = box.Redfield_O[j]

        flux = usedFract * supplyTray * Prod_j * Area_j * TIMESTEP

        AddedC += flux * RedC * Ox_ij
        AddedP += flux * RedP * Ox_ij
        AddedAlk -= flux * RedN * Ox_ij  # Alk decreases with nitrification
        AddedN += flux * RedN * Ox_ij
        AddedO_total += flux * RedO * Ox_ij

        # N15 from organic rain
        d15Norg_j = ocean.rain.d15Norg[j]
        AddedN15 += flux * RedN * IsoConcN(d15Norg_j) * Ox_ij

        # NO18: newly nitrified nitrate gets water d18O (or O2 d18O mix)
        if tray[i, _iO] > 0:
            F_NO18_new = ((1.0 - param.O2innewNitrate) * IsoConcPDB(tracer.H2Od18O[i])
                          + param.O2innewNitrate * (tray[i, _iO218] / tray[i, _iO]))
        else:
            F_NO18_new = IsoConcPDB(tracer.H2Od18O[i])
        AddedNO18 += flux * RedN * F_NO18_new * Ox_ij

        # C13, C14 isotopes of organic rain
        d13Corg_j = ocean.rain.d13Corg[j]
        d14Corg_j = ocean.rain.d14Corg[j]
        AddedC13 += flux * RedC * IsoConcPDB(d13Corg_j) * Ox_ij  # IsoConc for C13 uses Rpdb
        AddedC14 += flux * RedC * (((d14Corg_j / 1000.0 + 1.0) * NBSC14RATIO) / (1.0 + (d14Corg_j / 1000.0 + 1.0) * NBSC14RATIO)) * Ox_ij

    # ---- v27: native seaweed organic carbon, respired where it lands ----------------------
    # Flag-gated and off by default, so v26 behaviour is unchanged when nativeSeaweedCarbon is
    # not set. box.seaweedCflux[i] (mol C, set by the caller before run_ex) is split across the
    # oxic/OMZ/anoxic-aerobic compartments by the SAME usedFract*supplyTray focusing weights as
    # natural sinking rain, and consumes O2 at the Redfield O2:C ratio (170/106). Seaweed's own
    # N and P are cycled separately (exactly-conservatively) by the caller -- this path adds
    # carbon only, so it does not touch AddedN/AddedP/AddedN15/AddedNO18 above.
    if getattr(param, 'nativeSeaweedCarbon', False):
        SWflux = getattr(box, 'seaweedCflux', None)
        if SWflux is not None and SWflux[i] != 0.0:
            extraC = usedFract * supplyTray * SWflux[i]
            AddedC += extraC
            AddedO_total += extraC * (170.0 / 106.0)

    # Add to ComponentTray
    tray[i, _iC] += AddedC
    tray[i, _iP] += AddedP
    tray[i, _iAlk] += AddedAlk
    tray[i, _iN] += AddedN
    tray[i, _iN15] += AddedN15
    tray[i, _iNO18] += AddedNO18
    tray[i, _iC13] += AddedC13
    tray[i, _iC14] += AddedC14

    # O2 consumption with sub-stepping for O218 fractionation
    # Pascal uses Comp[O218]/Comp[O] (the main box tracer values, NOT the tray).
    # This means the isotope ratio is constant across sub-steps (no Rayleigh effect
    # within a single remineralization call).  After oxic remin, comp[] is updated,
    # so OMZ/ANOX remin inherits the post-oxic-remin ratio.
    if AddedO_total != 0:
        if tracer.O2[i] > 1e-30:
            IsoVal = param.alphaO218 * (tracer.O218[i] / tracer.O2[i])
        else:
            IsoVal = 0.0
        for q in range(REMINTIMESTEPS):
            tray[i, _iO] += AddedO_total * (TIMESTEP / REMINTIMESTEPS)
            tray[i, _iO218] += AddedO_total * IsoVal * (TIMESTEP / REMINTIMESTEPS)

    # Also handle carbonate dissolution C13/C14 (from standard dissolution pathway)
    # This is already handled by dissolve() in the main loop, so we skip it here.
    # The ComponentTray approach only adds the organic matter remineralization products.

    return AddedC  # return C added (used by denitrification)


#::
#:: --- def ncycle_denitrification ---
#:: WATER-COLUMN DENITRIFICATION in an anoxic core: respire organic carbon using nitrate instead of
#:: O2. Nitrate removed = denitriparam * organic-C; residual nitrate is Rayleigh-enriched (alpha ~ 0.975, ~ -25 per mil).
def ncycle_denitrification(i, ocean, usedFract, supplyTray, AddedComponentC, AddedP, AddedC13, AddedC14, param):
    """Water column denitrification with Rayleigh fractionation.

    In Pascal, denitrification adds C, P, C13, C14 from organic matter
    remineralization (using NO3 instead of O2) to the ComponentTray, then
    removes dissolved NO3 via the denitrification stoichiometry.  This
    function now mirrors that: first add the organic C/P/isotopes, then
    remove N.
    """
    box = ocean.box
    tray = box.ComponentTray

    # --- Add organic C, P, C13, C14 to tray (anaerobic remineralization) ---
    tray[i, _iC] += AddedComponentC
    tray[i, _iP] += AddedP
    tray[i, _iC13] += AddedC13
    tray[i, _iC14] += AddedC14

    # Calculate N lost through denitrification (Martin & Sayles stoichiometry)
    lostN = param.denitriparam * AddedComponentC
    if lostN <= 0 or tray[i, _iN] <= 0:
        box.lostN[i] = 0.0
        return
    # Cap per-step denitrification to leave a residual nitrate fraction in the box. A single
    # annual step never removes more than 90% of the box nitrate; the remainder is left for
    # subsequent steps (and resupplied by circulation). This bounds the Rayleigh enrichment
    # so the residual isotope fraction cannot ratchet toward 1 and produce inf/NaN under
    # strong (e.g. high-seaweed) forcing, while leaving normal-regime denitrification
    # unchanged (there lostN << tray[N]).
    maxLost = 0.90 * tray[i, _iN]
    if lostN > maxLost:
        lostN = maxLost
    if lostN <= 0:
        box.lostN[i] = 0.0
        return

    box.lostN[i] = lostN

    # Rayleigh fractionation
    unusedF = 1.0 - lostN / tray[i, _iN]
    if unusedF <= 0 or tray[i, _iN] <= 0:
        box.lostN[i] = 0.0
        return

    F_N15_old = tray[i, _iN15] / tray[i, _iN]
    F_NO18_old = tray[i, _iNO18] / tray[i, _iN]

    newF_N15 = RtoF(unusedF ** (param.alphadntrN15 - 1.0) * FtoR(F_N15_old))
    newF_NO18 = RtoF(unusedF ** (param.alphadntrNO18 - 1.0) * FtoR(F_NO18_old))
    # Defensive clamp: keep residual isotope fractions physical (well above any real
    # value; only catches pathological ratcheting). F=0.05 ~ delta +1.3e4 permil.
    newF_N15 = min(max(newF_N15, 0.0), 0.05)
    newF_NO18 = min(max(newF_NO18, 0.0), 0.05)

    # Remove N from tray
    tray[i, _iN] -= lostN
    tray[i, _iN15] = tray[i, _iN] * newF_N15
    tray[i, _iNO18] = tray[i, _iN] * newF_NO18

    # Additional N lost as ammonium from organic rain (denitrparam2)
    # This N never enters the water (it's lost as NH4 from the organic matter itself)
    lostNH4 = AddedComponentC * param.denitriparam2
    box.lostN[i] += lostNH4
    box.lostNtotal[i] += box.lostN[i]


#::
#:: --- def ncycle_sed_denitrification ---
#:: SEDIMENT DENITRIFICATION (Middelburg 1996): nitrate removed in the sediments as a function of
#:: the organic-carbon rain hitting the sea floor, integrated over the box's seafloor depth levels. Nearly no
#:: isotope effect -- the asymmetry with water-column denitrification is central to the global d15N budget.
def ncycle_sed_denitrification(i, ocean, oxTray, param):
    """Sediment denitrification using Middelburg (1996) parameterization.
    For non-surface boxes only.
    """
    box = ocean.box
    tracer = ocean.tracer

    if not param.sedimentdenitrification:
        box.lostNsed[i] = 0.0
        return

    # Calculate organic carbon flux at 100m depth into this box
    F100 = 0.0
    deltaorg = 0.0
    AddedC_total = 0.0
    TIMESTEP = 1.0

    # Compute Area_i for this box (matches Pascal ocean[i].area)
    depth_range = box.bottom[i] - box.top[i]
    if i < 8:
        Area_i = box.Area[i]
    elif depth_range > 0:
        Area_i = box.completeVolume[i] / depth_range  # m² (volume/depth)
    else:
        return

    for s in range(8):
        Prod_s = box.Production[s]
        if Prod_s <= 0:
            continue
        Ox_is = ocean.RainOrg[i - 8, s] if i >= 8 else 0.0
        if Ox_is <= 0:
            continue
        Area_s = box.Area[s]
        # Pascal: F100 += production[s]*area[s]/area[i]*oxidation[i,s]*redfield[C]*Timestep*1E6/(1E4*365)
        F100 += Prod_s * Area_s / Area_i * Ox_is * 106.0 * TIMESTEP / (1e4 * 365)
        cflux = Prod_s * Area_s * 106.0 * Ox_is * TIMESTEP
        AddedC_total += cflux
        deltaorg += ocean.rain.d15Norg[s] * cflux

    if AddedC_total > 0:
        deltaorg /= AddedC_total
    else:
        box.lostNsed[i] = 0.0
        return

    # For non-surface boxes, need to correct F100 to actual 100m flux
    # In Pascal: F100 := F100/pow(ocean[i].topdepth/100, MartinB)
    if i >= 8 and box.top[i] > 0:
        F100 = F100 / (box.top[i] / 100.0) ** param.MartinB

    if F100 <= 0:
        box.lostNsed[i] = 0.0
        return

    # Calculate sediment denitrification at each depth level
    lostNsed = 0.0
    for L in range(TOTALDEPTHLEVELS):
        sf_depth = box.SFdepth_levels[i, L]
        sf_fract = box.SFfractarea_levels[i, L]
        if sf_depth <= 0:
            continue
        if sf_depth * 1000 > box.top[i] and sf_depth * 1000 < box.bottom[i]:
            F = (1.0 + param.scalingDeep * ((-sf_depth + 3.5) / 3.5)) * F100 * (sf_depth * 1000 / 100) ** param.MartinB
            if F > 0:
                loose = -0.9543 + 0.7662 * np.log(F) - 0.235 * np.log(F) ** 2
                loose = np.exp(loose)
                loose = loose * sf_fract * Area_i * 1e4 * 365
                lostNsed += loose * 110.4 / 106.0

    box.lostNsed[i] = lostNsed
    box.lostNsedtotal[i] += lostNsed

    if lostNsed <= 0:
        return

    # Isotope composition of lost N: mix of organic N15 and water column NO3 N15
    # deltaNlostsed = (16*deltaorg + 94.4*IsoDelN(N15,N))/110.4
    d15N_water = IsoDelN(tracer.N15[i], tracer.N[i]) if tracer.N[i] > 1e-30 else 0.0
    deltaNlostsed = (16.0 * deltaorg + 94.4 * d15N_water) / 110.4
    lostN15sed = IsoConcN(deltaNlostsed) * lostNsed

    # NO18 lost with same isotope ratio as water column
    d18O_water = IsoDelPDB(tracer.NO18[i], tracer.N[i]) if tracer.N[i] > 1e-30 else 0.0
    lostNO18sed = IsoConcPDB(d18O_water) * lostNsed

    # Remove from water column (use completeVolume for robustness - Pascal uses current Volume)
    V = box.vol[i] if box.vol[i] > 0 else box.completeVolume[i]
    if V <= 0:
        return
    tracer.N[i] = (tracer.N[i] * V * KGPERM - lostNsed) / (V * KGPERM)
    tracer.N15[i] = (tracer.N15[i] * V * KGPERM - lostN15sed) / (V * KGPERM)
    tracer.NO18[i] = (tracer.NO18[i] * V * KGPERM - lostNO18sed) / (V * KGPERM)


#::
#:: ############################################################################################
#:: ## OMZ SUB-VOLUME MANAGER (the model's most distinctive part) -- hides a sub-grid
#:: ## suboxic pocket inside each interior box so a box-mean model can host a realistic, tiny
#:: ## anoxic core where water-column denitrification fires. Sizes the pocket from predicted O2,
#:: ## splits it into an OMZ shell and an anoxic core, exchanges them with the oxic majority, and
#:: ## respires/denitrifies each compartment.
#:: ############################################################################################
#::
#:: --- def ncycle_manager ---
#:: The OMZ manager loop over all boxes. Step 1: predict O2 (TestOx) and size the suboxic sub-volume
#:: (grows as O2 falls below 200). Step 2: pick one of four behaviours (stays oxic / pocket persists / pocket forms /
#:: pocket collapses). Steps 3-4 (behaviour 2/3): split into OMZ shell + anoxic core, exchange with the oxic
#:: majority at a capped rate, focus the organic rain into the pocket, and respire/denitrify each compartment.
def ncycle_manager(ocean, param):
    """OMZ Manager: manages sub-volumes (oxic, OMZ, anoxic) in each box.
    Implements 4 behaviors based on O2 levels.
    Calls remineralization and denitrification for each sub-volume.

    This is the central procedure for water column denitrification.
    Replaces the standard remin() call for N-cycle tracers.
    """
    tracer = ocean.tracer
    box = ocean.box
    TIMESTEP = 1.0

    for i in range(NB):
        # Initialize working arrays
        box.ComponentTray[i, :] = 0.0

        # Compute TestOx: O2 that would result after remineralization
        # Pascal uses ocean[i].volume (the current oxic volume, NOT completeVolume).
        # This means the O2 consumption from organic rain is concentrated in the
        # oxic sub-volume, making TestOx lower and OMZ formation more likely.
        V_test = box.vol[i]
        if V_test <= 0:
            continue  # skip zero-volume boxes
        TestOx = tracer.O2[i] * V_test * KGPERM
        for t in range(8):
            Prod_t = box.Production[t]
            if Prod_t <= 0:
                continue
            Ox_it = ocean.RainOrg[i - 8, t] if i >= 8 else 0.0
            if Ox_it > 0:
                TestOx += Prod_t * box.Area[t] * box.Redfield_O[t] * Ox_it * TIMESTEP
        TestOx = TestOx / (KGPERM * V_test)

        # Store old OMZ state
        oldVolFraction = box.VolFraction[i]
        olddV = box.dV[i]
        oldVolume = box.vol[i]
        oldomzV = box.omzV[i]
        oldanoxV = box.anoxV[i]

        # Calculate new suboxic volume fraction
        if TestOx < 200.0 and param.watercolumndenitrification:
            VolFraction = 4.0 * param.OxygenVolumeParam * ((-TestOx) + 200.0)
        else:
            VolFraction = 0.0
        VolFraction = max(0.0, min(1.0, VolFraction))
        # ---- v27: shallowODZonly -- confine the denitrifying suboxic pocket to the thermocline,
        # keep the deep ocean (DEEP_IDX) oxygenated even if its box-mean O2 would otherwise let a
        # pocket form. Flag-gated and off by default, so v26 is unchanged when unset. The baseline
        # is essentially untouched by this flag because deep boxes don't denitrify at baseline
        # anyway (see doc) -- it only matters once a sustained deployment tries to drive the deep
        # ocean suboxic.
        if getattr(param, 'shallowODZonly', False) and i in DEEP_IDX:
            VolFraction = 0.0
        box.VolFraction[i] = VolFraction

        dV = box.completeVolume[i] * VolFraction
        box.dV[i] = dV

        # Determine behavior
        if dV == 0 and olddV == 0:
            behaviour = 1
        elif dV > 0 and olddV > 0:
            behaviour = 2
        elif dV > 0 and olddV == 0:
            behaviour = 3
        else:  # dV == 0 and olddV > 0
            behaviour = 4

        if behaviour == 1:
            # No sub-volumes: standard remineralization, no WC denitrification
            oxTray = 1.0
            if i >= 8:
                ncycle_sed_denitrification(i, ocean, oxTray, param)
            box.vol[i] = box.completeVolume[i]

            # Fill ComponentTray with box contents
            _fill_tray_from_tracer(i, box, tracer)

            # Remineralize
            AddedC = ncycle_remin_box(i, ocean, 1.0, oxTray, param)

            box.lostN[i] = 0.0  # no WC denitrification

            # Write back to tracers
            _write_tray_to_tracer(i, box, tracer)

        elif behaviour == 2:
            # Both sub-volumes exist: full OMZ dynamics
            _manager_behaviour2(i, ocean, param, olddV, oldVolume, oldomzV, oldanoxV, dV, VolFraction)

        elif behaviour == 3:
            # OMZ forms for the first time
            _manager_behaviour3(i, ocean, param, olddV, oldVolume, oldomzV, dV, VolFraction)

        elif behaviour == 4:
            # OMZ collapses: merge everything back
            oxTray = 1.0
            if i >= 8:
                ncycle_sed_denitrification(i, ocean, oxTray, param)

            # Merge all sub-volumes back into main box
            Vol_total = box.completeVolume[i]
            if Vol_total > 0:
                for j in range(10):
                    tracer_val = _get_tracer_by_idx(tracer, i, j)
                    omz_val = box.CompOMZ[i, j]
                    anox_val = box.CompANOX[i, j]
                    merged = (tracer_val * oldVolume * KGPERM
                              + omz_val * oldomzV * KGPERM
                              + anox_val * oldanoxV * KGPERM) / (Vol_total * KGPERM)
                    _set_tracer_by_idx(tracer, i, j, merged)

            box.vol[i] = Vol_total

            _fill_tray_from_tracer(i, box, tracer)
            AddedC = ncycle_remin_box(i, ocean, 1.0, oxTray, param)
            _write_tray_to_tracer(i, box, tracer)

            box.omzV[i] = 0.0
            box.anoxV[i] = 0.0
            box.lostN[i] = 0.0


#::
#:: --- def _fill_tray_from_tracer ---
#:: Copy a box's tracer concentrations into the ComponentTray working array (as absolute amounts).
def _fill_tray_from_tracer(i, box, tracer):
    """Fill ComponentTray[i] from tracer concentrations * volume * KGPERM."""
    V = box.vol[i] * KGPERM
    box.ComponentTray[i, _iC] = tracer.C[i] * V
    box.ComponentTray[i, _iO] = tracer.O2[i] * V
    box.ComponentTray[i, _iP] = tracer.P[i] * V
    box.ComponentTray[i, _iAlk] = tracer.Alk[i] * V
    box.ComponentTray[i, _iC13] = tracer.dc13[i] * V
    box.ComponentTray[i, _iC14] = tracer.dc14[i] * V
    box.ComponentTray[i, _iN] = tracer.N[i] * V
    box.ComponentTray[i, _iN15] = tracer.N15[i] * V
    box.ComponentTray[i, _iNO18] = tracer.NO18[i] * V
    box.ComponentTray[i, _iO218] = tracer.O218[i] * V


#::
#:: --- def _write_tray_to_tracer ---
#:: Write the ComponentTray working amounts back to the box's tracer concentrations.
def _write_tray_to_tracer(i, box, tracer):
    """Write ComponentTray[i] back to tracer concentrations."""
    V = box.vol[i] * KGPERM
    if V <= 0:
        return
    tracer.C[i] = box.ComponentTray[i, _iC] / V
    tracer.O2[i] = box.ComponentTray[i, _iO] / V
    tracer.P[i] = box.ComponentTray[i, _iP] / V
    tracer.Alk[i] = box.ComponentTray[i, _iAlk] / V
    tracer.dc13[i] = box.ComponentTray[i, _iC13] / V
    tracer.dc14[i] = box.ComponentTray[i, _iC14] / V
    tracer.N[i] = box.ComponentTray[i, _iN] / V
    tracer.N15[i] = box.ComponentTray[i, _iN15] / V
    tracer.NO18[i] = box.ComponentTray[i, _iNO18] / V
    tracer.O218[i] = box.ComponentTray[i, _iO218] / V


#::
#:: --- def _get_tracer_by_idx ---
#:: Read one tracer of a box by its ComponentTray index (C,O2,P,Alk,13C,14C,N,15N,NO18,O218).
def _get_tracer_by_idx(tracer, i, idx):
    """Get tracer value by ComponentTray index."""
    if idx == _iC: return tracer.C[i]
    elif idx == _iO: return tracer.O2[i]
    elif idx == _iP: return tracer.P[i]
    elif idx == _iAlk: return tracer.Alk[i]
    elif idx == _iC13: return tracer.dc13[i]
    elif idx == _iC14: return tracer.dc14[i]
    elif idx == _iN: return tracer.N[i]
    elif idx == _iN15: return tracer.N15[i]
    elif idx == _iNO18: return tracer.NO18[i]
    elif idx == _iO218: return tracer.O218[i]
    return 0.0


#::
#:: --- def _set_tracer_by_idx ---
#:: Write one tracer of a box by its ComponentTray index.
def _set_tracer_by_idx(tracer, i, idx, val):
    """Set tracer value by ComponentTray index."""
    if idx == _iC: tracer.C[i] = val
    elif idx == _iO: tracer.O2[i] = val
    elif idx == _iP: tracer.P[i] = val
    elif idx == _iAlk: tracer.Alk[i] = val
    elif idx == _iC13: tracer.dc13[i] = val
    elif idx == _iC14: tracer.dc14[i] = val
    elif idx == _iN: tracer.N[i] = val
    elif idx == _iN15: tracer.N15[i] = val
    elif idx == _iNO18: tracer.NO18[i] = val
    elif idx == _iO218: tracer.O218[i] = val


#::
#:: --- def _manager_behaviour2 ---
#:: OMZ behaviour 2 (pocket persists from last year -- the full machinery): recompute the pocket
#:: geometry, split into OMZ shell + anoxic core, run the capped internal exchange, focus the rain (k2), then
#:: remineralize the oxic/OMZ/anoxic-aerobic parts and DENITRIFY the anoxic core (gated by k4).
def _manager_behaviour2(i, ocean, param, olddV, oldVolume, oldomzV, oldanoxV, dV, VolFraction):
    """Manager behaviour 2: both OMZ and ANOX sub-volumes exist."""
    tracer = ocean.tracer
    box = ocean.box
    

    # Cross-section geometry (ellipsoidal)
    depth_range = box.bottom[i] - box.top[i]
    if depth_range > 0:
        box.crossArea[i] = dV * 3.0 / (4.0 * depth_range * 0.5)
        r = (3.0 * dV / (4.0 * np.pi * depth_range * 0.5)) ** 0.5
        box.crossArea2[i] = np.pi * depth_range * r
    else:
        box.crossArea[i] = 0.0
        box.crossArea2[i] = 0.0

    # New volumes
    Volume_new = box.completeVolume[i] - dV
    oxGrad = box.CompOMZ[i, _iO] / tracer.O2[i] if tracer.O2[i] > 1e-30 else 0.99
    # Clamp to [0.01, 0.99]: keep a thin suboxic shell (omzV never collapses to exactly 0,
    # which would make the sub-volume bookkeeping degenerate) and keep a thin oxic-edge
    # (anoxV never takes the whole sub-volume).
    oxGrad = max(0.01, min(oxGrad, 0.99))
    anoxV = (1.0 - oxGrad) * dV
    omzV = dV - anoxV
    box.anoxV[i] = anoxV
    box.omzV[i] = omzV

    # Internal flux
    angle = 2.0 * np.pi * param.fluxangle / 360.0
    Flux = (np.cos(angle) * param.InternalFlux * SECPERYEAR * box.crossArea2[i]
            + np.sin(angle) * param.InternalFlux * SECPERYEAR * box.crossArea[i])
    # Cap the suboxic<->surrounding exchange at a fractional turnover of the suboxic
    # sub-volume per year (a physical, geometry-independent timescale) rather than at
    # 100% (omzV). The original Pascal cap (turnover=1.0) lets the large appended boxes
    # fully re-mix the anoxic core with box-mean water every step, pinning CompANOX_O2 >=
    # CompOMZ_O2 (k4==1 -> no water-column denitrification). A turnover < 1 lets the core
    # hold its respiration drawdown so the anoxic core forms. maxInternalTurnover is
    # size-independent, so the same value works across NLAYER without re-tuning.
    box.Flux[i] = min(Flux, param.maxInternalTurnover * omzV)
    Flux = box.Flux[i]

    # Organic flux focusing
    k2 = ((((VolFraction * 100) - 100) ** 14 / 1.5e27) + 1.0) * param.orgfluxscale
    # Pascal: oxTray := 1 - k2*crossArea/Area, where Area = total box area
    depth_range_i = box.bottom[i] - box.top[i]
    Area_i = box.completeVolume[i] / depth_range_i if depth_range_i > 0 else 1.0
    oxTray = 1.0 - k2 * box.crossArea[i] / max(Area_i, 1.0)
    anoxTray = k2 * box.crossArea[i] / max(Area_i, 1.0)
    if oxTray < 0:
        oxTray = 0.0
        anoxTray = 1.0

    # Sediment denitrification
    if i >= 8:
        ncycle_sed_denitrification(i, ocean, oxTray, param)

    # Volume change: redistribute tracers between oxic and OMZ+ANOX
    changedV = dV - olddV
    for j in range(10):
        comp_val = _get_tracer_by_idx(tracer, i, j)
        omz_val = box.CompOMZ[i, j]
        if changedV < 0:
            # Suboxic volume shrinking: water returns to oxic at suboxic concentration.
            # When omzV ≈ 0, CompOMZ may be stale/unphysical. Use volume-weighted
            # average of OMZ + ANOX concentrations for the returning water.
            if oldomzV + oldanoxV > 1e-10:
                suboxic_val = (omz_val * oldomzV + box.CompANOX[i, j] * oldanoxV) / (oldomzV + oldanoxV)
            else:
                suboxic_val = omz_val
            add_anox = changedV * suboxic_val * KGPERM
            add_ox = -changedV * suboxic_val * KGPERM
        else:
            add_anox = changedV * comp_val * KGPERM
            add_ox = -changedV * comp_val * KGPERM
        new_comp = (comp_val * oldVolume * KGPERM + add_ox) / (Volume_new * KGPERM) if Volume_new > 0 else comp_val
        new_omz = (omz_val * oldomzV * KGPERM + add_anox) / ((dV - oldanoxV) * KGPERM) if (dV - oldanoxV) > 0 else omz_val
        _set_tracer_by_idx(tracer, i, j, new_comp)
        box.CompOMZ[i, j] = new_omz

    # Redistribute between OMZ and ANOX
    changedV_anox = anoxV - oldanoxV
    for j in range(10):
        omz_val = box.CompOMZ[i, j]
        anox_val = box.CompANOX[i, j]
        if changedV_anox < 0:
            add_anox = changedV_anox * anox_val * KGPERM
            add_ox = -changedV_anox * anox_val * KGPERM
        else:
            add_anox = changedV_anox * omz_val * KGPERM
            add_ox = -changedV_anox * omz_val * KGPERM
        new_omz = (omz_val * (dV - oldanoxV) * KGPERM + add_ox) / (omzV * KGPERM) if omzV > 0 else omz_val
        new_anox = (anox_val * oldanoxV * KGPERM + add_anox) / (anoxV * KGPERM) if anoxV > 0 else anox_val
        box.CompOMZ[i, j] = new_omz
        box.CompANOX[i, j] = new_anox

    # Internal flux: oxic <-> OMZ
    if Flux > 0:
        for j in range(10):
            comp_val = _get_tracer_by_idx(tracer, i, j)
            omz_val = box.CompOMZ[i, j]
            add_anox = Flux * comp_val * KGPERM
            add_ox = Flux * omz_val * KGPERM
            new_comp = ((Volume_new - Flux) * comp_val * KGPERM + add_ox) / (Volume_new * KGPERM) if Volume_new > 0 else comp_val
            new_omz = ((omzV - Flux) * omz_val * KGPERM + add_anox) / (omzV * KGPERM) if omzV > 0 else omz_val
            _set_tracer_by_idx(tracer, i, j, new_comp)
            box.CompOMZ[i, j] = new_omz

    # Internal flux: OMZ <-> ANOX
    Flux2 = Flux * anoxV / dV if dV > 0 else 0.0
    if omzV < anoxV:
        Flux2 = min(Flux2, omzV)
    else:
        Flux2 = min(Flux2, anoxV)
    box.Flux2[i] = Flux2

    if Flux2 > 0:
        for j in range(10):
            omz_val = box.CompOMZ[i, j]
            anox_val = box.CompANOX[i, j]
            add_anox = Flux2 * omz_val * KGPERM
            add_ox = Flux2 * anox_val * KGPERM
            new_omz = ((omzV - Flux2) * omz_val * KGPERM + add_ox) / (omzV * KGPERM) if omzV > 0 else omz_val
            new_anox = ((anoxV - Flux2) * anox_val * KGPERM + add_anox) / (anoxV * KGPERM) if anoxV > 0 else anox_val
            box.CompOMZ[i, j] = new_omz
            box.CompANOX[i, j] = new_anox

    box.vol[i] = Volume_new

    # Remineralize oxic part
    _fill_tray_from_tracer(i, box, tracer)
    AddedC = ncycle_remin_box(i, ocean, 1.0, oxTray, param)
    _write_tray_to_tracer(i, box, tracer)

    # Remineralize OMZ part
    _fill_tray_from_omz(i, box, omzV)
    AddedC_omz = ncycle_remin_box(i, ocean, omzV / dV if dV > 0 else 0.0, anoxTray, param)
    _write_tray_to_omz(i, box, omzV)

    # Remineralize + denitrify ANOX part
    _fill_tray_from_anox(i, box, anoxV)
    usedFract_anox = anoxV / dV if dV > 0 else 0.0
    k4 = ((box.CompANOX[i, _iO] - param.ResidualOxygen)
           / (box.CompOMZ[i, _iO] - param.ResidualOxygen)) if (box.CompOMZ[i, _iO] - param.ResidualOxygen) > 1e-30 else 1.0
    k4 = min(max(k4, 0.0), 1.0)
    # Aerobic remin fraction
    AddedC_anox = ncycle_remin_box(i, ocean, k4 * usedFract_anox, anoxTray, param)
    # Denitrification fraction
    # Pascal does usedFract := (1-k4)*usedFract/k4 where usedFract was already k4*(anoxV/dV).
    # So the result is (1-k4)*(anoxV/dV). We use the original usedFract_anox directly.
    usedFract_dntr = (1.0 - k4) * usedFract_anox
    AddedC_dntr, AddedP_dntr, AddedC13_dntr, AddedC14_dntr = _calc_added_C(i, ocean, usedFract_dntr, anoxTray)
    ncycle_denitrification(i, ocean, usedFract_dntr, anoxTray, AddedC_dntr, AddedP_dntr, AddedC13_dntr, AddedC14_dntr, param)
    _write_tray_to_anox(i, box, anoxV)


#::
#:: --- def _manager_behaviour3 ---
#:: OMZ behaviour 3 (pocket forms for the first time this year): initialize the sub-volumes from the
#:: oxic box, then run the same remineralization/denitrification as behaviour 2.
def _manager_behaviour3(i, ocean, param, olddV, oldVolume, oldomzV, dV, VolFraction):
    """Manager behaviour 3: OMZ forms for first time (no ANOX yet)."""
    tracer = ocean.tracer
    box = ocean.box
    

    Volume_new = box.completeVolume[i] - dV
    omzV = dV
    box.omzV[i] = omzV
    box.anoxV[i] = 0.0

    # Cross-section geometry
    depth_range = box.bottom[i] - box.top[i]
    if depth_range > 0:
        box.crossArea[i] = dV * 3.0 / (4.0 * depth_range * 0.5)
        r = (3.0 * dV / (4.0 * np.pi * depth_range * 0.5)) ** 0.5
        box.crossArea2[i] = np.pi * depth_range * r

    angle = 2.0 * np.pi * param.fluxangle / 360.0
    Flux = (np.cos(angle) * param.InternalFlux * SECPERYEAR * box.crossArea2[i]
            + np.sin(angle) * param.InternalFlux * SECPERYEAR * box.crossArea[i])
    Flux = min(Flux, omzV)
    box.Flux[i] = Flux

    k2 = ((((VolFraction * 100) - 100) ** 14 / 1.5e27) + 1.0) * param.orgfluxscale
    depth_range_i = box.bottom[i] - box.top[i]
    Area_i = box.completeVolume[i] / depth_range_i if depth_range_i > 0 else 1.0
    oxTray = 1.0 - k2 * box.crossArea[i] / max(Area_i, 1.0)
    anoxTray = k2 * box.crossArea[i] / max(Area_i, 1.0)
    if oxTray < 0:
        oxTray = 0.0
        anoxTray = 1.0

    if i >= 8:
        ncycle_sed_denitrification(i, ocean, oxTray, param)

    # Grow OMZ from oxic volume
    changedV = dV - olddV
    for j in range(10):
        comp_val = _get_tracer_by_idx(tracer, i, j)
        omz_val = box.CompOMZ[i, j]
        if changedV < 0:
            add_anox = changedV * omz_val * KGPERM
            add_ox = -changedV * omz_val * KGPERM
        else:
            add_anox = changedV * comp_val * KGPERM
            add_ox = -changedV * comp_val * KGPERM
        new_comp = (comp_val * oldVolume * KGPERM + add_ox) / (Volume_new * KGPERM) if Volume_new > 0 else comp_val
        new_omz = (omz_val * oldomzV * KGPERM + add_anox) / (omzV * KGPERM) if omzV > 0 else omz_val
        _set_tracer_by_idx(tracer, i, j, new_comp)
        box.CompOMZ[i, j] = new_omz

    box.vol[i] = Volume_new

    # Remineralize oxic part
    _fill_tray_from_tracer(i, box, tracer)
    ncycle_remin_box(i, ocean, 1.0, oxTray, param)
    _write_tray_to_tracer(i, box, tracer)

    # Remineralize OMZ part (no denitrification yet - no ANOX)
    _fill_tray_from_omz(i, box, omzV)
    ncycle_remin_box(i, ocean, 1.0, anoxTray, param)
    box.lostN[i] = 0.0
    _write_tray_to_omz(i, box, omzV)


#::
#:: --- def _fill_tray_from_omz ---
#:: Load the OMZ-shell sub-volume contents into the working tray.
def _fill_tray_from_omz(i, box, omzV):
    """Fill ComponentTray from CompOMZ."""
    V = omzV * KGPERM
    for j in range(10):
        box.ComponentTray[i, j] = box.CompOMZ[i, j] * V


#::
#:: --- def _write_tray_to_omz ---
#:: Write the working tray back to the OMZ-shell sub-volume.
def _write_tray_to_omz(i, box, omzV):
    """Write ComponentTray back to CompOMZ."""
    V = omzV * KGPERM
    if V <= 0:
        return
    for j in range(10):
        box.CompOMZ[i, j] = box.ComponentTray[i, j] / V


#::
#:: --- def _fill_tray_from_anox ---
#:: Load the anoxic-core sub-volume contents into the working tray.
def _fill_tray_from_anox(i, box, anoxV):
    """Fill ComponentTray from CompANOX."""
    V = anoxV * KGPERM
    for j in range(10):
        box.ComponentTray[i, j] = box.CompANOX[i, j] * V


#::
#:: --- def _write_tray_to_anox ---
#:: Write the working tray back to the anoxic-core sub-volume.
def _write_tray_to_anox(i, box, anoxV):
    """Write ComponentTray back to CompANOX."""
    V = anoxV * KGPERM
    if V <= 0:
        return
    for j in range(10):
        box.CompANOX[i, j] = box.ComponentTray[i, j] / V


#::
#:: --- def _calc_added_C ---
#:: Compute the organic C (and P, 13C, 14C) delivered to a box/compartment by the sinking rain --
#:: used to drive the denitrification stoichiometry in the anoxic core.
def _calc_added_C(i, ocean, usedFract, supplyTray, param=None):
    """Calculate total C added from organic rain to a box (for denitrification stoichiometry).
    Also returns P, C13, C14 for adding to ComponentTray (matching Pascal).
    """
    box = ocean.box
    TIMESTEP = 1.0
    AddedC = 0.0
    AddedP = 0.0
    AddedC13 = 0.0
    AddedC14 = 0.0
    for j in range(8):
        Prod_j = box.Production[j]
        if Prod_j <= 0:
            continue
        Ox_ij = ocean.RainOrg[i - 8, j] if i >= 8 else 0.0
        if Ox_ij <= 0:
            continue
        flux = usedFract * supplyTray * Prod_j * box.Area[j] * Ox_ij * TIMESTEP
        AddedC += flux * 106.0
        AddedP += flux * 1.0  # Redfield P = 1
        # C13 from organic rain
        d13Corg_j = ocean.rain.d13Corg[j]
        AddedC13 += flux * 106.0 * IsoConcPDB(d13Corg_j)
        # C14 from organic rain
        d14Corg_j = ocean.rain.d14Corg[j]
        AddedC14 += flux * 106.0 * (((d14Corg_j / 1000.0 + 1.0) * NBSC14RATIO) / (1.0 + (d14Corg_j / 1000.0 + 1.0) * NBSC14RATIO))

    # ---- v27: native seaweed carbon also drives denitrification stoichiometry where it lands.
    # No P is added here (seaweed P is cycled separately, conservatively) -- only C, so that
    # ncycle_denitrification's Redfield-based lostN = denitriparam * AddedComponentC applies to
    # the seaweed carbon too. Because seaweed's real C:N differs from Redfield, this leaves a
    # small (~few TgN/yr) flux-budget non-closure -- an accepted, documented approximation.
    if param is not None and getattr(param, 'nativeSeaweedCarbon', False):
        SWflux = getattr(box, 'seaweedCflux', None)
        if SWflux is not None and SWflux[i] != 0.0:
            AddedC += usedFract * supplyTray * SWflux[i]

    return AddedC, AddedP, AddedC13, AddedC14


#::
#:: --- def ncycle_init_tracers ---
#:: Initialize the nitrogen tracers (nitrate, 15N, 18O) and the OMZ sub-volume state to sensible
#:: starting values before an N-cycle spin-up.
def ncycle_init_tracers(ocean, param):
    """Initialize N isotope tracers from existing NO3 (tracer.N) and default delta values.
    Call this once at the start when Ncycle=True.
    """
    tracer = ocean.tracer
    box = ocean.box

    # Initialize N from P * 16 (should already be set)
    # Initialize N15 from default d15N = 5.0 permil
    # Initialize NO18 from default d18O = 1.0 permil
    default_d15N = 5.0
    default_d18O = 1.0
    default_d18O_O2 = 24.7

    for i in range(NB):
        if tracer.N[i] <= 0:
            tracer.N[i] = tracer.P[i] * 16.0  # Redfield N:P = 16, both in µmol/kg
        tracer.N15[i] = tracer.N[i] * IsoConcN(default_d15N)
        tracer.NO18[i] = tracer.N[i] * IsoConcPDB(default_d18O)

    # Initialize O2 from saturation for surface, realistic values for deep
    ox_exchange(tracer, box)
    # Initialize subsurface O2 to values close to expected steady state.
    # Starting from saturation (~312) causes transient N depletion because
    # sed denitrification outpaces fixation before OMZ can develop.
    # Targets from Pascal Sigman2009b: intO2~111, deepO2~163.
    # Per-box estimates based on circulation patterns:
    target_O2 = {
        8: 120.0,   # Int.Atl
        9: 100.0,   # Int.Ind
        10: 90.0,   # Int.SPac
        11: 60.0,   # Int.NPac (lowest O2 - OMZ region)
        12: 220.0,  # D.NCW (young, O2-rich deep water)
        13: 160.0,  # D.CDW
        14: 180.0,  # D.Atl
        15: 150.0,  # D.Ind
        16: 140.0,  # D.SPac
        17: 130.0,  # D.NPac
        # lower-thermocline boxes (22-box config): start near their upper partners
        18: 40.0, 19: 40.0, 20: 30.0, 21: 20.0,   # middle thermocline (ODZ seed)
        22: 130.0, 23: 110.0, 24: 100.0, 25: 80.0,  # lower thermocline
    }
    for i in INTERIOR_IDX:
        tracer.O2[i] = target_O2.get(i, 120.0)
        tracer.O218[i] = tracer.O2[i] * IsoConcPDB(default_d18O_O2)

    # Initialize H2Od18O from salinity
    for i in range(8):
        tracer.H2Od18O[i] = tracer.Sal[i] * 0.5333 - 18.5

    # Initialize completeVolume and vol
    for i in range(NB):
        box.completeVolume[i] = box.vol[i]
        box.dV[i] = 0.0
        box.omzV[i] = 0.0
        box.anoxV[i] = 0.0
        box.CompOMZ[i, :] = 0.0
        box.CompANOX[i, :] = 0.0

    # Initialize Production_d15Norg and Production_d18ONorg
    box.Production_d15Norg = np.zeros(8)
    box.Production_d18ONorg = np.zeros(8)


#::
#:: ############################################################################################
#:: ## THE ANNUAL TIME-STEP LOOP -- one pass per model year. Order: set forcing flags ->
#:: ## advect (circulation) -> production + N2 fixation -> CaCO3 dissolution -> OMZ manager
#:: ## (remineralization + denitrification) -> air-sea gas/O2 exchange -> external sources ->
#:: ## carbonate solve -> record diagnostics.
#:: ############################################################################################
#::
#:: --- def run_ex ---
#:: MAIN LOOP. For each of Nyears: (1) set intervention flags for the year (interventions active 2030-2100);
#:: (2) circ_advect; (3) prod + ncycle_production (production, nitrate strip, N2 fixation); (4) dissolve CaCO3;
#:: (5) ncycle_manager (interior remineralization + water-column & sediment denitrification via the OMZ pockets),
#:: or plain remin() if the N-cycle is off; (6) gas_ex + ox_exchange (surface re-equilibration with the air);
#:: (7) external sources (14C, volcano, river) and the final carbonate solve; advance the year and record diagnostics.
def run_ex(ocean, atmosphere, geosphere, param, Nyears, emissionsFlag):
    """Main time-stepping loop."""
    for t in range(Nyears):
        # Set up experimental flags
        SWflag = False
        SurfSWflag = False
        pulseCO2flag = False
        OAEflag = False
        IronFertflag = False
        ArtifUpwellflag = False
        SAZSWflag = False

        # Activate interventions during 2030-2100
        if 2030 <= param.year <= 2100:
            SWflag = True
            SurfSWflag = True
            pulseCO2flag = True
            OAEflag = True
            ArtifUpwellflag = True
            SAZSWflag = True

            # Copy gas transfer coefficients
            ocean.Ksurf2x.K0 = ocean.Ksurf.K0.copy()
            ocean.Ksurf2x.K1 = ocean.Ksurf.K1.copy()
            ocean.Ksurf2x.K2 = ocean.Ksurf.K2.copy()
            ocean.Ksurf2x.Kb = ocean.Ksurf.Kb.copy()
            ocean.Ksurf2x.Ks = ocean.Ksurf.Ks.copy()
            ocean.Ksurf2x.K0[6] = 5.0 * ocean.Ksurf.K0[6]

            if ocean.box.setP[6] > ocean.box.setIronFertP[6]:
                IronFertflag = True

        # Experiment flag processing
        if param.Exflag == 1:
            if t < 1000:
                CperYear = (param.Ex1 / 4.0) * 1e21 / 12.0
                AtoC = param.Ex2 / 10.0
                ocean.tracer.C[10] += CperYear * ocean.box.NtoC[10]
                ocean.tracer.Alk[10] += AtoC * CperYear * ocean.box.NtoC[10]
            if t % 50 == 0:
                row = t // 50
                param.ExOUT[row, 0] = atmosphere.ppm
                param.ExOUT[row, 1] = D14Ccalc(atmosphere.ppm, atmosphere.dn13, atmosphere.dn14)

        # DGL forcing
        if param.DGLFall.F4.value == 2:
            ocean.VolParams.scale = 40.0 * 100 / 1000
            change_vol(ocean.Schemes.DOtoMDviaAApLL, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams)

        if param.DGLFall.init_true == 1:
            update_dgl_forcing(param.DGLFall, param.year)
            if param.DGLFall.trigerID == 1:
                param.DGLFall.F1.value = 0
                param.DGLFall.F4.value = 0
                param.DGLFall.trigerID = 0

            ID = int(param.DGLFall.F1.value)
            if ID >= 0:
                if ID == 0:
                    change_circ(ocean.circulationM, 0)
                elif ID == 1:
                    change_circ(ocean.circulationM, 1)
                elif ID == 2:
                    change_circ(ocean.circulationM, 98)
                else:
                    print(f"XXX CIRC-WARNING: ID={ID}")
                    change_circ(ocean.circulationM, 0)
                init_circ(ocean.circulationM, ocean.box.vol, ocean.box.vol_inv)

            paz_mix(ocean.circulationM, ocean.box.vol, 3.0 + 17.0 * param.DGLFall.F2.value)
            ocean.box.setP[7] = 1.0 + 1.0 * param.DGLFall.F2.value
            param.PAZiceX = 0.5 - 0.5 * param.DGLFall.F2.value
            ocean.box.Area[7] = (1.0 - param.PAZiceX) * param.PAZarea
            ocean.box.setP[6] = 1.2 - (0.5 * (1.0 - param.DGLFall.F3.value)**2)

            ID = int(param.DGLFall.F4.value)
            if ID != 0:
                if ID == 1:
                    ocean.VolParams.scale = 40.0 * 100 / 1000 * 2 * param.DGLFall.F4.yrstep / 1000
                    change_vol(ocean.Schemes.DOtoMDviaAApLL, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams)
                elif ID == 2:
                    ocean.VolParams.scale = 40.0 * 100 / 1400 * 2 * param.DGLFall.F4.yrstep / 1400
                    change_vol(ocean.Schemes.DOtoMDviaAApLL, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams)
                elif ID == -1:
                    ocean.VolParams.scale = 40
                    change_vol(ocean.Schemes.MDtoDOviaNA, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams)

        # Recalculate K's if T*S changed significantly
        if abs(param.TxS - (np.sum(ocean.tracer.Temp) * np.sum(ocean.tracer.Sal))) >= 0.1:
            kcalc(ocean.Ksurf, ocean.tracer.Temp, ocean.tracer.Sal, ocean.box.top, ocean.box.bottom, ocean.SF)
            param.TxS = np.sum(ocean.tracer.Temp) * np.sum(ocean.tracer.Sal)

        # Seafloor pH and dissolution
        if (t % 50 == 0) and (param.OpenSystem == 1):
            sf_ph_calc(ocean.tracer.Sal, ocean.tracer.C, ocean.tracer.Alk, ocean.SF, param.CaX, param.DissolveX)
        elif param.OpenSystem == 0:
            ocean.SF.Fdiss[:] = 1.0

        # Artificial upwelling circulation modification
        if not ArtifUpwellflag:
            ocean.ArtifUpwellVdot[:] = 0.0

        if ArtifUpwellflag:
            ocean.circM_ArtifUpwell[:] = ocean.circM_Sv.copy()
            for box in range(4):
                deltaP = ocean.tracer.P[box + 8] - ocean.tracer.P[box]
                Vdot = (ocean.box.setArtifUpwellSW[box] / deltaP) * ocean.box.vol[box]
                Vdot = Vdot / (1e6 * SECPERYEAR)
                ocean.circM_ArtifUpwell[box, box + 8] += Vdot
                ocean.circM_ArtifUpwell[box + 8, box] += Vdot
                ocean.ArtifUpwellVdot[box] = Vdot
                ocean.ArtifUpwellVdot[box + 4] = Vdot
            init_circ(ocean.circM_ArtifUpwell, ocean.box.vol, ocean.box.vol_inv)

        # Advection
        if not ArtifUpwellflag:
            circ_advect(ocean.circulationM, ocean.tracer, Ncycle=param.Ncycle, box=ocean.box)
        else:
            circ_advect(ocean.circM_ArtifUpwell, ocean.tracer, Ncycle=param.Ncycle, box=ocean.box)

        # N-cycle: store pre-production N for computing Production rates
        if param.Ncycle:
            _preP = ocean.tracer.P.copy()

        # Biological production
        prod(ocean.tracer, ocean.box.setP, ocean.box.setProdP, ocean.box.setSi,
             ocean.box.setSW, ocean.box.setSurfSW, ocean.box.setOAE, ocean.box.setIronFertP,
             ocean.box.setArtifUpwellSW, ocean.box.setSAZSW,
             ocean.box.CaRatio, ocean.rain, ocean.rainSW, ocean.stoichSW,
             ocean.rainSurfSW, ocean.stoichSurfSW,
             ocean.rainArtifUpwellSW, ocean.stoichArtifUpwellSW,
             ocean.rainSAZSW, ocean.stoichSAZSW,
             param.alphaSi, ocean.box.CtoN, ocean.box.NtoC, ocean.box.ORGe,
             SWflag, SurfSWflag, OAEflag, IronFertflag, ArtifUpwellflag, SAZSWflag,
             Ncycle=param.Ncycle)

        # N-cycle: compute Production rates and do N isotope production
        if param.Ncycle:
            # Production rate = (preP - postP) * vol * KGPERM / Area / timestep
            # But rain.P (before CtoN scaling) = preP - postP (in mol/kg)
            # After prod(), rain.P has been scaled by CtoN, so:
            # actual mol P stripped = rain.P[i] * NtoC[i]  (undo CtoN scaling)
            # Production (mol P/m2/yr) = actual_stripped * vol * KGPERM / Area
            for sb in range(8):
                actual_P = (_preP[sb] - ocean.tracer.P[sb])
                if actual_P > 0 and ocean.box.Area[sb] > 0:
                    ocean.box.Production[sb] = actual_P * ocean.box.vol[sb] * KGPERM / ocean.box.Area[sb]
                else:
                    ocean.box.Production[sb] = 0.0
            ncycle_production(ocean.tracer, ocean.box, param, ocean.rain)
            assert ocean.tracer.N[:8].min() >= 0, f"Negative nitrate in box {ocean.tracer.N[:8].argmin()}"

        # Remineralization (skip when Ncycle - manager handles all remin)
        if not param.Ncycle:
            remin(ocean.tracer, ocean.RainOrg, ocean.RainSi,
              ocean.RainSW, ocean.RainSurfSW, ocean.RainArtifUpwellSW, ocean.RainSAZSW,
              ocean.rain, ocean.rainSW, ocean.stoichSW,
              ocean.rainSurfSW, ocean.stoichSurfSW,
              ocean.rainArtifUpwellSW, ocean.stoichArtifUpwellSW,
              ocean.rainSAZSW, ocean.stoichSAZSW,
              ocean.box.NtoC, SWflag, SurfSWflag, ArtifUpwellflag, SAZSWflag)

        # Dissolution
        dissolve(ocean.tracer, ocean.RainCC, ocean.rain.Ca, ocean.rain.d13Ccc, ocean.rain.d14Ccc, ocean.box.NtoC, ocean.SF.Fdiss)

        # N-cycle: OMZ Manager (handles N-specific remin, WC denitrification, sed denitrification)
        if param.Ncycle:
            ncycle_manager(ocean, param)
            # Mechanistic N-flux rebuild (plan Section 3): applied after the manager,
            # replacing legacy fixation + OMZ denitrification when toggled on.
            if getattr(param, "NfluxV2", False):
                import ncycle_nflux_v26 as _nfx
                _nfx.attach_defaults(param)
                ocean._nflux_v2_last = _nfx.nflux_v2_step(ocean, param)
            # Si dissolution is handled by standard remin() but that's skipped when Ncycle=True.
            # Si is independent of OMZ dynamics, so apply it directly here.
            addOrg = ocean.RainSi @ ocean.rain.Si
            for Box in INTERIOR_IDX:
                ocean.tracer.Si[Box] += addOrg[Box - 8] * ocean.box.NtoC[Box]
            addOrg = ocean.RainSi @ (ocean.rain.Si * ocean.rain.d30Si)
            for Box in INTERIOR_IDX:
                ocean.tracer.dc30[Box] += addOrg[Box - 8] * ocean.box.NtoC[Box]

        # Check for NaN
        if not (atmosphere.ppm > 0) and param.flag == 0:
            print(f"XXXDIC problem in year {param.year}")

        # CO2 emissions
        if pulseCO2flag:
            atmosphere.ppm += atmosphere.setCO2pulse
        if emissionsFlag == 1 and param.year >= 1750:
            atmosphere.ppm += 2.0 / 2.12

        # Gas exchange
        if not SAZSWflag:
            gas_ex(atmosphere, ocean.tracer.Sal[:8], ocean.tracer.Temp[:8], ocean.box.ORGe,
                   ocean.tracer.C, ocean.tracer.dc13, ocean.tracer.dc14,
                   ocean.tracer.Alk[:8], ocean.box.NtoC[:8], ocean.Ksurf, ocean.box.Area)
        else:
            gas_ex(atmosphere, ocean.tracer.Sal[:8], ocean.tracer.Temp[:8], ocean.box.ORGe,
                   ocean.tracer.C, ocean.tracer.dc13, ocean.tracer.dc14,
                   ocean.tracer.Alk[:8], ocean.box.NtoC[:8], ocean.Ksurf2x, ocean.box.Area)

        # N-cycle: O2 air-sea exchange (reset surface O2 to saturation)
        if param.Ncycle:
            ox_exchange(ocean.tracer, ocean.box)

        # Radiocarbon
        handle_14c(atmosphere, ocean.tracer.dc14, param, geosphere)

        # Volcanic and river input
        Appm_ref = [atmosphere.ppm]
        dn13_ref = [atmosphere.dn13]
        dn14_ref = [atmosphere.dn14]
        volcano(Appm_ref, dn13_ref, param.VolcX)
        atmosphere.ppm = Appm_ref[0]
        atmosphere.dn13 = dn13_ref[0]

        river(ocean.tracer.C, ocean.tracer.dc13, ocean.tracer.dc14, ocean.tracer.Alk,
              ocean.box.NtoC[:4], Appm_ref, dn13_ref, dn14_ref,
              param.WeathX, param.RivX, param.SetCO2)
        atmosphere.ppm = Appm_ref[0]
        atmosphere.dn13 = dn13_ref[0]
        atmosphere.dn14 = dn14_ref[0]

        # Carbonate chemistry and ventilation tracking
        final_c_solve(ocean.tracer.C, ocean.tracer.Alk, ocean.tracer.Sal,
                      ocean.Ksurf.K0, ocean.Ksurf.K1, ocean.Ksurf.K2, ocean.Ksurf.Kb, ocean.Ksurf.Ks,
                      param.CaX, ocean.box.Csolved)
        vent_track(ocean.circulationM, ocean.venttracer, atmosphere,
                   ocean.tracer.C[:8], ocean.tracer.dc13[:8], ocean.tracer.dc14[:8])

        # SEAWEED output recording
        if param.year >= -5999999:
            if param.year >= 1700 and param.year % 1 == 0:
                if ocean.OUTSW is not None and ocean.OUTSWrow < ocean.OUTSW.shape[0]:
                    row = ocean.OUTSWrow
                    ocean.OUTSW[row, :] = 0.0  # reset row
                    ocean.OUTSW[row, 0] = param.year
                    ocean.OUTSW[row, 1] = atmosphere.ppm

                    # SW export (PgC/yr)
                    for box in range(4):
                        ocean.OUTSW[row, 2] += ocean.rainSW.P[box] * 12e-21 * ocean.stoichSW[0]

                    # Low-lat productivity
                    for box in range(4):
                        ocean.OUTSW[row, 4] += 106.0 * ocean.rain.P[box] * 12e-21

                    # PO4 concentrations by depth tier
                    surf_vol_ll = sum(ocean.box.vol[i] for i in range(4))
                    surf_vol_hl = sum(ocean.box.vol[i] for i in range(4, 8))
                    int_vol = sum(ocean.box.vol[i] for i in INTERMEDIATE_IDX)
                    deep_vol = sum(ocean.box.vol[i] for i in DEEP_IDX)
                    tot_vol = sum(ocean.box.vol[i] for i in range(NB))

                    for box in range(4):
                        ocean.OUTSW[row, 5] += ocean.tracer.P[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)
                    for box in range(4, 8):
                        ocean.OUTSW[row, 6] += ocean.tracer.P[box] * ocean.box.CtoN[box] / (surf_vol_hl * 1024)
                    for box in INTERMEDIATE_IDX:
                        ocean.OUTSW[row, 7] += ocean.tracer.P[box] * ocean.box.CtoN[box] / (int_vol * 1024)
                    for box in DEEP_IDX:
                        ocean.OUTSW[row, 8] += ocean.tracer.P[box] * ocean.box.CtoN[box] / (deep_vol * 1024)

                    # pCO2 area-weighted
                    surfArea = np.sum(ocean.box.Area)
                    for box in range(8):
                        ocean.OUTSW[row, 9] += ocean.box.Csolved.pCO2[box] * ocean.box.Area[box] / surfArea

                    surfVol = sum(ocean.box.vol[i] for i in range(8))
                    for box in range(8):
                        ocean.OUTSW[row, 10] += ocean.box.Csolved.CO3[box] * ocean.box.CtoN[box] / (surfVol * 1024)
                        ocean.OUTSW[row, 11] += ocean.box.Csolved.H[box] * ocean.box.CtoN[box] / (surfVol * 1024)
                        ocean.OUTSW[row, 12] += ocean.box.Csolved.omega[box] * ocean.box.CtoN[box] / (surfVol * 1024)

                    for box in range(NB):
                        ocean.OUTSW[row, 13] += ocean.tracer.Alk[box] * ocean.box.CtoN[box] / (tot_vol * 1024)

                    # Col 14: low-lat H+ (vol-weighted)
                    for box in range(4):
                        ocean.OUTSW[row, 14] += ocean.box.Csolved.H[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)

                    # Col 15: low-lat Alk_non-carbonate = Alk - HCO3 - 2*CO3 + H
                    for box in range(4):
                        ocean.OUTSW[row, 15] += (ocean.tracer.Alk[box] - ocean.box.Csolved.HCO3[box] - 2*ocean.box.Csolved.CO3[box] + ocean.box.Csolved.H[box]) * ocean.box.CtoN[box] / (surf_vol_ll * 1024)

                    # Col 16: low-lat CO3
                    for box in range(4):
                        ocean.OUTSW[row, 16] += ocean.box.Csolved.CO3[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)

                    # Col 17: low-lat HCO3
                    for box in range(4):
                        ocean.OUTSW[row, 17] += ocean.box.Csolved.HCO3[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)

                    # Col 18: low-lat B(OH)3 = BOH4*H/(Kb+H)
                    for box in range(4):
                        ocean.OUTSW[row, 18] += (ocean.box.Csolved.BOH4[box] * ocean.box.Csolved.H[box] / (ocean.Ksurf.Kb[box] + ocean.box.Csolved.H[box])) * ocean.box.CtoN[box] / (surf_vol_ll * 1024)

                    # Col 19: low-lat DIC
                    for box in range(4):
                        ocean.OUTSW[row, 19] += ocean.tracer.C[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)

                    # Col 20: low-lat carbonate alkalinity (HCO3 + 2*CO3)
                    for box in range(4):
                        ocean.OUTSW[row, 20] += (ocean.box.Csolved.HCO3[box] + 2*ocean.box.Csolved.CO3[box]) * ocean.box.CtoN[box] / (surf_vol_ll * 1024)

                    # Col 21-24: Preg by depth tier
                    for box in range(4):
                        ocean.OUTSW[row, 21] += ocean.tracer.Preg[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)
                    for box in range(4, 8):
                        ocean.OUTSW[row, 22] += ocean.tracer.Preg[box] * ocean.box.CtoN[box] / (surf_vol_hl * 1024)
                    for box in INTERMEDIATE_IDX:
                        ocean.OUTSW[row, 23] += ocean.tracer.Preg[box] * ocean.box.CtoN[box] / (int_vol * 1024)
                    for box in DEEP_IDX:
                        ocean.OUTSW[row, 24] += ocean.tracer.Preg[box] * ocean.box.CtoN[box] / (deep_vol * 1024)

                    # Col 25-28: PregSW by depth tier
                    for box in range(4):
                        ocean.OUTSW[row, 25] += ocean.tracer.PregSW[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)
                    for box in range(4, 8):
                        ocean.OUTSW[row, 26] += ocean.tracer.PregSW[box] * ocean.box.CtoN[box] / (surf_vol_hl * 1024)
                    for box in INTERMEDIATE_IDX:
                        ocean.OUTSW[row, 27] += ocean.tracer.PregSW[box] * ocean.box.CtoN[box] / (int_vol * 1024)
                    for box in DEEP_IDX:
                        ocean.OUTSW[row, 28] += ocean.tracer.PregSW[box] * ocean.box.CtoN[box] / (deep_vol * 1024)

                    # Col 29-32: PregSurfSW by depth tier
                    for box in range(4):
                        ocean.OUTSW[row, 29] += ocean.tracer.PregSurfSW[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)
                    for box in range(4, 8):
                        ocean.OUTSW[row, 30] += ocean.tracer.PregSurfSW[box] * ocean.box.CtoN[box] / (surf_vol_hl * 1024)
                    for box in INTERMEDIATE_IDX:
                        ocean.OUTSW[row, 31] += ocean.tracer.PregSurfSW[box] * ocean.box.CtoN[box] / (int_vol * 1024)
                    for box in DEEP_IDX:
                        ocean.OUTSW[row, 32] += ocean.tracer.PregSurfSW[box] * ocean.box.CtoN[box] / (deep_vol * 1024)

                    # Col 33-36: PregArtifUpwellSW by depth tier
                    for box in range(4):
                        ocean.OUTSW[row, 33] += ocean.tracer.PregArtifUpwellSW[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)
                    for box in range(4, 8):
                        ocean.OUTSW[row, 34] += ocean.tracer.PregArtifUpwellSW[box] * ocean.box.CtoN[box] / (surf_vol_hl * 1024)
                    for box in INTERMEDIATE_IDX:
                        ocean.OUTSW[row, 35] += ocean.tracer.PregArtifUpwellSW[box] * ocean.box.CtoN[box] / (int_vol * 1024)
                    for box in DEEP_IDX:
                        ocean.OUTSW[row, 36] += ocean.tracer.PregArtifUpwellSW[box] * ocean.box.CtoN[box] / (deep_vol * 1024)

                    # Col 37: total ocean PO4
                    for box in range(NB):
                        ocean.OUTSW[row, 37] += ocean.tracer.P[box] * ocean.box.CtoN[box] / (tot_vol * 1024)

                    # Col 38: total ocean Preg
                    for box in range(NB):
                        ocean.OUTSW[row, 38] += ocean.tracer.Preg[box] * ocean.box.CtoN[box] / (tot_vol * 1024)

                    # Col 39: total ocean PregSW
                    for box in range(NB):
                        ocean.OUTSW[row, 39] += ocean.tracer.PregSW[box] * ocean.box.CtoN[box] / (tot_vol * 1024)

                    # Col 40: total ocean PregSurfSW
                    for box in range(NB):
                        ocean.OUTSW[row, 40] += ocean.tracer.PregSurfSW[box] * ocean.box.CtoN[box] / (tot_vol * 1024)

                    # Col 41: total ocean PregArtifUpwellSW
                    for box in range(NB):
                        ocean.OUTSW[row, 41] += ocean.tracer.PregArtifUpwellSW[box] * ocean.box.CtoN[box] / (tot_vol * 1024)

                    # Col 42-43: O2-equivalent from Preg intermediate and deep
                    ocean.OUTSW[row, 42] = ocean.OUTSW[row, 23] * (-170.0)
                    ocean.OUTSW[row, 43] = ocean.OUTSW[row, 24] * (-170.0)

                    # Col 44-45: O2-equivalent from PregSW intermediate and deep (scaled by SW stoichiometry)
                    ocean.OUTSW[row, 44] = ocean.OUTSW[row, 27] * (-170.0 / 106.0) * ocean.stoichSW[0]
                    ocean.OUTSW[row, 45] = ocean.OUTSW[row, 28] * (-170.0 / 106.0) * ocean.stoichSW[0]

                    # Col 46-47: O2-equivalent from PregSurfSW intermediate and deep
                    ocean.OUTSW[row, 46] = ocean.OUTSW[row, 31] * (-170.0 / 106.0) * ocean.stoichSurfSW[0]
                    ocean.OUTSW[row, 47] = ocean.OUTSW[row, 32] * (-170.0 / 106.0) * ocean.stoichSurfSW[0]

                    # Col 48-49: O2-equivalent from PregArtifUpwellSW intermediate and deep
                    ocean.OUTSW[row, 48] = ocean.OUTSW[row, 35] * (-170.0 / 106.0) * ocean.stoichArtifUpwellSW[0]
                    ocean.OUTSW[row, 49] = ocean.OUTSW[row, 36] * (-170.0 / 106.0) * ocean.stoichArtifUpwellSW[0]

                    # Col 50-57: ArtifUpwellVdot for boxes 0-7
                    for box in range(8):
                        ocean.OUTSW[row, 50 + box] = ocean.ArtifUpwellVdot[box]

                    # Col 58: high-lat productivity
                    for box in range(4, 8):
                        ocean.OUTSW[row, 58] += 106.0 * ocean.rain.P[box] * 12e-21

                    # Col 59: SAZ (box 6) productivity
                    ocean.OUTSW[row, 59] = 106.0 * ocean.rain.P[6] * 12e-21

                    # Col 60-63: PregSAZSW by depth tier
                    for box in range(4):
                        ocean.OUTSW[row, 60] += ocean.tracer.PregSAZSW[box] * ocean.box.CtoN[box] / (surf_vol_ll * 1024)
                    for box in range(4, 8):
                        ocean.OUTSW[row, 61] += ocean.tracer.PregSAZSW[box] * ocean.box.CtoN[box] / (surf_vol_hl * 1024)
                    for box in INTERMEDIATE_IDX:
                        ocean.OUTSW[row, 62] += ocean.tracer.PregSAZSW[box] * ocean.box.CtoN[box] / (int_vol * 1024)
                    for box in DEEP_IDX:
                        ocean.OUTSW[row, 63] += ocean.tracer.PregSAZSW[box] * ocean.box.CtoN[box] / (deep_vol * 1024)

                    # Col 64-65: O2-equivalent from PregSAZSW intermediate and deep
                    ocean.OUTSW[row, 64] = ocean.OUTSW[row, 62] * (-170.0 / 106.0) * ocean.stoichSAZSW[0]
                    ocean.OUTSW[row, 65] = ocean.OUTSW[row, 63] * (-170.0 / 106.0) * ocean.stoichSAZSW[0]

                    # Col 66: OAZ (box 5) productivity
                    ocean.OUTSW[row, 66] = 106.0 * ocean.rain.P[5] * 12e-21

                    # Col 67: PAZ (box 7) productivity
                    ocean.OUTSW[row, 67] = 106.0 * ocean.rain.P[7] * 12e-21

                    # Col 68: Boreal (box 4) productivity
                    ocean.OUTSW[row, 68] = 106.0 * ocean.rain.P[4] * 12e-21

                    ocean.OUTSWrow += 1

        param.year += 1

        # Progress output
        if t > 0 and t % 50000 == 0:
            print(f"  Year {param.year}: CO2={atmosphere.ppm:.2f} ppm")