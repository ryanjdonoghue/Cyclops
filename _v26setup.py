"""Generalized N-layer thermocline builder for cyclops_v26.
Splits each low-lat intermediate column (8,9,10,11) into NLAYER stacked layers:
surface ventilates the TOP layer, deep ventilates the BOTTOM layer, and the MIDDLE
layer(s) are ventilated only by vertical exchange -> where the O2 minimum/ODZ forms."""
import numpy as np
import cyclops_v26 as C

SURF=set(range(8)); DEEP=set(range(12,18)); INT0=set([8,9,10,11])

def expand_thermocline(ocean, fvol=None, Fv_Sv=8.0, rain_split=None):
    NL=C.NLAYER; cols=C.THERM_COLUMN  # cols[c] = [top..bottom] box indices
    if fvol is None:
        # depth-proportional layer thicknesses across 100-1500 m (top thin, then thicker)
        if NL==1: th=[1400.]
        elif NL==2: th=[500.,900.]
        elif NL==3: th=[300.,400.,700.]
        else:
            # thin upper layers, thicker lower (top-weighted resolution near the ODZ)
            th=[200.,200.,250.,300.,450.][:NL] if NL==5 else [1400./NL]*NL
        s=sum(th); fvol=[t/s for t in th]
    if rain_split is None:
        if NL>=5: base=[0.40,0.25,0.18,0.10,0.07][:NL]
        elif NL==4: base=[0.45,0.28,0.17,0.10]
        else: base=[0.5,0.3,0.2][:NL]
        rain_split=[r/sum(base) for r in base]
    box=ocean.box; tr=ocean.tracer
    depths=[100.]; 
    for k in range(NL): depths.append(depths[-1]+(1400.*fvol[k]))
    # ---- geometry: split each column's volume across layers; copy tracer state ----
    intensive=['P','C','dc13','Alk','N','Sal','Temp','Si','dc30','dc14','O2','N15','NO18','O218','H2Od18O',
               'Preg','PregSW','PregSurfSW','PregArtifUpwellSW','PregSAZSW','Alkreg']
    for c in range(4):
        L=cols[c]; m=L[0]; V=box.vol[m]
        for k,b in enumerate(L):
            box.vol[b]=fvol[k]*V
            box.top[b]=depths[k]; box.bottom[b]=depths[k+1]
            box.completeVolume[b]=box.vol[b]
            box.CtoN[b]=box.vol[b]*1024.0; box.NtoC[b]=1.0/(box.vol[b]*1024.0)
            if k>0:
                for arr in intensive:
                    a=getattr(tr,arr,None)
                    if a is not None and len(a)>b: a[b]=a[m]
    # ---- circulation: rebuild raw Sv (NB x NB) ----
    raw=np.array(ocean.circulationM); new=np.zeros((C.NB,C.NB))
    def colof(i): return i-8
    def layerbox(i,k): return cols[colof(i)][k]
    for i in range(18):
        for j in range(18):
            v=raw[i,j]
            if abs(v)<1e-12 or i==j: continue
            ri,ci=i in INT0, j in INT0
            if ci and ri:                 # lateral intermediate<->intermediate: split by layer vol
                for k in range(NL): new[cols[colof(i)][k], cols[colof(j)][k]]+=fvol[k]*v
            elif ri and (j in SURF):      # surface -> top layer
                new[i,j]+=v
            elif ri and (j in DEEP):      # deep -> bottom layer
                new[cols[colof(i)][-1], j]+=v
            elif ci and (i in SURF):      # top layer -> surface
                new[i,j]+=v
            elif ci and (i in DEEP):      # bottom layer -> deep
                new[i, cols[colof(j)][-1]]+=v
            else:
                new[i,j]+=v
    # vertical chain per column with through-flow + mixing (guarantees per-layer balance)
    for c in range(4):
        L=cols[c]
        ext_net=[new[b,:].sum()-new[:,b].sum() for b in L]   # external imbalance per layer
        cum=0.0
        for k in range(NL-1):
            cum+=ext_net[k]; Dk=cum    # net downward flux across interface k
            up=L[k]; lo=L[k+1]
            if Dk>=0: new[lo,up]+=Fv_Sv+Dk; new[up,lo]+=Fv_Sv
            else:     new[lo,up]+=Fv_Sv;     new[up,lo]+=Fv_Sv-Dk
    ocean.circulationM[:]=new; ocean.circM_Sv[:]=new
    # ---- rain matrices to (NB-8, 8) ----
    nint=C.NB-8
    def expand_rain(R):
        Rn=np.zeros((nint,8))
        Rn[0:4,:]=rain_split[0]*R[0:4,:]     # top therm (boxes 8-11) -> rows 0-3
        Rn[4:10,:]=R[4:10,:]                  # deep (12-17) -> rows 4-9 unchanged
        for k in range(1,NL):                 # middle/lower therm layers -> appended rows
            r0=(10+(k-1)*4)
            Rn[r0:r0+4,:]=rain_split[k]*R[0:4,:]
        return Rn
    ocean.RainOrg=expand_rain(np.array(ocean.RainOrg))
    ocean.RainCC=expand_rain(np.array(ocean.RainCC))
    ocean.RainSi=expand_rain(np.array(ocean.RainSi))
    for nm in ['RainSW','RainSurfSW','RainArtifUpwellSW','RainSAZSW']:
        R=getattr(ocean,nm,None)
        if R is not None: setattr(ocean,nm,expand_rain(np.array(R)))
    # seafloor depth levels: give each new layer box its column-top box's hypsometry
    try:
        from run_experiments import init_seafloor_depth_levels
        init_seafloor_depth_levels(box)  # fills boxes 0-17
        for c in range(4):
            L=cols[c]; top=L[0]
            for b in L[1:]:
                box.SFdepth_levels[b,:]=box.SFdepth_levels[top,:]
                box.SFfractarea_levels[b,:]=box.SFfractarea_levels[top,:]
    except Exception as _e:
        pass
    return ocean

def check_balance(ocean):
    raw=np.array(ocean.circM_Sv); bad=[]
    for i in range(C.NB):
        d=raw[i,:].sum()-raw[:,i].sum()
        if abs(d)>1e-6: bad.append((i,round(d,3)))
    return bad

def setup26(Fv=8.0):
    E=C.Experiment(); p=E.param
    C.open_system(p,1); p.Exflag=0;p.flag=0;p.TxS=0;p.VolcX=0;p.WeathX=0;p.CaX=1;p.RivX=1;p.SetCO2=280.0
    p.ORGe=20.0;p.Spike=0;p.SpikeDelta=-50.0;p.DissolveX=1.0;p.alphaSi=0.9989
    p.scalelength=2000.0;p.year=-9999999;p.C14X=1.0;p.Q14.ExNo=0
    p.Q14.OUT=np.zeros((501,41));p.Q14.OUT2=np.zeros((501,38)); E.ocean.OUTSW=np.zeros((1301,69))
    C.input_model(E.ocean,E.atmosphere,E.geosphere)
    expand_thermocline(E.ocean,Fv_Sv=Fv)
    C.init_circ(E.ocean.circulationM,E.ocean.box.vol,E.ocean.box.vol_inv)
    C.init_vol_schemes(E.ocean.Schemes); E.ocean.VolParams.scale=1.0; E.ocean.tracer.Temp[:]=5.0
    return E
