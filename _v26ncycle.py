"""Faithful 26-box N-cycle setup: replicates the ncycle_diag (Sigman) configuration
on the 18-box state, THEN splits the thermocline into NLAYER layers, THEN normalizes."""
import os, numpy as np
import cyclops_v26 as C, _v26setup as S

def build(Fv=8.0, rain_split=None, circ_scale=1.0):
    E=C.Experiment(); p=E.param
    C.open_system(p,1); p.Exflag=0;p.flag=0;p.TxS=0;p.VolcX=0;p.WeathX=0;p.CaX=1;p.RivX=1;p.SetCO2=280.0
    p.ORGe=20.0;p.Spike=0;p.SpikeDelta=-50.0;p.DissolveX=1.0;p.alphaSi=0.9989;p.scalelength=2000.0
    p.year=-9999999;p.C14X=1.0;p.Q14.ExNo=0;p.Q14.OUT=np.zeros((501,41));p.Q14.OUT2=np.zeros((501,38))
    E.ocean.OUTSW=np.zeros((1301,69))
    C.input_model(E.ocean,E.atmosphere,E.geosphere)
    oc=E.ocean
    # ---- load NADW (modern) circulation into top-left 18x18 (raw Sv) ----
    base=C.get_base_path()
    data=np.loadtxt(os.path.join(base,'CIRCULATIONS','NADW_HainGBC2010_MYNADW2.txt')).reshape(18,18)
    oc.circulationM[:18,:18]=data*circ_scale   # scale overturning + interior exchange vigor
    # PAZ-CDW mixing
    oc.circulationM[7,13]=20.0*circ_scale; oc.circulationM[13,7]=20.0*circ_scale
    # ---- production targets (Sigman config) ----
    oc.box.setP[4]=0.55; oc.box.setP[5]=1.62; oc.box.setP[6]=1.22; oc.box.setP[7]=2.0
    p.PAZarea=oc.box.Area[7]; p.PAZiceX=0.0
    # ---- SAZ (box6) organic-rain routing on the 10-row RainOrg BEFORE expand ----
    a=0.01; dd=0.3; R=np.array(oc.RainOrg); R[:,6]=0.0
    R[5,6]=a
    R[0,6]=((1-a)*dd)/6; R[1,6]=2*((1-a)*dd)/6; R[2,6]=3*((1-a)*dd)/6
    R[6,6]=((1-a)*(1-dd))/6; R[7,6]=2*((1-a)*(1-dd))/6; R[8,6]=3*((1-a)*(1-dd))/6
    oc.RainOrg=R
    # ---- temps/salinity ----
    for i in range(4): oc.tracer.Temp[i]=18.5
    oc.tracer.Temp[4]=4.0; oc.tracer.Temp[5]=0.0
    oc.tracer.Sal[0]=36.0; oc.tracer.Sal[1]=35.0; oc.tracer.Sal[2]=35.0; oc.tracer.Sal[3]=35.0
    oc.tracer.Sal[4]=35.0; oc.tracer.Sal[5]=33.8; oc.tracer.Sal[6]=34.0; oc.tracer.Sal[7]=33.5
    for i in range(8): oc.tracer.H2Od18O[i]=oc.tracer.Sal[i]*0.5333-18.5
    # ---- box depths (surface/deep; thermocline set by expand) ----
    for i in range(4): oc.box.top[i]=0.0; oc.box.bottom[i]=100.0
    for i in range(4,8): oc.box.top[i]=0.0; oc.box.bottom[i]=112.0
    for i in range(8,12): oc.box.top[i]=100.0; oc.box.bottom[i]=1500.0
    oc.box.top[12]=100.0; oc.box.bottom[12]=3700.0
    oc.box.top[13]=100.0; oc.box.bottom[13]=3700.0
    for i in range(14,18): oc.box.top[i]=1500.0; oc.box.bottom[i]=3700.0
    # ---- now split thermocline + normalize ----
    S.expand_thermocline(oc, Fv_Sv=Fv, rain_split=rain_split)
    C.init_circ(oc.circulationM, oc.box.vol, oc.box.vol_inv)
    C.init_vol_schemes(oc.Schemes); oc.VolParams.scale=1.0
    return E
