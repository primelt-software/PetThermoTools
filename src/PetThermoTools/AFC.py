import numpy as np
import pandas as pd
from PetThermoTools.GenFuncs import *
from PetThermoTools.Plotting import *
from PetThermoTools.MELTS import *
import multiprocessing
from multiprocessing import Queue
from multiprocessing import Process
import time
import sys
from tqdm.notebook import tqdm, trange

# ------------------------------------------------------------------------------------
# --- Helper functions ---------------------------------------------------------------
# ------------------------------------------------------------------------------------

def comp_fix(Model=None, comp=None, Fe3Fet_Liq=None, H2O_Liq=None, CO2_Liq=None):
    """
    Normalize a bulk composition dict to *_Liq keys and Fe3Fet_Liq fraction.
    """
    c = comp.copy()
    # Ensure *_Liq suffix
    if not any(k.endswith("_Liq") for k in c.keys()):
        new = {}
        for k, v in c.items():
            if k not in ["Fe3Fet_Liq", "H2O_Liq", "CO2_Liq"]:
                new[f"{k}_Liq"] = v
            else:
                new[k] = v
        c = new
    # Defaults
    if Fe3Fet_Liq is not None:
        c["Fe3Fet_Liq"] = Fe3Fet_Liq
    elif "Fe3Fet_Liq" not in c:
        c["Fe3Fet_Liq"] = 0.15
    if H2O_Liq is not None:
        c["H2O_Liq"] = H2O_Liq
    elif "H2O_Liq" not in c:
        c["H2O_Liq"] = 0.0
    if CO2_Liq is not None:
        c["CO2_Liq"] = CO2_Liq
    elif "CO2_Liq" not in c:
        c["CO2_Liq"] = 0.0
    return c


def dict_to_bulk(comp):
    """
    Convert *_Liq dict to 19-element bulk vector for MELTS.
    """
    fe2o3 = comp['Fe3Fet_Liq']*((159.69/2)/71.844)*comp['FeOt_Liq'] if 'FeOt_Liq' in comp else 0.0
    feo   = (1 - comp['Fe3Fet_Liq'])*comp['FeOt_Liq'] if 'FeOt_Liq' in comp else 0.0
    bulk = [
        comp.get('SiO2_Liq',0), comp.get('TiO2_Liq',0), comp.get('Al2O3_Liq',0),
        fe2o3, comp.get('Cr2O3_Liq',0), feo, comp.get('MnO_Liq',0),
        comp.get('MgO_Liq',0), 0, 0, comp.get('CaO_Liq',0),
        comp.get('Na2O_Liq',0), comp.get('K2O_Liq',0), comp.get('P2O5_Liq',0),
        comp.get('H2O_Liq',0), comp.get('CO2_Liq',0), 0, 0, 0
    ]
    total = sum(bulk)
    return list(100*np.array(bulk)/total) if total > 0 else bulk


def mix_liquids(compA, mA, compB, mB):
    """
    Mass-weighted mixing of two liquids (in *_Liq dict format).
    """
    keys = [k for k in compA if k.endswith('_Liq')]
    total = max(mA + mB, 1e-12)
    mixed = {k: (mA*compA.get(k,0.0) + mB*compB.get(k,0.0))/total for k in keys}
    if 'Fe3Fet_Liq' in compA:
        mixed['Fe3Fet_Liq'] = (mA*compA['Fe3Fet_Liq'] + mB*compB['Fe3Fet_Liq'])/total
    return mixed


def _specific_h(Model, comp, T_C, P_bar):
    """
    Compute bulk specific enthalpy (J/kg) for given composition at T,P.
    """
    m = MELTSdynamic(1 if Model is None else {"MELTSv1.0.2":1,"pMELTS":2,"MELTSv1.1.0":3,"MELTSv1.2.0":4}.get(Model,1))
    e = m.engine
    e.setBulkComposition(dict_to_bulk(comp))
    e.temperature = T_C
    e.pressure = P_bar
    e.calcEquilibriumState(1, 0)
    H = float(e.getProperty("h","bulk"))
    M = float(e.getProperty("mass","bulk"))
    return H / max(M, 1e-12)

# ------------------------------------------------------------------------------------
# --- Main AFC Function --------------------------------------------------------------
# ------------------------------------------------------------------------------------

def energy_balance_AFC(
    Model="MELTSv1.0.2",
    bulk_magma=None,
    bulk_assimilant=None,
    P_bar=None, P_path_bar=None, P_start_bar=None, P_end_bar=None, dp_bar=None,
    T_path_C=None, T_start_C=None, T_end_C=None, dt_C=None,
    T_wall_C=700.0,
    find_liquidus=True,
    Frac_solid=True, Frac_fluid=False,
    fO2_buffer=None, fO2_offset=None,
    Suppress=['rutile','tridymite'], Suppress_except=False,
    Crystallinity_limit=None,
    fluid_sat=False
):
    """
    Energy-balance Assimilation–Fractional Crystallization (AFC).

    Parameters
    ----------
    Model : str
        MELTS model ("MELTSv1.0.2", "pMELTS", etc.)
    bulk_magma : dict
        Starting liquid composition.
    bulk_assimilant : dict
        Wallrock bulk composition (used to compute enthalpy of fusion and melt mix).
    P_bar, P_path_bar : float or array
        Pressure conditions.
    T_path_C, T_start_C, T_end_C, dt_C : float or array
        Temperature path controls.
    T_wall_C : float
        Wallrock initial temperature.
    """
    # --- Temperature and pressure path normalization ---
    if T_path_C is not None:
        T = np.atleast_1d(np.array(T_path_C, dtype=float))
    else:
        if T_start_C is None or (T_end_C is None and dt_C is None):
            raise ValueError("Provide T_path_C or (T_start_C, T_end_C, dt_C).")
        if T_end_C is not None and dt_C is not None:
            nT = 1 + int(round((T_start_C - T_end_C) / dt_C))
            T = np.linspace(T_start_C, T_end_C, nT)
        else:
            T = np.atleast_1d(np.array([T_start_C], dtype=float))

    if P_path_bar is not None:
        P = np.atleast_1d(np.array(P_path_bar, dtype=float))
    elif P_bar is not None:
        P = np.atleast_1d(np.array([P_bar], dtype=float))
    elif P_start_bar is not None and (P_end_bar is not None) and (dp_bar is not None):
        nP = 1 + int(round((P_start_bar - P_end_bar) / dp_bar))
        P = np.linspace(P_start_bar, P_end_bar, nP)
    else:
        raise ValueError("Provide P_bar or P_path_bar or (P_start_bar, P_end_bar, dp_bar).")

    if P.size == 1 and T.size > 1:
        P = np.repeat(P[0], T.size)
    elif T.size == 1 and P.size > 1:
        T = np.repeat(T[0], P.size)
    elif P.size != T.size:
        raise ValueError("Length of P and T vectors must match (or one must be scalar).")

    T = np.asarray(T, dtype=float).ravel()
    P = np.asarray(P, dtype=float).ravel()

    # --- Initialize compositions ---
    magma = comp_fix(Model, bulk_magma)
    wall  = comp_fix(Model, bulk_assimilant)

    # --- Initialize MELTS engine for magma ---
    model_map = {"MELTSv1.0.2":1,"pMELTS":2,"MELTSv1.1.0":3,"MELTSv1.2.0":4}
    melts = MELTSdynamic(model_map.get(Model,1))
    eng = melts.engine
    eng.setBulkComposition(dict_to_bulk(magma))
    eng.setSystemProperties("Mode", "Fractionate Solids" if Frac_solid else "Fractionate None")
    if Frac_fluid:
        eng.setSystemProperties("Mode", "Fractionate Fluids")

    # --- Data structures ---
    n = len(T)
    Results = {
        'Conditions': pd.DataFrame(np.zeros((n, 12)),
            columns=['T','P','h','mass','liq_mass','cryst_mass','assim_mass','r','H_release','liq_SiO2','liq_MgO','liq_H2O'])
    }

    # --- Initial enthalpies ---
    H_prev = None
    M_liq_prev = None

    for i in range(n):
        eng.temperature = float(T[i])
        eng.pressure = float(P[i])
        eng.calcEquilibriumState(1, 1)

        H_now = float(eng.getProperty('h','bulk'))
        M_liq = float(eng.getProperty('mass','liquid1'))
        liq_comp = {ox: eng.getProperty('dispComposition','liquid1',ox)
                    for ox in ['SiO2','MgO','H2O','FeO','Fe2O3','CaO','Na2O','K2O']}

        Mc = 0.0 if M_liq_prev is None else max(0.0, M_liq_prev - M_liq)
        dH_release = 0.0 if H_prev is None else max(0.0, H_prev - H_now)

        # --- Assimilation mass via energy balance ---
        if dH_release > 0.0:
            try:
                h_wall_Ta = _specific_h(Model, wall, T_wall_C, P[i])
                h_wall_Tm = _specific_h(Model, wall, T[i], P[i])
                q_per_mass = max(h_wall_Tm - h_wall_Ta, 1e-6)
                if not np.isfinite(q_per_mass) or q_per_mass <= 0.0:
                    q_per_mass = 3.5e5  # fallback J/kg
            except Exception:
                q_per_mass = 3.5e5
            Ma = dH_release / q_per_mass
        else:
            Ma = 0.0

        r_eff = Ma / max(Mc, 1e-12)

        # --- Record step ---
        Results['Conditions'].loc[i] = [
            T[i], P[i], H_now, eng.getProperty('mass','bulk'),
            M_liq, Mc, Ma, r_eff, dH_release,
            liq_comp['SiO2'], liq_comp['MgO'], liq_comp['H2O']
        ]

        # --- Mix assimilant melt into magma ---
        if Ma > 0.0:
            magma_liq = {f"{k}_Liq": v for k,v in liq_comp.items()}
            mixed = mix_liquids(magma_liq, M_liq, wall, Ma)
            new_bulk = dict_to_bulk(mixed)
            melts = melts.addNodeAfter()
            eng = melts.engine
            eng.setBulkComposition(new_bulk)
            eng.setSystemProperties("Mode", "Fractionate Solids" if Frac_solid else "Fractionate None")
            if Frac_fluid:
                eng.setSystemProperties("Mode", "Fractionate Fluids")
            eng.temperature = float(T[i])
            eng.pressure = float(P[i])
            eng.calcEquilibriumState(1,1)

        H_prev = H_now
        M_liq_prev = M_liq

    return Results

