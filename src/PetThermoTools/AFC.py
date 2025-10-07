import numpy as np
import pandas as pd
from PetThermoTools.MELTS import path_MELTS, equilibrate_MELTS

def _normalize_bulk_from_liqdict(liq_dict):
    """
    Take a dict of oxide wt% for the liquid and return a MELTS 'bulk' list
    in the order used throughout PetThermoTools.MELTS.* helpers.
    Missing oxides default to 0.0.
    """
    # Ensure keys exist; default 0.0 if missing
    get = lambda k: float(liq_dict.get(k, 0.0))

    # Build the 19-length MELTS bulk array (same order as your code)
    # [SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, 0.0, 0.0, CaO, Na2O, K2O, P2O5, H2O, CO2, 0.0, 0.0, 0.0]
    bulk = [
        get('SiO2'), get('TiO2'), get('Al2O3'),
        get('Fe2O3'), get('Cr2O3'), get('FeO'),
        get('MnO'), get('MgO'), 0.0, 0.0,
        get('CaO'), get('Na2O'), get('K2O'),
        get('P2O5'), get('H2O'), get('CO2'),
        0.0, 0.0, 0.0
    ]
    # Normalize to sum = 100 (MELTS-friendly)
    s = np.sum(bulk)
    if s == 0:
        return bulk
    return list(100.0 * np.array(bulk) / s)

def _liqdict_from_engine(melts_engine):
    """
    Read the current liquid composition from an initialized MELTS engine
    to a plain dict (oxide wt%).
    """
    ox = ['SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','H2O','CO2']
    out = {}
    for k in ox:
        try:
            out[k] = float(melts_engine.getProperty('dispComposition','liquid1',k))
        except:
            out[k] = 0.0
    return out

def _mix_two_liquids_wt(liqA_dict, massA, liqB_dict, massB):
    """
    Weighted-average mixing of two liquid compositions (wt% basis).
    Returns dict of wt% normalized.
    """
    ox = ['SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','H2O','CO2']
    total_mass = float(massA) + float(massB)
    if total_mass <= 0:
        return {k:0.0 for k in ox}

    tmp = {}
    for k in ox:
        a = float(liqA_dict.get(k,0.0))
        b = float(liqB_dict.get(k,0.0))
        tmp[k] = (a*massA + b*massB)/total_mass

    # Already normalized as wt% by construction; still guard tiny drift:
    s = sum(tmp.values())
    if s > 0:
        for k in ox:
            tmp[k] = 100.0*tmp[k]/s
    return tmp

def _get_bulk_enthalpy_and_mass(Results_conditions):
    """
    Given Results['Conditions'] DataFrame with 'h' and 'mass', return H = h * mass.
    If either missing, return None.
    """
    if 'h' not in Results_conditions.columns or 'mass' not in Results_conditions.columns:
        return None
    # Use the last row (current step) total enthalpy-like value
    h = float(Results_conditions['h'].iloc[-1])
    m = float(Results_conditions['mass'].iloc[-1])
    return h * m

def _equilibrate_assimilant(Model, P_bar, T_C, bulk_assim_dict, fO2_buffer=None, fO2_offset=None, Suppress=['rutile','tridymite']):
    """
    Run a single-point MELTS equilibration for the assimilant at (P,T).
    Returns:
      - liq_wt% dict (may be all zeros if no liquid),
      - liq_mass (mass of liquid1 phase),
      - bulk_h (bulk specific enthalpy),
      - bulk_mass (bulk mass)
    """
    # Build a composition dict with "…_Liq" suffixes expected by equilibrate_MELTS
    # (we only need those the function reads; others can be zero)
    def conv(key_simple):
        return bulk_assim_dict.get(key_simple, 0.0)

    # We need Fe3Fet_Liq and FeOt_Liq for the converter in equilibrate_MELTS.
    # If user gave FeO & Fe2O3 in wt%, convert to FeOt and Fe3Fet.
    FeO = conv('FeO'); Fe2O3 = conv('Fe2O3')
    FeOt = FeO + 71.844/(159.69/2.0)*Fe2O3
    Fe3Fet = 0.0 if (FeOt<=0) else (71.844/(159.69/2.0)*Fe2O3)/FeOt

    compL = {
        'SiO2_Liq': conv('SiO2'), 'TiO2_Liq': conv('TiO2'), 'Al2O3_Liq': conv('Al2O3'),
        'Fe3Fet_Liq': Fe3Fet, 'FeOt_Liq': FeOt,
        'Cr2O3_Liq': conv('Cr2O3'), 'MnO_Liq': conv('MnO'), 'MgO_Liq': conv('MgO'),
        'CaO_Liq': conv('CaO'), 'Na2O_Liq': conv('Na2O'), 'K2O_Liq': conv('K2O'),
        'P2O5_Liq': conv('P2O5'), 'H2O_Liq': conv('H2O'), 'CO2_Liq': conv('CO2')
    }

    try:
        Res, _aff = equilibrate_MELTS(Model=Model, P_bar=P_bar, T_C=T_C, comp=compL,
                                      fO2_buffer=fO2_buffer, fO2_offset=fO2_offset,
                                      Suppress=Suppress)
    except:
        return ({k:0.0 for k in ['SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','H2O','CO2']},
                0.0, np.nan, np.nan)

    # Collect assimilant liquid composition and masses
    if 'liquid1' in Res and 'liquid1_prop' in Res:
        liq_dict = {k: float(Res['liquid1'].loc[0,k]) for k in Res['liquid1'].columns}
        liq_mass = float(Res['liquid1_prop'].loc[0,'mass'])
    else:
        liq_dict = {k:0.0 for k in ['SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','H2O','CO2']}
        liq_mass = 0.0

    # Bulk enthalpy-like and mass
    if 'Conditions' in Res and 'h' in Res['Conditions'].columns:
        bulk_h = float(Res['Conditions'].loc[0,'h'])
    else:
        bulk_h = np.nan

    if 'Conditions' in Res and 'mass' in Res['Conditions'].columns:
        bulk_mass = float(Res['Conditions'].loc[0,'mass'])
    else:
        bulk_mass = np.nan

    return (liq_dict, liq_mass, bulk_h, bulk_mass)

def energy_balance_AFC(
    *,
    Model=None,
    bulk_magma=None,
    bulk_assimilant=None,
    P_bar=None,
    P_path_bar=None,
    P_start_bar=None,
    P_end_bar=None,
    dp_bar=None,
    T_path_C=None,
    T_start_C=None,
    T_end_C=None,
    dt_C=None,
    T_wall_C=700.0,
    find_liquidus=None,
    Frac_solid=True,
    Frac_fluid=None,
    fO2_buffer=None,
    fO2_offset=None,
    Suppress=['rutile','tridymite'],
    Suppress_except=False,
    Crystallinity_limit=None,
    fluid_sat=None
):
    """
    Energy-balance AFC along a T–P path using MELTS.

    At each step i -> i+1:
        1) Crystallize the magma (typically with 'Fractionate Solids' on) using MELTS at (P[i], T[i]).
        2) Compute ΔH_system = (h*mass)_{i+1} - (h*mass)_i (from Results['Conditions']).
           The released energy is Q_release = -ΔH_system (if ΔH_system < 0).
        3) Compute energy required to bring wallrock from T_wall_C to T[i]:
                q_per_mass = (h_assim(T[i]) - h_assim(T_wall_C))
           using single-point MELTS equilibrations of the assimilant at P[i] and those T.
        4) The mass of wallrock that can be assimilated is:
                ΔM_assim_in = max(0, Q_release) / q_per_mass
           Only the *melt* part of that equilibrated wallrock at T[i] mixes into the magma.
        5) Mix current magma liquid with assimilant melt (mass-weighted wt%).
        6) Use the mixed liquid as the bulk for the next MELTS node (so that AFC continues).

    Notes
    -----
    - This routine mirrors your `path_MELTS` stepping logic (isobaric or polybaric) but injects an
      assimilation event after each equilibrium step based on enthalpy balance.
    - It uses `equilibrate_MELTS` for the assimilant at the relevant T (and T_wall_C) to avoid any
      guesses for cp or latent heats — we simply use Δ(h*mass) from MELTS as the energy metric.
    - By default, solids are fractionated (Frac_solid=True) which is standard for AFC.

    Returns
    -------
    Results : dict
        Same structure as `path_MELTS`, but with:
          - an extra key 'AFC' holding a DataFrame with columns:
                ['step','P_bar','T_C','Q_release','q_assim_per_mass','M_assim_total','M_assim_melt']
          - the liquid composition history already reflects assimilation each step.
    """
    if bulk_magma is None or bulk_assimilant is None:
        raise ValueError("Please provide both bulk_magma and bulk_assimilant as dicts of oxide wt%.")

    # Build a P and T path just like in path_MELTS
    # (copying the same path creation logic, simplified)
    if P_bar is not None and P_path_bar is None:
        P_path_bar = P_bar
    if T_path_C is None and T_start_C is None and find_liquidus is None:
        raise ValueError("Provide either T_path_C or T_start_C (or set find_liquidus=True).")

    # Linear paths if arrays not given
    if T_path_C is None:
        if T_end_C is None or dt_C is None:
            T = np.array([T_start_C], dtype=float)
        else:
            nT = 1 + int(round((T_start_C - T_end_C)/dt_C))
            T = np.linspace(T_start_C, T_end_C, nT)
    else:
        T = np.array(T_path_C, dtype=float)

    if P_path_bar is None:
        if P_end_bar is None or dp_bar is None:
            P = np.array([P_start_bar], dtype=float)
        else:
            nP = 1 + int(round((P_start_bar - P_end_bar)/dp_bar))
            P = np.linspace(P_start_bar, P_end_bar, nP)
    else:
        P = np.array(P_path_bar, dtype=float)

    # Make lengths consistent with path_MELTS behavior
    if T.size != P.size:
        if T.size == 1:
            T = np.repeat(T[0], P.size)
        elif P.size == 1:
            P = np.repeat(P[0], T.size)
        else:
            raise ValueError("Length of P and T vectors must match (or one must be scalar).")

    # ====== AFC iteration driver ======
    # We will emulate path_MELTS stepping, but after each step we compute assimilation
    # and reset the bulk for the next node to the mixed liquid.

    # 1) Initialize with a single-step path_MELTS call at the first (P,T) to create containers
    base_kwargs = dict(
        Model=Model,
        comp= bulk_magma,                 # dict with ..._Liq keys handled inside path_MELTS
        Frac_solid= True if Frac_solid else None,
        Frac_fluid= True if Frac_fluid else None,
        T_path_C= np.array([float(T[0])]),
        P_path_bar= np.array([float(P[0])]),
        T_start_C= float(T[0]),
        P_start_bar= float(P[0]),
        find_liquidus= find_liquidus,
        fO2_buffer= fO2_buffer,
        fO2_offset= fO2_offset,
        fluid_sat= fluid_sat,
        Crystallinity_limit= Crystallinity_limit,
        Suppress= Suppress,
        Suppress_except= Suppress_except
    )

    Results = path_MELTS(**base_kwargs)
    if not Results or 'Conditions' not in Results:
        return Results

    # AFC bookkeeping
    AFC_log = {
        'step': [], 'P_bar': [], 'T_C': [],
        'Q_release': [],              # positive = energy available (J or consistent MELTS units)
        'q_assim_per_mass': [],       # energy per unit mass to heat wallrock from T_wall to T
        'M_assim_total': [],          # total wallrock mass heated (at T), per system mass basis
        'M_assim_melt': []            # assimilant melt mass mixed into the magma
    }

    # Store previous total enthalpy-like H_prev = h*mass for "release" budget
    H_prev = _get_bulk_enthalpy_and_mass(Results['Conditions'])
    if H_prev is None:
        # If we can't compute enthalpy, we still run, but AFC reduces to zero.
        H_prev = np.nan

    # Step through the rest of the path
    for i in range(1, len(T)):
        # 2) Run the *next* MELTS node at (P[i], T[i]) starting from the previous state
        step_res = path_MELTS(
            Model=Model,
            comp=None,                 # continue from previous node (path_MELTS handles that internally)
            Frac_solid=True if Frac_solid else None,
            Frac_fluid=True if Frac_fluid else None,
            T_path_C=np.array([float(T[i])]),
            P_path_bar=np.array([float(P[i])]),
            fO2_buffer=fO2_buffer,
            fO2_offset=fO2_offset,
            fluid_sat=fluid_sat,
            Crystallinity_limit=Crystallinity_limit,
            Suppress=Suppress,
            Suppress_except=Suppress_except,
            melts=None                 # let helper create/use engine across nodes
        )

        # Merge this node's values into the running Results container, expanding frames
        # The structure of Results from path_MELTS has fixed lengths; we’ll append row-wise.
        # Grow Conditions
        for col in Results['Conditions'].columns:
            Results['Conditions'].loc[len(Results['Conditions']), col] = step_res['Conditions'].iloc[-1][col]

        # Ensure phases present in either dict are initialized in Results
        all_phases = set([k for k in Results.keys() if k not in ['Conditions']])
        all_phases.update([k for k in step_res.keys() if k not in ['Conditions']])

        for phase in all_phases:
            if phase == 'Conditions': 
                continue

            # create tables if missing
            if phase not in Results:
                Results[phase] = pd.DataFrame(columns=['SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','H2O','CO2'])
            if phase + '_prop' not in Results:
                Results[phase + '_prop'] = pd.DataFrame(columns=['h','mass','v','rho'])

            # Append last row from step_res (if available), else zeros
            if phase in step_res:
                Results[phase].loc[len(Results[phase])] = step_res[phase].iloc[-1].reindex(Results[phase].columns, fill_value=0.0)
            else:
                Results[phase].loc[len(Results[phase])] = [0.0]*len(Results[phase].columns)

            if phase + '_prop' in step_res:
                Results[phase+'_prop'].loc[len(Results[phase+'_prop'])] = step_res[phase+'_prop'].iloc[-1].reindex(Results[phase+'_prop'].columns, fill_value=np.nan)
            else:
                Results[phase+'_prop'].loc[len(Results[phase+'_prop'])] = [np.nan]*len(Results[phase+'_prop'].columns)

        # 3) Energy released by this step (from bulk enthalpy-like change)
        H_curr = _get_bulk_enthalpy_and_mass(Results['Conditions'])
        if (H_prev is not None) and (not np.isnan(H_prev)) and (H_curr is not None) and (not np.isnan(H_curr)):
            dH = H_curr - H_prev
            Q_release = max(0.0, -dH)   # positive energy budget available for assimilation
        else:
            Q_release = 0.0
        H_prev = H_curr

        # 4) Energy required per unit mass of wallrock to go from T_wall_C to T[i]
        #    and its liquid mass produced at (P[i],T[i])
        liq_assim_T, liq_mass_T, h_assim_T, m_assim_T = _equilibrate_assimilant(
            Model, P[i], T[i], bulk_assimilant,
            fO2_buffer=fO2_buffer, fO2_offset=fO2_offset, Suppress=Suppress
        )
        liq_assim_Twall, _, h_assim_Twall, m_assim_Twall = _equilibrate_assimilant(
            Model, P[i], T_wall_C, bulk_assimilant,
            fO2_buffer=fO2_buffer, fO2_offset=fO2_offset, Suppress=Suppress
        )

        # Guard NaNs
        if (not np.isnan(h_assim_T)) and (not np.isnan(h_assim_Twall)) and (not np.isnan(m_assim_T)) and (m_assim_T > 0):
            q_per_mass = (h_assim_T - h_assim_Twall)  # "specific" energy scale (same units as 'h')
        else:
            q_per_mass = np.nan

        # 5) Determine assimilant mass and melt mass to be added
        if (Q_release > 0.0) and (q_per_mass is not None) and (not np.isnan(q_per_mass)) and (q_per_mass > 0):
            M_assim_total = Q_release / q_per_mass
            # Melted fraction of assimilant at T[i] is estimated by (liq_mass / bulk_mass) at that equilibration point.
            # We need bulk mass at T[i]; if not available, approximate with liquid presence:
            if (not np.isnan(m_assim_T)) and (m_assim_T > 0):
                # If equilibrate_MELTS normalized bulk to ~100 mass units, treat liq_mass_T directly as mass units.
                M_assim_melt = M_assim_total * (liq_mass_T / m_assim_T if m_assim_T > 0 else 0.0)
            else:
                M_assim_melt = 0.0
        else:
            M_assim_total = 0.0
            M_assim_melt = 0.0

        # 6) Mix assimilant melt into current magma liquid, and re-seed bulk for next step
        #    (So AFC effect is carried forward.)
        # Current liquid composition and mass (from Results at this step)
        liq_mag = _liqdict_from_engine  # we don't have engine; instead read from tables
        # Grab the last recorded liquid row and mass:
        if 'liquid1' in Results and len(Results['liquid1'])>0:
            liq_mag_dict = {k: float(Results['liquid1'].iloc[-1][k]) for k in Results['liquid1'].columns}
            mass_mag_liq = float(Results['liquid1_prop'].iloc[-1]['mass']) if 'liquid1_prop' in Results else 100.0
        else:
            liq_mag_dict = {k:0.0 for k in ['SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','H2O','CO2']}
            mass_mag_liq = 100.0

        # Assimilant liquid composition at T[i]
        liq_assim_dict = liq_assim_T

        # Mix with mass weighting (units: consistent with how MELTS returns masses; relative scaling is fine)
        if M_assim_melt > 0:
            liq_mixed = _mix_two_liquids_wt(liq_mag_dict, mass_mag_liq, liq_assim_dict, M_assim_melt)
        else:
            liq_mixed = liq_mag_dict.copy()

        # Re-seed "bulk_magma" liquid for the *next* step by updating bulk_magma dict (…_Liq keys)
        # Convert mixed liquid dict to a new bulk list and map back to ..._Liq keys
        bulk_list = _normalize_bulk_from_liqdict(liq_mixed)

        # Convert FeO/Fe2O3 wt% in bulk_list back to FeOt and Fe3Fet as required by path_MELTS
        Fe2O3 = bulk_list[3]; FeO = bulk_list[5]
        FeOt = FeO + 71.844/(159.69/2.0)*Fe2O3
        Fe3Fet = 0.0 if (FeOt<=0) else (71.844/(159.69/2.0)*Fe2O3)/FeOt

        # Update the input dict for next node
        bulk_magma = {
            'SiO2_Liq': bulk_list[0], 'TiO2_Liq': bulk_list[1], 'Al2O3_Liq': bulk_list[2],
            'Fe3Fet_Liq': Fe3Fet, 'FeOt_Liq': FeOt, 'Cr2O3_Liq': bulk_list[4],
            'MnO_Liq': bulk_list[6], 'MgO_Liq': bulk_list[7], 'CaO_Liq': bulk_list[10],
            'Na2O_Liq': bulk_list[11], 'K2O_Liq': bulk_list[12], 'P2O5_Liq': bulk_list[13],
            'H2O_Liq': bulk_list[14], 'CO2_Liq': bulk_list[15]
        }

        # Log AFC step info
        AFC_log['step'].append(int(i))
        AFC_log['P_bar'].append(float(P[i]))
        AFC_log['T_C'].append(float(T[i]))
        AFC_log['Q_release'].append(float(Q_release))
        AFC_log['q_assim_per_mass'].append(float(q_per_mass) if (q_per_mass==q_per_mass) else np.nan)
        AFC_log['M_assim_total'].append(float(M_assim_total))
        AFC_log['M_assim_melt'].append(float(M_assim_melt))

        # IMPORTANT: Reset the system bulk for the next iteration by forcing MELTS
        # to start from the mixed liquid composition at the *current* state.
        # We achieve that by inserting one "re-seed" call at the *same* P,T before proceeding,
        # so that subsequent step starts from this mixed liquid.
        reseed = path_MELTS(
            Model=Model,
            comp=bulk_magma,                 # inject the mixed liquid as new 'bulk'
            Frac_solid=None,                 # do not fractionate in this reseed call
            Frac_fluid=None,
            T_path_C=np.array([float(T[i])]),
            P_path_bar=np.array([float(P[i])]),
            fO2_buffer=fO2_buffer,
            fO2_offset=fO2_offset,
            fluid_sat=None,
            Suppress=Suppress,
            Suppress_except=Suppress_except
        )
        # Overwrite the last rows in Results with reseeded liquid (to reflect AFC mix at this node)
        if 'liquid1' in reseed and 'liquid1' in Results and len(Results['liquid1'])>0:
            Results['liquid1'].iloc[-1,:] = reseed['liquid1'].iloc[-1,:]
        if 'liquid1_prop' in reseed and 'liquid1_prop' in Results and len(Results['liquid1_prop'])>0:
            Results['liquid1_prop'].iloc[-1,:] = reseed['liquid1_prop'].iloc[-1,:]

    # Attach AFC log
    Results['AFC'] = pd.DataFrame(AFC_log, columns=['step','P_bar','T_C','Q_release','q_assim_per_mass','M_assim_total','M_assim_melt'])
    return Results
