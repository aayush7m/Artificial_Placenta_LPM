# ===========================================================================
# v12_1  Strategy: Add capacitors to ECMO & Restrictor in the AP circuit
# ===========================================================================

import csv
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# ============================================================
#  TIME WINDOW FOR COMPARISON
# ============================================================
START_TIME = "13:38:24"   # HH:MM:SS
END_TIME   = "13:38:54"   # HH:MM:SS
# ============================================================

def dPump (Q):
    return -0.2638 * Q + 62.207
def P_ECMO (Q):
    return -1.5363 * Q - 1.9786
def P_RES (Q):
    return -0.6194 * Q - 0.6247

class ClinicalData:

    SAMPLE_RATE_HZ = 10.0
    MIN_FLOW_ML_S  = 0.001   # ~3 mL/min

    def __init__(self, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Clinical data file not found: {csv_path}")

        self._times        = []
        self._flow_ml_s    = []   # raw flow (mL/s)
        self._dp_pump      = []   # pre-ECMO minus pre-pump pressure (mmHg)
        self._timestamps   = []   # wall-clock string from col 9
        self._p_pre_ecmo    = []   # pre-ECMO pressure (mmHg)
        self._p_post_ecmo   = []   # post-ECMO pressure (mmHg)
        self._p_post_res    = []   # post-restrictor / reservoir pressure (mmHg)
        self._p_pre_pump = []

        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)   # skip header
            next(reader)   # skip units

            for idx, row in enumerate(reader):
                if len(row) < 6:
                    continue

                t_elapsed = idx / self.SAMPLE_RATE_HZ

                try:
                    flow_ml_min  = float(row[0])
                    p_pre_ecmo   = float(row[2])
                    p_post_ecmo  = float(row[3])
                    p_post_res   = float(row[4])
                    p_pre_pump   = float(row[5])
                except ValueError as e:
                    msg = (
                        f"Error parsing clinical CSV at row {idx + 1}: {e}. Exiting program"
                    )
                    print(msg, file=sys.stderr)
                    sys.exit(1)

                # clock timestamp (column 9)
                ts = row[9].strip() if len(row) > 9 else ""
                self._timestamps.append(ts)

                flow_ml_s = flow_ml_min / 60.0
                self._flow_ml_s.append(flow_ml_s)

                # Clinical pump dP = pre-ECMO - pre-pump
                self._dp_pump.append(p_pre_ecmo - p_pre_pump)

                # store raw clinical pressures
                self._p_pre_ecmo.append(p_pre_ecmo)
                self._p_post_ecmo.append(p_post_ecmo)
                self._p_post_res.append(p_post_res)
                self._p_pre_pump.append(p_pre_pump)
                self._times.append(t_elapsed)

        if not self._times:
            raise ValueError("No valid data rows found in clinical CSV.")

        self._times      = np.array(self._times,     dtype=float)
        self._flow_ml_s  = np.array(self._flow_ml_s, dtype=float)
        self._dp_pump    = np.array(self._dp_pump,   dtype=float)
        self._timestamps = np.array(self._timestamps)
        self._p_pre_ecmo  = np.array(self._p_pre_ecmo,  dtype=float)
        self._p_post_ecmo = np.array(self._p_post_ecmo, dtype=float)
        self._p_post_res  = np.array(self._p_post_res,  dtype=float)
        self._p_pre_pump  = np.array(self._p_pre_pump,  dtype=float)

        self.duration = self._times[-1]

    # ------------------------------------------------------------------
    def get_time_window_indices(self, start_str: str, end_str: str):
        fmt = "%H:%M:%S"
        t_start = datetime.strptime(start_str, fmt)
        t_end   = datetime.strptime(end_str, fmt)

        mask = np.zeros(len(self._timestamps), dtype=bool)
        for i, ts in enumerate(self._timestamps):
            if not ts:
                continue
            # timestamps may repeat
            try:
                t_row = datetime.strptime(ts, fmt)
            except ValueError:
                continue
            if t_start <= t_row <= t_end:
                mask[i] = True
        return mask

class ArtificialPlacentaModel:
    # AP CIRCUIT: UA -> Pump -> ECMO -> Restrictor -> UV
    def __init__(self, reflex_on=True):
        self.reflex_on = reflex_on
        #correction factors
        GA40_weight = 3500
        GA28_weight = 1000
        C_factor = ((GA28_weight/GA40_weight) ** (0.335)) # 0.096
        R_factor = ((GA28_weight/GA40_weight) ** (-0.47)) # 1.8
        L_factor = (GA28_weight/GA40_weight) ** (-0.33)

        self.params = {
            # Ventricular & Atrial Elastance Extremes
            'E_max_L': 12.5/2,     # LV maximum systolic elastance (mmHg/ml)
            'E_min_L': 0.333*2,    # LV minimum diastolic elastance (mmHg/ml)
            'E_max_R': 12.5/2,     # RV maximum systolic elastance (mmHg/ml)
            'E_min_R': 0.333*2,    # RV minimum diastolic elastance (mmHg/ml)
            'E_max_LA': 3.333,   # LA maximum systolic elastance (mmHg/ml)
            'E_min_LA': 1.0,     # LA minimum diastolic elastance (mmHg/ml)
            'E_max_RA': 2.223,   # RA maximum systolic elastance (mmHg/ml)
            'E_min_RA': 0.5,     # RA minimum diastolic elastance (mmHg/ml)

            # Linear Unidirectional Valve Resistances
            'R_aortic': 0.0798 * R_factor,   'R_pulmonary': 0.0798 * R_factor,
            'R_mitral': 0.04002 * R_factor,  'R_tricuspid': 0.04002 * R_factor, 'R_FO': 0.096 * R_factor,

            # Circulation Resistances
            'R_AA': 0.12 * R_factor, 'R_ISTHS': 0.4 * R_factor, 'R_THAO': 0.04 * R_factor, 'R_ADAO': 0.06 * R_factor,
            'R_PA': 0.084 * R_factor, 'R_LUNG': 5.832 * R_factor, 'R_DA': 0.3 * R_factor, 'R_CAR': 0.3 * R_factor,
            'R_CA_BR': 3.0 * R_factor, 'R_BR': 8.5 * R_factor, 'R_UB': 8.0 * R_factor, 'R_UB_SVC': 4.9 * R_factor,
            'R_COEL': 81.0 * R_factor, 'R_MES': 34.0 * R_factor, 'R_POR': 7.0 * R_factor, 'R_REA': 3.5 * R_factor,
            'R_REV': 14.0 * R_factor, 'R_AO4_UA': 3.9 * R_factor, 'R_FA': 3.5 * R_factor,
            'R_LEG_IVC': 0.6 * R_factor, 'R_PRSN': (0.5) * R_factor, 'R_HV': 0.16 * R_factor, 'R_DV': (1.3) * R_factor,
            'R_SVC': 0.2 * R_factor, 'R_IVC': 0.12 * R_factor, 'R_LA': 1.92 * R_factor,

            'R_TUB': (1.66+0.95+0.24+1.19),

            # Vascular Compliances C
            'C_AR': 0.05 * C_factor, 'C_AO1': 0.08 * C_factor, 'C_AO2': 0.07 * C_factor, 'C_AO3': 0.04 * C_factor, 'C_AO4': 0.05 * C_factor,
            'C_PA1': 0.08 * C_factor, 'C_PA2': 0.08 * C_factor, 'C_LUNG': 0.4 * C_factor, 'C_CA': 0.01 * C_factor, 'C_BR': 0.3 * C_factor,
            'C_UB': 0.85 * C_factor, 'C_INTE': 0.25 * C_factor, 'C_KID': 0.02 * C_factor, 'C_HE': 3.0 * C_factor,
            'C_LEG': 4.0 * C_factor, 'C_UV': 0.3 * C_factor, 'C_SVC': 1.0 * C_factor, 'C_IVC': 0.6 * C_factor, 'C_UA': 0.1 * C_factor,

            # --- AP circuit capacitances (White et al. 2025, Table 1) ---
            'C_ECMO': 0.1,    # Oxygenator capacitance  [mL/mmHg]
            'C_RES':  0.04,   # Outflow-cannula / restrictor capacitance [mL/mmHg]
        }

        # Derived: split tubing resistance equally upstream/downstream of pump
        self.R_tub_in  = self.params['R_TUB'] / 2.0
        self.R_tub_out = self.params['R_TUB'] / 2.0

        # Effective resistances extracted from the linear P-Q curves
        # P_ECMO(Q) = -1.5363*Q - 1.9786  =>  R_ecmo_eff = 1.5363, offset = -1.9786
        # P_RES(Q)  = -0.6194*Q - 0.6247  =>  R_res_eff  = 0.6194, offset = -0.6247
        self.R_ecmo_eff  = 1.5363
        self.R_ecmo_off  = -1.9786
        self.R_res_eff   = 0.6194
        self.R_res_off   = -0.6247
        # dPump(Q)  = -0.2638*Q + 62.207  =>  R_pump = 0.2638, gain0 = 62.207
        self.R_pump      = 0.2638
        self.dP_pump0    = 62.207

        # Baroreceptor Reflex Baseline Constants
        self.HR_baseline = 140.0 / 60.0  # Baseline heart rate (Hz) -> 140 BPM
        self.P_sens_ref = 57.0           # Normotensive reference mean pressure (mmHg)
        self.tau_baro = 1.0              # Sensory filter time-constant (seconds)
        self.G_HR = 0.5 if reflex_on else 0.0  # Chronotropic gain
        self.G_R = 0.4 if reflex_on else 0.0   # Vasomotor gain

    # ------------------------------------------------------------------

    def compute_derivatives(self, t, x):
        # State vector layout:
        #   x[0:23]  = 23 vascular compartment volumes
        #   x[23]    = P_sens  (baroreceptor sensory filter)
        #   x[24]    = V_ecmo_cap  (volume in ECMO capacitor node)
        #   x[25]    = V_res_cap   (volume in restrictor capacitor node)
        V      = x[0:23]
        P_sens = x[23]
        V_ecmo_cap = x[24]
        V_res_cap  = x[25]
        prms   = self.params

        # 1. Baroreceptor Reflex Modulation
        p_error = (P_sens - self.P_sens_ref) / self.P_sens_ref

        # Modulate Heart Rate (Negative Chronotropic Effect)
        HR = self.HR_baseline * (1.0 - self.G_HR * p_error)
        HR = max(90.0 / 60.0, min(190.0 / 60.0, HR))  # Safe physiological bounds
        T_cycle = 1.0 / HR

        # Modulate Systemic Peripheral Resistances (Vasomotor Component)
        r_factor = 1.0 + self.G_R * p_error
        r_factor = max(0.5, min(2.0, r_factor))

        # 2. Time-Varying Cardiac Elastance Curves
        t_mod = t % T_cycle
        t_n = t_mod * HR
        term1_v = (t_n / 0.303)**1.32
        en_v = 1.67 * (term1_v / (1.0 + term1_v)) * (1.0 / (1.0 + (t_n / 0.508)**21.9))

        t_n_a = (t_n + 0.275) % 1.0
        term1_a = (t_n_a / 0.303)**1.32
        en_a = 2.70 * (term1_a / (1.0 + term1_a)) * (1.0 / (1.0 + (t_n_a / 0.25)**21.9))

        E_R = (prms['E_max_R'] - prms['E_min_R']) * en_v + prms['E_min_R']
        E_L = (prms['E_max_L'] - prms['E_min_L']) * en_v + prms['E_min_L']
        E_RA = (prms['E_max_RA'] - prms['E_min_RA']) * en_a + prms['E_min_RA']
        E_LA = (prms['E_max_LA'] - prms['E_min_LA']) * en_a + prms['E_min_LA']

        # 3. Compartment Pressures (P = V / C)
        P = {
            'AR': V[0] / prms['C_AR'], 'AO1': V[1] / prms['C_AO1'], 'AO2': V[2] / prms['C_AO2'],
            'AO3': V[3] / prms['C_AO3'], 'AO4': V[4] / prms['C_AO4'], 'CA': V[5] / prms['C_CA'],
            'BR': V[6] / prms['C_BR'], 'UB': V[7] / prms['C_UB'], 'PA1': V[8] / prms['C_PA1'],
            'PA2': V[9] / prms['C_PA2'], 'LUNG': V[10] / prms['C_LUNG'], 'KID': V[11] / prms['C_KID'],
            'HE': V[12] / prms['C_HE'], 'INTE': V[13] / prms['C_INTE'], 'LEG': V[14] / prms['C_LEG'],
            'UA': V[15] / prms['C_UA'],
            'UV': V[16] / prms['C_UV'], 'SVC': V[17] / prms['C_SVC'], 'IVC': V[18] / prms['C_IVC'],
            'RA': E_RA * V[19], 'LA': E_LA * V[20], 'RV': E_R * V[21], 'LV': E_L * V[22]
        }

        # 4. AP circuit with capacitive intermediate nodes
        # Node pressures
        P_A = V_ecmo_cap / prms['C_ECMO']   # post-pump / pre-ECMO node
        P_B = V_res_cap  / prms['C_RES']     # post-ECMO / pre-restrictor node

        # Segment 1: UA -> pump -> node_A
        # P_UA + dPump(Q1) - Q1*R_tub_in = P_A
        # => Q1 = (P_UA + dP_pump0 - P_A) / (R_pump + R_tub_in)
        Q1 = (P['UA'] + self.dP_pump0 - P_A) / (self.R_pump + self.R_tub_in)

        # Segment 2: node_A -> ECMO -> node_B
        # P_A + P_ECMO(Q2) = P_B  =>  P_A - R_ecmo*Q2 + offset = P_B
        # => Q2 = (P_A + offset - P_B) / R_ecmo
        Q2 = (P_A + self.R_ecmo_off - P_B) / self.R_ecmo_eff

        # Segment 3: node_B -> restrictor -> UV
        # P_B + P_RES(Q3) - Q3*R_tub_out = P_UV
        # => Q3 = (P_B + offset - P_UV) / (R_res + R_tub_out)
        Q3 = (P_B + self.R_res_off - P['UV']) / (self.R_res_eff + self.R_tub_out)

        # 5. Remaining flows (unchanged from v11)
        Q_AoV = max(0.0, (P['LV'] - P['AR']) / prms['R_aortic']) # aortic valve
        Q_PuV = max(0.0, (P['RV'] - P['PA1']) / prms['R_pulmonary']) # pulmonary valve
        Q_MiV = max(0.0, (P['LA'] - P['LV']) / prms['R_mitral']) # mitral valve
        Q_TrV = max(0.0, (P['RA'] - P['RV']) / prms['R_tricuspid']) # tricuspid valve
        Q_FO = max(0.0, (P['RA'] - P['LA']) / prms['R_FO']) # foramen ovale

        Q_AA = (P['AR'] - P['AO1']) / prms['R_AA'] # ascending aorta
        Q_CaAr = (P['AO1'] - P['CA']) / prms['R_CAR'] #carotid arteries
        Q_ScAr = (P['AO1'] - P['UB']) / (prms['R_UB'] * r_factor) # subclavian arteries
        Q_ISTH = (P['AO1'] - P['AO2']) / prms['R_ISTHS'] # aortic isthmus

        Q_CA_BR = (P['CA'] - P['BR']) / prms['R_CA_BR'] # cerebral arteries to brain
        Q_BR_SVC = (P['BR'] - P['SVC']) / (prms['R_BR'] * r_factor) # brain to SVC
        Q_UB_SVC = (P['UB'] - P['SVC']) / prms['R_UB_SVC'] # upper body to SVC
        Q_SVC_RA = (P['SVC'] - P['RA']) / prms['R_SVC'] # SVC to RA

        Q_PA = (P['PA1'] - P['PA2']) / prms['R_PA'] # pulmonary artery
        Q_DA = (P['PA2'] - P['AO2']) / prms['R_DA'] # ductus arteriosus

        Q_HIAT = (P['AO2'] - P['AO3']) / prms['R_THAO'] # aortic hiatus
        Q_ILIA = (P['AO3'] - P['AO4']) / prms['R_ADAO'] # iliac arteries
        Q_CEL = (P['AO3'] - P['HE']) / (prms['R_COEL'] * r_factor) # celiac artery
        Q_MES = (P['AO3'] - P['INTE']) / (prms['R_MES'] * r_factor) # mesenteric artery
        Q_RENAr = (P['AO3'] - P['KID']) / (prms['R_REA'] * r_factor) # renal artery

        Q_PA2_LUNG = (P['PA2'] - P['LUNG']) / prms['R_LUNG'] # lung in
        Q_LUNG_LA = (P['LUNG'] - P['LA']) / prms['R_LA'] # lung out

        Q_POR = (P['INTE'] - P['HE']) / prms['R_POR'] # portal vein
        Q_RENVe = (P['KID'] - P['IVC']) / prms['R_REV'] # renal vein

        Q_FA = (P['AO4'] - P['LEG']) / (prms['R_FA'] * r_factor) # femoral artery

        Q_AO4_UA = (P['AO4'] - P['UA']) / prms['R_AO4_UA']
        Q_DV = max(0.0, (P['UV'] - P['IVC']) / prms['R_DV']) # ductus venosus
        Q_PRSN = (P['UV'] - P['HE']) / prms['R_PRSN'] # portal sinus
        Q_LEG_IVC = (P['LEG'] - P['IVC']) / prms['R_LEG_IVC'] # leg to IVC
        Q_HVe = (P['HE'] - P['IVC']) / prms['R_HV'] # hepatic vein
        Q_IVC_RA = (P['IVC'] - P['RA']) / prms['R_IVC'] # IVC to RA

        # 6. Mass Balance Equations (dV/dt)
        dV = np.zeros(23)
        dV[0] = Q_AoV - Q_AA # Ao root
        dV[1] = Q_AA - Q_CaAr - Q_ScAr - Q_ISTH # Ao arch
        dV[2] = Q_ISTH + Q_DA - Q_HIAT # thoracic desc Ao
        dV[3] = Q_HIAT - Q_ILIA - Q_CEL - Q_MES - Q_RENAr # abd Ao
        dV[4] = Q_ILIA - Q_AO4_UA - Q_FA # femoral bifurc
        dV[5] = Q_CaAr - Q_CA_BR # cerebral arteries
        dV[6] = Q_CA_BR - Q_BR_SVC # brain
        dV[7] = Q_ScAr - Q_UB_SVC # upper body
        dV[8] = Q_PuV - Q_PA # pulm trunk
        dV[9] = Q_PA - Q_PA2_LUNG - Q_DA # pulm art
        dV[10] = Q_PA2_LUNG - Q_LUNG_LA # lungs
        dV[11] = Q_RENAr - Q_RENVe # kidneys
        dV[12] = Q_CEL + Q_POR + Q_PRSN - Q_HVe # liver
        dV[13] = Q_MES - Q_POR # intestines
        dV[14] = Q_FA - Q_LEG_IVC # lower body
        dV[15] = Q_AO4_UA - Q1                # UA: flow in minus flow leaving to pump
        dV[16] = Q3 - Q_PRSN - Q_DV           # UV: flow arriving from restrictor
        dV[17] = Q_UB_SVC + Q_BR_SVC - Q_SVC_RA # superior vena cava
        dV[18] = Q_RENVe + Q_LEG_IVC + Q_HVe + Q_DV - Q_IVC_RA # inferior vena cava
        dV[19] = Q_SVC_RA + Q_IVC_RA - Q_TrV - Q_FO # right atrium
        dV[20] = Q_LUNG_LA + Q_FO - Q_MiV # left atrium
        dV[21] = Q_TrV - Q_PuV # right ventricle
        dV[22] = Q_MiV - Q_AoV # left ventricle

        # Baroreceptor sensory filter
        dP_sens = (P['AR'] - P_sens) / self.tau_baro

        # Capacitor node mass balances
        dV_ecmo_cap = Q1 - Q2    # post-pump / pre-ECMO node
        dV_res_cap  = Q2 - Q3    # post-ECMO / pre-restrictor node

        return np.concatenate([dV, [dP_sens, dV_ecmo_cap, dV_res_cap]])

    # ------------------------------------------------------------------
    def run_simulation(self, duration=8.0):
        p = self.params
        c_list = [
            p['C_AR'], p['C_AO1'], p['C_AO2'], p['C_AO3'], p['C_AO4'],
            p['C_CA'], p['C_BR'],  p['C_UB'],  p['C_PA1'], p['C_PA2'],
            p['C_LUNG'],p['C_KID'],p['C_HE'],  p['C_INTE'],p['C_LEG'],
            p['C_UA'], p['C_UV'],  p['C_SVC'], p['C_IVC']
        ]
        x0 = np.zeros(26)   # 23 volumes + P_sens + V_ecmo_cap + V_res_cap
        P_init = 16.5
        for i, comp_c in enumerate(c_list):
            x0[i] = comp_c * P_init

        x0[19], x0[20], x0[21], x0[22] = 4.0, 2.0, 10.4, 6.5
        x0[23] = 55.0

        # Initialize capacitor nodes at a reasonable intermediate pressure
        x0[24] = p['C_ECMO'] * P_init   # V_ecmo_cap
        x0[25] = p['C_RES']  * P_init   # V_res_cap

        return solve_ivp(self.compute_derivatives, [0, duration], x0,
                         method='Radau', max_step=0.002)


# ===========================================================================
def post_process(sol, model):
    """Re-derive AP flows and pressures at every saved solver time step."""
    dp_pump  = []
    q_ap     = []
    pre_ecmo_list = []
    post_ecmo_list = []
    p_uv_list = []
    p_ua_list = []

    for i in range(len(sol.t)):
        V_t   = sol.y[0:23, i]
        P_UA  = V_t[15] / model.params['C_UA']
        P_UV  = V_t[16] / model.params['C_UV']

        V_ecmo_cap = sol.y[24, i]
        V_res_cap  = sol.y[25, i]
        P_A = V_ecmo_cap / model.params['C_ECMO']
        P_B = V_res_cap  / model.params['C_RES']

        # Segment flows (same formulas as in compute_derivatives)
        Q1 = (P_UA + model.dP_pump0 - P_A) / (model.R_pump + model.R_tub_in)
        Q2 = (P_A + model.R_ecmo_off - P_B) / model.R_ecmo_eff
        Q3 = (P_B + model.R_res_off - P_UV) / (model.R_res_eff + model.R_tub_out)

        # Use Q1 as representative AP flow (flow entering circuit from UA)
        Q_AP_t = Q1

        pump_gain = dPump(Q_AP_t)
        pre_ecmo = pump_gain + P_UA          # = P_A (approximately, ignoring tubing)
        post_ecmo = P_A + P_ECMO(Q2)         # pressure after ECMO drop

        p_ua_list.append(P_UA)
        p_uv_list.append(P_UV)
        pre_ecmo_list.append(P_A)            # node_A IS the pre-ECMO pressure
        post_ecmo_list.append(P_B)           # node_B IS the post-ECMO pressure
        dp_pump.append(pump_gain)
        q_ap.append(Q_AP_t)

    return np.array(dp_pump), np.array(q_ap), np.array(pre_ecmo_list), np.array(post_ecmo_list), np.array(p_uv_list), np.array(p_ua_list)


# ===========================================================================
def comparison_plots(sol, model, clin, start_str, end_str):
    dp_pump_model, q_ap_model, pre_ecmo_model, post_ecmo_model, post_restrictor_model, ua_model = post_process(sol, model)
    t_model = sol.t

    mask = clin.get_time_window_indices(start_str, end_str)
    if mask.sum() == 0:
        raise ValueError(f"No clinical rows found between {start_str} and {end_str}.")

    t_clin_raw  = clin._times[mask]
    dp_clin_raw = clin._dp_pump[mask]
    p_ua_clin_raw = clin._p_pre_pump[mask]
    q_clin_raw  = clin._flow_ml_s[mask]
    t_clin = t_clin_raw - t_clin_raw[0]
    q_clin_ml_min  = q_clin_raw  * 60.0
    q_model_ml_min = q_ap_model  * 60.0

    plt.rcParams.update({'font.size': 12})

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 10))
    axes = axes.flatten() # Flatten the 2D array of axes to 1D for easy indexing

    common_title_info = (
        f"LPM Model vs Clinical Data  |  "
        f"Clinical window: {start_str} - {end_str}  ({t_clin[-1]:.1f} s)  |  "
        f"Model duration: {t_model[-1]:.1f} s  (reflex_on={model.reflex_on})"
    )
    fig.suptitle(common_title_info, fontsize=16, fontweight='bold')

    def plot_on_axis(ax, y_clin, y_model, clin_label, model_label, ylabel, title_prefix, ylim_bottom=None):
        ax.plot(t_clin, y_clin, color='green', lw=1.5, label=clin_label)
        ax.plot(t_model, y_model, color='red', lw=1.5, ls='-', label=model_label)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title_prefix, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)

    # 1. Pump ΔP
    plot_on_axis(axes[0], dp_clin_raw, dp_pump_model, 'Clinical  (Pre-ECMO - Pre-Pump)', 'LPM Model', 'Pump ΔP  (mmHg)', 'Pump ΔP')
    # 2. AP Flow
    plot_on_axis(axes[1], q_clin_ml_min, q_model_ml_min, 'Clinical Flow (mL/min)', 'LPM Model  Q_AP (mL/min)', 'AP Flow (mL/min)', 'AP Flow')
    # 3. Pre-Pump Pressure
    plot_on_axis(axes[4], p_ua_clin_raw, ua_model, 'Clinical Pre-Pump', 'Model Pre-Pump', 'Pressure (mmHg)', 'Pre-Pump Pressure')
    # 4. Pre-ECMO Pressure
    plot_on_axis(axes[5], clin._p_pre_ecmo[mask], pre_ecmo_model, 'Clinical Pre-ECMO', 'Model Pre-ECMO', 'Pressure (mmHg)', 'Pre-ECMO Pressure', ylim_bottom=0)
    # 5. Post-ECMO Pressure
    plot_on_axis(axes[6], clin._p_post_ecmo[mask], post_ecmo_model, 'Clinical Post-ECMO', 'Model Post-ECMO', 'Pressure (mmHg)', 'Post-ECMO Pressure', ylim_bottom=0)
    # 6. Post-Restrictor Pressure
    plot_on_axis(axes[7], clin._p_post_res[mask], post_restrictor_model, 'Clinical Post-Restrictor', 'Model Post-Restrictor (P_UV)', 'Pressure (mmHg)', 'Post-Restrictor Pressure', ylim_bottom=0)
    # 7. ECMO Pressure Drop
    plot_on_axis(axes[2], clin._p_pre_ecmo[mask] - clin._p_post_ecmo[mask], pre_ecmo_model - post_ecmo_model, 'Clinical ECMO drop', 'Model ECMO drop', 'Pressure (mmHg)', 'ECMO Pressure Drop')
    # 8. Restrictor Pressure Drop
    plot_on_axis(axes[3], clin._p_post_ecmo[mask] - clin._p_post_res[mask], post_ecmo_model - post_restrictor_model, 'Clinical Restrictor drop', 'Model Restrictor drop', 'Pressure (mmHg)', 'Restrictor Pressure Drop')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    plt.show()


if __name__ == '__main__':
    csv_path = os.environ.get(
        'ECMO_CSV_PATH',
        os.path.join(os.path.dirname(__file__), 'ECMO_Trial_2024-03-05_cleaned_v2.csv')
    )
    clin = ClinicalData(csv_path)

    # Duration driven by the chosen clinical window
    fmt = "%H:%M:%S"
    t0 = datetime.strptime(START_TIME, fmt)
    t1 = datetime.strptime(END_TIME,   fmt)
    window_s = (t1 - t0).total_seconds()
    if window_s <= 0:
        raise ValueError("END_TIME must be after START_TIME.")

    print(f"Clinical window: {START_TIME} → {END_TIME}  ({window_s:.0f} s)")
    print("Running LPM simulation (v12_1: ECMO+Restrictor capacitors) …")

    model_on = ArtificialPlacentaModel(reflex_on=True)
    sol_on   = model_on.run_simulation(duration=window_s)

    print(f"Simulation complete.  Solver steps: {len(sol_on.t)}")
    comparison_plots(sol_on, model_on, clin, START_TIME, END_TIME)
