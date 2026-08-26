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

                # Clinical pump dP = pre-ECMO – pre-pump
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

# ===========================================================================
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
            'C_LEG': 4.0 * C_factor, 'C_UV': 0.3 * C_factor, 'C_SVC': 1.0 * C_factor, 'C_IVC': 0.6 * C_factor, 'C_UA': 0.1 * C_factor
        }
        
        # Baroreceptor Reflex Baseline Constants
        self.HR_baseline = 140.0 / 60.0  # Baseline heart rate (Hz) -> 140 BPM
        self.P_sens_ref = 57.0           # Normotensive reference mean pressure (mmHg)
        self.tau_baro = 1.0              # Sensory filter time-constant (seconds)
        self.G_HR = 0.5 if reflex_on else 0.0  # Chronotropic gain
        self.G_R = 0.4 if reflex_on else 0.0   # Vasomotor gain

    # ------------------------------------------------------------------
    
    def compute_derivatives(self, t, x):
        V      = x[0:23]
        P_sens = x[23]
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
            'UA': V[15] / prms['C_UA'],   # Pressure in artificial placenta oxygenator
            'UV': V[16] / prms['C_UV'], 'SVC': V[17] / prms['C_SVC'], 'IVC': V[18] / prms['C_IVC'],
            'RA': E_RA * V[19], 'LA': E_LA * V[20], 'RV': E_R * V[21], 'LV': E_L * V[22]
        }

        # 4. Flows and Interconnections
        def min_Q (Q_AP):  
            return P['UA'] - Q_AP*prms['R_TUB'] + P_ECMO(Q_AP) + P_RES(Q_AP) + dPump(Q_AP) - P['UV']
        
        sol = root_scalar(min_Q, bracket=[-500,500], method="brentq")

        Q_AP = sol.root

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
        
        # 5. Mass Balance Equations (dV/dt)
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
        dV[15] = Q_AO4_UA - Q_AP # umbilical artery
        dV[16] = Q_AP - Q_PRSN - Q_DV # umbilical vein
        dV[17] = Q_UB_SVC + Q_BR_SVC - Q_SVC_RA # superior vena cava
        dV[18] = Q_RENVe + Q_LEG_IVC + Q_HVe + Q_DV - Q_IVC_RA # inferior vena cava
        dV[19] = Q_SVC_RA + Q_IVC_RA - Q_TrV - Q_FO # right atrium
        dV[20] = Q_LUNG_LA + Q_FO - Q_MiV # left atrium
        dV[21] = Q_TrV - Q_PuV # right ventricle
        dV[22] = Q_MiV - Q_AoV # left ventricle
        
        # First-order sensing filter state derivative
        dP_sens = (P['AR'] - P_sens) / self.tau_baro
        
        return np.append(dV, dP_sens)

    # ------------------------------------------------------------------
    def run_simulation(self, duration=8.0):
        p = self.params
        c_list = [
            p['C_AR'], p['C_AO1'], p['C_AO2'], p['C_AO3'], p['C_AO4'],
            p['C_CA'], p['C_BR'],  p['C_UB'],  p['C_PA1'], p['C_PA2'],
            p['C_LUNG'],p['C_KID'],p['C_HE'],  p['C_INTE'],p['C_LEG'],
            p['C_UA'], p['C_UV'],  p['C_SVC'], p['C_IVC']
        ]
        x0 = np.zeros(24)
        P_init = 16.5
        for i, comp_c in enumerate(c_list):
            x0[i] = comp_c * P_init

        # x0 = 0.45 * x0
        x0[19], x0[20], x0[21], x0[22] = 4.0, 2.0, 10.4, 6.5
        x0[23] = 55.0


        return solve_ivp(self.compute_derivatives, [0, duration], x0,
                         method='Radau', max_step=0.002)


# ===========================================================================
def post_process(sol, model):
    """Re-derive Q_AP and dP_pump at every saved solver time step."""
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

        def min_Q_post(Q_AP):
            return P_UA - Q_AP * model.params['R_TUB'] + P_ECMO(Q_AP) + P_RES(Q_AP) + dPump(Q_AP) - P_UV

        sol_Q  = root_scalar(min_Q_post, bracket=[-500,500], method="brentq")
        Q_AP_t = sol_Q.root

        pump_gain = dPump(Q_AP_t)
        pre_ecmo = pump_gain + P_UA
        post_ecmo = pre_ecmo + + P_ECMO(Q_AP_t)

        p_ua_list.append(P_UA)
        p_uv_list.append(P_UV)
        pre_ecmo_list.append(pre_ecmo)
        post_ecmo_list.append(post_ecmo)
        dp_pump.append(pump_gain)
        q_ap.append(Q_AP_t)

    return np.array(dp_pump), np.array(q_ap), np.array(pre_ecmo_list), np.array(post_ecmo_list), np.array(p_uv_list), np.array(p_ua_list)


# ===========================================================================
def comparison_plots(sol, model, clin, start_str, end_str):
    # --- 1.  Post-process model -------------------------------------------
    dp_pump_model, q_ap_model, pre_ecmo_model, post_ecmo_model, post_restrictor_model,ua_model = post_process(sol, model)
    t_model = sol.t # seconds from simulation start (t=0)

    # --- 2.  Clinical window mask -----------------------------------------
    mask = clin.get_time_window_indices(start_str, end_str)
    if mask.sum() == 0:
        raise ValueError(
            f"No clinical rows found between {start_str} and {end_str}. "
            "Check that the timestamps are within the CSV range."
        )

    t_clin_raw  = clin._times[mask] # elapsed seconds in CSV
    dp_clin_raw = clin._dp_pump[mask] # pre-ECMO – pre-pump (mmHg)
    p_ua_clin_raw = clin._p_pre_pump[mask]
    q_clin_raw  = clin._flow_ml_s[mask] # mL/s
    
    # Shift clinical time so window starts at 0 (align with model t=0)
    t_clin = t_clin_raw - t_clin_raw[0]

    q_clin_ml_min  = q_clin_raw  * 60.0
    q_model_ml_min = q_ap_model  * 60.0

    # --- 3.  Plot -----------------------------------------------------------
    plt.rcParams.update({'font.size': 12})

    def create_plot(y_clin, y_model, clin_label, model_label, ylabel, title_prefix, figsize=(10, 4), ylim_bottom=None):
        common_title_info = (
            f" — LPM Model vs Clinical Data\n"
            f"Clinical window: {start_str} - {end_str}  ({t_clin[-1]:.1f} s)  "
            f"Model duration: {t_model[-1]:.1f} s  (reflex_on={model.reflex_on})"
        )   
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Plot data
        ax.plot(t_clin, y_clin, color='green', lw=1.5, label=clin_label)
        ax.plot(t_model, y_model, color='red', lw=1.5, ls='-', label=model_label)
        
        # Common formatting
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f"{title_prefix}{common_title_info}", fontsize=14)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        
        # Optional y-limit
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)
            
        return fig, ax

 
    # Pump ΔP
    fig_pump, ax_pump = create_plot(
        y_clin=dp_clin_raw, y_model=dp_pump_model,
        clin_label='Clinical  (Pre-ECMO - Pre-Pump)', model_label='LPM Model',
        ylabel='Pump ΔP  (mmHg)', title_prefix='Pump ΔP'
    )

    # AP Flow
    fig_flow, ax_flow = create_plot(
        y_clin=q_clin_ml_min, y_model=q_model_ml_min,
        clin_label='Clinical Flow (mL/min)', model_label='LPM Model  Q_AP (mL/min)',
        ylabel='AP Flow (mL/min)', title_prefix='AP Flow'
    )

    # UA P (pre-pump)
    fig_UA, ax_UA = create_plot(
        y_clin=p_ua_clin_raw, y_model=ua_model,
        clin_label='Clinical Pre-Pump', model_label='Model Pre-Pump',
        ylabel='Pressure (mmHg)', title_prefix='Pre-Pump Pressure'
    )

    # Pre-ECMO
    fig_pre_ECMO, ax_pre_ECMO = create_plot(
        y_clin=clin._p_pre_ecmo[mask], y_model=pre_ecmo_model,
        clin_label='Clinical Pre-ECMO', model_label='Model Pre-ECMO',
        ylabel='Pressure (mmHg)', title_prefix='Pre-ECMO Pressure', 
        ylim_bottom=0
    )

    # Post-ECMO
    fig_post_ECMO, ax_post_ECMO = create_plot(
        y_clin=clin._p_post_ecmo[mask], y_model=post_ecmo_model,
        clin_label='Clinical Post-ECMO', model_label='Model Post-ECMO',
        ylabel='Pressure (mmHg)', title_prefix='Post-ECMO Pressure', 
        ylim_bottom=0
    )

    # Post-Restrictor
    fig_post_res, ax_post_res = create_plot(
        y_clin=clin._p_post_res[mask], y_model=post_restrictor_model,
        clin_label='Clinical Post-Restrictor', model_label='Model Post-Restrictor (P_UV)',
        ylabel='Pressure (mmHg)', title_prefix='Post-Restrictor Pressure', 
        ylim_bottom=0
    )

    # ECMO Drop
    fig_ECMO, ax_ECMO = create_plot(
        y_clin=clin._p_pre_ecmo[mask] - clin._p_post_ecmo[mask], 
        y_model=pre_ecmo_model - post_ecmo_model,
        clin_label='Clinical ECMO drop', model_label='Model ECMO drop',
        ylabel='Pressure (mmHg)', title_prefix='ECMO Pressure Drop'
    )

    # Restrictor Drop
    fig_RES, ax_RES = create_plot(
        y_clin=clin._p_post_ecmo[mask] - clin._p_post_res[mask], 
        y_model=post_ecmo_model - post_restrictor_model,
        clin_label='Clinical Restrictor drop', model_label='Model Restrictor drop',
        ylabel='Pressure (mmHg)', title_prefix='Restrictor Pressure Drop'
    )

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
    print("Running LPM simulation …")

    model_on = ArtificialPlacentaModel(reflex_on=True)
    sol_on   = model_on.run_simulation(duration=window_s)

    print(f"Simulation complete.  Solver steps: {len(sol_on.t)}")
    comparison_plots(sol_on, model_on, clin, START_TIME, END_TIME)