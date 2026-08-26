import csv
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


class ClinicalData:

    SAMPLE_RATE_HZ = 10.0        
    MIN_FLOW_ML_S  = 0.01        # ~3 mL/min

    def __init__(self, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Clinical data file not found: {csv_path}")

        self._times    = []   # elapsed time (s) for each row
        self._r_ecmo   = []   # R_ECMO (mmHg·s/mL) for each row
        self._r_res    = []   # R_RES  (mmHg·s/mL) for each row

        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)   # skip units-name header row
            next(reader)   # skip units row

            last_r_ecmo = 2.0  # fall back values
            last_r_res  = 1.0

            for idx, row in enumerate(reader):
                if len(row) < 5:
                    continue

                t_elapsed = idx / self.SAMPLE_RATE_HZ

                try:
                    flow_ml_min  = float(row[0])
                    p_pre_ecmo   = float(row[2])
                    p_post_ecmo  = float(row[3])
                    p_post_res   = float(row[4])
                except ValueError:
                    # carry forward last valid resistances
                    self._times.append(t_elapsed)
                    self._r_ecmo.append(last_r_ecmo)
                    self._r_res.append(last_r_res)
                    exit()
                    continue

                flow_ml_s = flow_ml_min / 60.0

                if abs(flow_ml_s) >= self.MIN_FLOW_ML_S:
                    r_ecmo = (p_pre_ecmo  - p_post_ecmo) / flow_ml_s
                    r_res  = (p_post_ecmo - p_post_res ) / flow_ml_s
                    last_r_ecmo = r_ecmo
                    last_r_res  = r_res
                else:
                    r_ecmo = last_r_ecmo
                    r_res  = last_r_res

                self._times.append(t_elapsed)
                self._r_ecmo.append(r_ecmo)
                self._r_res.append(r_res)

        if not self._times:
            raise ValueError("No valid data rows found in clinical CSV.")

        self._times  = np.array(self._times,  dtype=float)
        self._r_ecmo = np.array(self._r_ecmo, dtype=float)
        self._r_res  = np.array(self._r_res,  dtype=float)

        self.duration = self._times[-1]   # total span of clinical data (s)

    def get_resistances(self, t: float):
        """
        Return (R_ECMO, R_RES) at simulation time t using nearest-neighbour
        lookup.  If t exceeds the clinical data duration the last sample is
        returned (hold-last behaviour).
        """
        # Clamp to data range then find nearest index
        t_clamped = np.clip(t, self._times[0], self._times[-1])
        idx = int(np.argmin(np.abs(self._times - t_clamped)))
        return float(self._r_ecmo[idx]), float(self._r_res[idx])

class AdvancedFetalCirculationModel:
    def __init__(self, reflex_on=True, clinical_data: ClinicalData = None):
        self.reflex_on = reflex_on
        self.clinical_data = clinical_data   # ClinicalData instance for R lookup
        self.params = {
            # Ventricular & Atrial Elastance Extremes
            'E_max_L': 12.5,     # LV maximum systolic elastance (mmHg/ml)
            'E_min_L': 0.333,    # LV minimum diastolic elastance (mmHg/ml)
            'E_max_R': 12.5,     # RV maximum systolic elastance (mmHg/ml)
            'E_min_R': 0.333,    # RV minimum diastolic elastance (mmHg/ml)
            'E_max_LA': 3.333,   # LA maximum systolic elastance (mmHg/ml)
            'E_min_LA': 1.0,     # LA minimum diastolic elastance (mmHg/ml)
            'E_max_RA': 2.223,   # RA maximum systolic elastance (mmHg/ml)
            'E_min_RA': 0.5,     # RA minimum diastolic elastance (mmHg/ml)
            
            # Linear Unidirectional Valve Resistances
            'R_aortic': 0.0798,   'R_pulmonary': 0.0798,
            'R_mitral': 0.04002,  'R_tricuspid': 0.04002, 'R_FO': 0.096,
            
            # Circulation Resistances
            'R_AA': 0.12, 'R_ISTHS': 0.4, 'R_THAO': 0.04, 'R_ADAO': 0.06,
            'R_PA': 0.084, 'R_LUNG': 5.832, 'R_DA': 0.3, 'R_CAR': 0.3,
            'R_CA_BR': 3.0, 'R_BR': 8.5, 'R_UB': 8.0, 'R_UB_SVC': 4.9,
            'R_COEL': 81.0, 'R_MES': 34.0, 'R_POR': 7.0, 'R_REA': 3.5,
            'R_REV': 14.0, 'R_AO4_UA': 3.9, 'R_FA': 3.5,
            'R_LEG_IVC': 0.6, 'R_HA': 0.5, 'R_HV': 0.16, 'R_DV': 1.3,
            'R_SVC': 0.2, 'R_IVC': 0.12, 'R_LA': 1.92,
            
            # Artificial Placenta (AP) Circuit Parameters
            # R_RES and R_ECMO are now computed from clinical data;
            # fallback values below are used only when no ClinicalData is provided.
            'R_RES_fallback': 1.0,
            'R_ECMO_fallback': 2.0,
            
            # Vascular Compliances C
            'C_AR': 0.05, 'C_AO1': 0.08, 'C_AO2': 0.07, 'C_AO3': 0.04, 'C_AO4': 0.05,
            'C_PA1': 0.08, 'C_PA2': 0.08, 'C_LUNG': 0.4, 'C_CA': 0.01, 'C_BR': 0.3,
            'C_UB': 0.85, 'C_INTE': 0.25, 'C_KID': 0.02, 'C_HE': 3.0,
            'C_LEG': 4.0, 'C_UV': 0.3, 'C_SVC': 1.0, 'C_IVC': 0.6, 'C_UA':0.10
        }
        
        # Baroreceptor Reflex Baseline Constants
        self.HR_baseline = 140.0 / 60.0  # Baseline heart rate (Hz) -> 140 BPM
        self.P_sens_ref = 57.0           # Normotensive reference mean pressure (mmHg)
        self.tau_baro = 1.0              # Sensory filter time-constant (seconds)
        self.G_HR = 0.5 if reflex_on else 0.0  # Chronotropic gain
        self.G_R = 0.4 if reflex_on else 0.0   # Vasomotor gain

    def compute_derivatives(self, t, x):
        V = x[0:23]
        P_sens = x[23]
        prms = self.params

        # ── Dynamic AP circuit resistances from clinical data ──────────────
        # R_ECMO and R_RES are looked up from the nearest clinical sample to
        # the current solver time t.  If no ClinicalData was provided the
        # static fallback values are used instead.
        if self.clinical_data is not None:
            R_ECMO, R_RES = self.clinical_data.get_resistances(t)
        else:
            R_ECMO = prms['R_ECMO_fallback']
            R_RES  = prms['R_RES_fallback']
        # ──────────────────────────────────────────────────────────────────
        
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
        Q_HAr = (P['UV'] - P['HE']) / prms['R_HA'] # hepatic artery
        Q_LEG_IVC = (P['LEG'] - P['IVC']) / prms['R_LEG_IVC'] # leg to IVC
        Q_HVe = (P['HE'] - P['IVC']) / prms['R_HV'] # hepatic vein
        Q_IVC_RA = (P['IVC'] - P['RA']) / prms['R_IVC'] # IVC to RA
        
        def min_Q (Q_AP):  
            return P['UA'] - Q_AP*(R_RES + R_ECMO) + (-0.2374 * Q_AP + 50.315) - P['UV']
        
        sol = root_scalar(min_Q, bracket=[-500,500], method="brentq")

        Q_AP = sol.root

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
        dV[12] = Q_CEL + Q_POR + Q_HAr - Q_HVe # liver
        dV[13] = Q_MES - Q_POR # intestines
        dV[14] = Q_FA - Q_LEG_IVC # lower body
        dV[15] = Q_AO4_UA - Q_AP # umbilical artery
        dV[16] = Q_AP - Q_HAr - Q_DV # umbilical vein
        dV[17] = Q_UB_SVC + Q_BR_SVC - Q_SVC_RA # superior vena cava
        dV[18] = Q_RENVe + Q_LEG_IVC + Q_HVe + Q_DV - Q_IVC_RA # inferior vena cava
        dV[19] = Q_SVC_RA + Q_IVC_RA - Q_TrV - Q_FO # right atrium
        dV[20] = Q_LUNG_LA + Q_FO - Q_MiV # left atrium
        dV[21] = Q_TrV - Q_PuV # right ventricle
        dV[22] = Q_MiV - Q_AoV # left ventricle
        
        # First-order sensing filter state derivative
        dP_sens = (P['AR'] - P_sens) / self.tau_baro
        
        return np.append(dV, dP_sens)

    def run_simulation(self, duration=8.0):
        p = self.params
        c_list = [
            p['C_AR'], p['C_AO1'], p['C_AO2'], p['C_AO3'], p['C_AO4'],
            p['C_CA'], p['C_BR'], p['C_UB'], p['C_PA1'], p['C_PA2'],
            p['C_LUNG'], p['C_KID'], p['C_HE'], p['C_INTE'], p['C_LEG'],
            p['C_UA'], p['C_UV'], p['C_SVC'], p['C_IVC']
        ]
        x0 = np.zeros(24)
        P_init = 16.5  
        for i, comp_c in enumerate(c_list):
            x0[i] = comp_c * P_init
        x0[19], x0[20], x0[21], x0[22] = 4.0, 2.0, 10.4, 6.5  # Heart volumes
        x0[23] = 55.0 # Initial P_sens
        
        return solve_ivp(self.compute_derivatives, [0, duration], x0, method='Radau', max_step=0.002)

if __name__ == '__main__':
    # Load clinical data – path can be overridden via environment variable
    csv_path = os.environ.get(
        'ECMO_CSV_PATH',
        os.path.join(os.path.dirname(__file__), 'ECMO_short_trial.csv')
    )
    clin = ClinicalData(csv_path)

    # Run Reflex ON Scenario
    model_on = AdvancedFetalCirculationModel(reflex_on=True, clinical_data=clin)
    sol_on = model_on.run_simulation(duration=8.0)

    # Post-process to find relevant data at every time step
    dP_pump_history = []
    Q_AP_history = []
    
    for i in range(len(sol_on.t)):
        # get volumes
        V_t = sol_on.y[0:23, i]
        
        # recalc pressures for min_Q
        P_UA = V_t[15] / model_on.params['C_UA']
        P_UV = V_t[16] / model_on.params['C_UV']

        # look up resistances at this solver time
        t_i = sol_on.t[i]
        if model_on.clinical_data is not None:
            R_ECMO_i, R_RES_i = model_on.clinical_data.get_resistances(t_i)
        else:
            R_ECMO_i = model_on.params['R_ECMO_fallback']
            R_RES_i  = model_on.params['R_RES_fallback']
        
        # use root scalar to get Q_AP
        def min_Q_post(Q_AP):
            return P_UA - Q_AP*(R_RES_i + R_ECMO_i) + (-0.2374 * Q_AP + 50.315) - P_UV
            
        sol_Q = root_scalar(min_Q_post, bracket=[-500,500], method="brentq")
        Q_AP_t = sol_Q.root
        
        # calc and store pressure drop
        dP_pump_t = -0.2374 * Q_AP_t + 50.315
        dP_pump_history.append(dP_pump_t)
        Q_AP_history.append(Q_AP_t)
        
    # compare pump history with pre-pump to pre-ecmo pressure
    dP_pump_history = np.array(dP_pump_history)
    Q_AP_history = np.array(Q_AP_history)