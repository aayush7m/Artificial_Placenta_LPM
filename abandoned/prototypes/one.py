import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class FetalCirculationModel:
    def __init__(self):
        self.params = {
            # Heart Parameters (Table 1 - Pennati 1997)
            'U_R0': 40.0,       # RV Isovolumic pressure generator constant (mmHg)
            'U_L0': 40.0,       # LV Isovolumic pressure generator constant (mmHg)
            'U_aR0': 3.0,       # RA contractile pressure constant (mmHg)
            'U_aL0': 3.0,       # LA contractile pressure constant (mmHg)
            'E_sys_R': 3.0,     # RV systolic elastance (mmHg/ml)
            'E_dia_R': 0.3,     # RV diastolic elastance (mmHg/ml)
            'E_sys_L': 3.0,     # LV systolic elastance (mmHg/ml)
            'E_dia_L': 0.3,     # LV diastolic elastance (mmHg/ml)
            'R_vR': 0.08,       # RV dissipative myocardial resistance (mmHg*s/ml)
            'R_vL': 0.08,       # LV dissipative myocardial resistance (mmHg*s/ml)
            'C_aR': 2.0,        # Right atrium compliance (ml/mmHg)
            'C_aL': 1.0,        # Left atrium compliance (ml/mmHg)
            'L_Li': 0.0016,     # Mitral valve inertance (mmHg*s^2/ml)
            'L_Ri': 0.0016,     # Tricuspid valve inertance (mmHg*s^2/ml)
            'K_Ri': 0.002,      # Tricuspid valve coefficient (mmHg*s^2/ml^2)
            'K_Li': 0.002,      # Mitral valve coefficient (mmHg*s^2/ml^2)
            'K_Ro': 0.001,      # Pulmonary valve coefficient (mmHg*s^2/ml^2)
            'K_Lo': 0.001,      # Aortic valve coefficient (mmHg*s^2/ml^2)
            
            # Circulation Resistances (Table 2 - Pennati 1997)
            'R_AA': 0.12, 'R_ISTHS': 0.4, 'R_THAO': 0.04, 'R_ADAO': 0.06,
            'R_PA': 0.07, 'R_LUNG': 13.5, 'R_DA': 0.01, 'R_CARG': 0.3,
            'R_SUCGA': 3.0, 'R_BR': 8.5, 'R_UB': 8.0, 'R_UBV': 4.9,
            'R_COEL': 81.0, 'R_MES': 34.0, 'R_POR': 7.0, 'R_REA': 3.5,
            'R_REV': 14.0, 'R_UA': 3.9, 'R_PLAC': 3.4, 'R_FA': 3.5,
            'R_LEGV': 0.6, 'R_HA': 0.5, 'R_HV': 0.16, 'R_DV': 1.3,
            'R_SVC': 0.2, 'R_IVC': 0.12, 'R_LA': 2.0,
            
            # Non-linear Shunt Coefficients K
            'K_FO': 0.4,        # Foramen ovale resistance coefficient
            'K_DA': 0.009,      # Ductus arteriosus local dissipation coefficient
            'K_DV': 0.26,       # Ductus venosus local dissipation coefficient
            
            # Shunt & Large Vessel Inertances L
            'L_DA': 0.006, 'L_CAR': 0.08, 'L_AA': 0.002, 'L_PA': 0.002,
            
            # Vascular Compliances C
            'C_AA': 0.05, 'C_AO1': 0.08, 'C_AO2': 0.07, 'C_AO3': 0.04, 'C_AO4': 0.05,
            'C_PA1': 0.08, 'C_PA2': 0.08, 'C_LUNG': 0.4, 'C_CA': 0.01, 'C_BR': 0.3,
            'C_UB': 0.85, 'C_INTE': 0.25, 'C_KID': 0.02, 'C_HE': 3.0, 'C_PLAC': 1.5,
            'C_LEG': 4.0, 'C_UV': 0.3, 'C_SVC': 1.0, 'C_IVC': 0.6
        }
        
        self.T_cycle = 0.43  # Fetal cardiac cycle period (~140 BPM)
        
    def get_activation(self, t):
        t_mod = t % self.T_cycle
        
        # Ventricular activation curve (Systole duration ~0.16 seconds)
        T_vs = 0.16
        A = np.sin(np.pi * t_mod / T_vs)**2 if t_mod < T_vs else 0.0
            
        # Atrial activation curve (Precedes ventricular systole at cycle wrap-around)
        T_as = 0.08
        t_as_start = self.T_cycle - T_as
        A_a = np.sin(np.pi * (t_mod - t_as_start) / T_as)**2 if t_mod >= t_as_start else 0.0
        
        return A, A_a

    def compute_derivatives(self, t, x):
        """derivatives for volumes and inertance flows."""
        # Unpack state vector
        V = x[0:23]
        Q_Li, Q_Ri, Q_AA, Q_PA, Q_DA, Q_CAR = x[23:29]
        prms = self.params
        
        A, A_a = self.get_activation(t)
        
        # 2. Linear Compliant Compartment Pressures (P = V / C)
        P = {
            'AA': V[0] / prms['C_AA'], 'AO1': V[1] / prms['C_AO1'], 'AO2': V[2] / prms['C_AO2'],
            'AO3': V[3] / prms['C_AO3'], 'AO4': V[4] / prms['C_AO4'], 'CA': V[5] / prms['C_CA'],
            'BR': V[6] / prms['C_BR'], 'UB': V[7] / prms['C_UB'], 'PA1': V[8] / prms['C_PA1'],
            'PA2': V[9] / prms['C_PA2'], 'LUNG': V[10] / prms['C_LUNG'], 'KID': V[11] / prms['C_KID'],
            'HE': V[12] / prms['C_HE'], 'INTE': V[13] / prms['C_INTE'], 'LEG': V[14] / prms['C_LEG'],
            'PLAC': V[15] / prms['C_PLAC'], 'UV': V[16] / prms['C_UV'], 'SVC': V[17] / prms['C_SVC'],
            'IVC': V[18] / prms['C_IVC'],
            'RA': prms['U_aR0'] * A_a + V[19] / prms['C_aR'],
            'LA': prms['U_aL0'] * A_a + V[20] / prms['C_aL']
        }
        
        # 3. Ventricular Elastance Calculations
        E_L = prms['E_dia_L'] + prms['E_sys_L'] * A
        E_R = prms['E_dia_R'] + prms['E_sys_R'] * A
        
        P_LV0 = prms['U_L0'] * A + E_L * V[22] + prms['R_vL'] * Q_Li
        P_RV0 = prms['U_R0'] * A + E_R * V[21] + prms['R_vR'] * Q_Ri
        
        # Solve for outflow valve streams using implicit quadratic resolution
        if P_LV0 > P['AA']:
            Q_Ao = (-prms['R_vL'] + np.sqrt(prms['R_vL']**2 + 4 * prms['K_Lo'] * (P_LV0 - P['AA']))) / (2 * prms['K_Lo'])
        else:
            Q_Ao = 0.0
            
        if P_RV0 > P['PA1']:
            Q_Pu = (-prms['R_vR'] + np.sqrt(prms['R_vR']**2 + 4 * prms['K_Ro'] * (P_RV0 - P['PA1']))) / (2 * prms['K_Ro'])
        else:
            Q_Pu = 0.0
            
        P_LV = P_LV0 - prms['R_vL'] * Q_Ao
        P_RV = P_RV0 - prms['R_vR'] * Q_Pu
        
        # 4. Non-linear and Linear Resistive Shunt Flows
        Q_UB = (P['AO1'] - P['UB']) / prms['R_UB']
        Q_UBV = (P['UB'] - P['SVC']) / prms['R_UBV']
        Q_CA_BR = (P['CA'] - P['BR']) / prms['R_SUCGA']
        Q_BR_SVC = (P['BR'] - P['SVC']) / prms['R_BR']
        Q_ISTHS = (P['AO1'] - P['AO2']) / prms['R_ISTHS']
        Q_THAO = (P['AO2'] - P['AO3']) / prms['R_THAO']
        Q_ADAO = (P['AO3'] - P['AO4']) / prms['R_ADAO']
        Q_COEL = (P['AO3'] - P['HE']) / prms['R_COEL']
        Q_MES = (P['AO3'] - P['INTE']) / prms['R_MES']
        Q_POR = (P['INTE'] - P['HE']) / prms['R_POR']
        Q_REA = (P['AO3'] - P['KID']) / prms['R_REA']
        Q_REV = (P['KID'] - P['IVC']) / prms['R_REV']
        Q_UA = (P['AO4'] - P['PLAC']) / prms['R_UA']
        Q_PLAC_UV = (P['PLAC'] - P['UV']) / prms['R_PLAC']
        Q_FA = (P['AO4'] - P['LEG']) / prms['R_FA']
        Q_LEG_IVC = (P['LEG'] - P['IVC']) / prms['R_LEGV']
        Q_HA = (P['UV'] - P['HE']) / prms['R_HA']
        Q_HV = (P['HE'] - P['IVC']) / prms['R_HV']
        
        # Ductus Venosus quadratic pressure & flow
        if P['UV'] > P['IVC']:
            Q_DV = (-prms['R_DV'] + np.sqrt(prms['R_DV']**2 + 4 * prms['K_DV'] * (P['UV'] - P['IVC']))) / (2 * prms['K_DV'])
        else:
            Q_DV = 0.0
            
        Q_LUNG_IN = (P['PA2'] - P['LUNG']) / prms['R_LUNG']
        Q_LUNG_OUT = (P['LUNG'] - P['LA']) / prms['R_LA']
        Q_SVC_RA = (P['SVC'] - P['RA']) / prms['R_SVC']
        Q_IVC_RA = (P['IVC'] - P['RA']) / prms['R_IVC']
        
        # Foramen Ovale (beta exponent = 0.625)
        Q_FO = ((P['IVC'] - P['LA']) / prms['K_FO'])**(1.0 / 0.625) if P['IVC'] > P['LA'] else 0.0
        
        # 5. Mass Balance Differential Equations (dV/dt)
        dV = np.zeros(23)
        dV[0] = Q_Ao - Q_AA
        dV[1] = Q_AA - Q_CAR - Q_UB - Q_ISTHS
        dV[2] = Q_ISTHS + Q_DA - Q_THAO
        dV[3] = Q_THAO - Q_ADAO - Q_COEL - Q_MES - Q_REA
        dV[4] = Q_ADAO - Q_UA - Q_FA
        dV[5] = Q_CAR - Q_CA_BR
        dV[6] = Q_CA_BR - Q_BR_SVC
        dV[7] = Q_UB - Q_UBV
        dV[8] = Q_Pu - Q_PA
        dV[9] = Q_PA - Q_LUNG_IN - Q_DA
        dV[10] = Q_LUNG_IN - Q_LUNG_OUT
        dV[11] = Q_REA - Q_REV
        dV[12] = Q_COEL + Q_POR + Q_HA - Q_HV
        dV[13] = Q_MES - Q_POR
        dV[14] = Q_FA - Q_LEG_IVC
        dV[15] = Q_UA - Q_PLAC_UV
        dV[16] = Q_PLAC_UV - Q_HA - Q_DV
        dV[17] = Q_UBV + Q_BR_SVC - Q_SVC_RA
        dV[18] = Q_REV + Q_LEG_IVC + Q_HV + Q_DV - Q_IVC_RA - Q_FO
        dV[19] = Q_SVC_RA + Q_IVC_RA - Q_Ri
        dV[20] = Q_LUNG_OUT + Q_FO - Q_Li
        dV[21] = Q_Ri - Q_Pu
        dV[22] = Q_Li - Q_Ao
        
        # 6. Momentum Balance Differential Equations (dQ/dt)
        dQ = np.zeros(6)
        dQ[0] = ((P['LA'] - P_LV - prms['K_Li'] * Q_Li * abs(Q_Li)) / prms['L_Li']) if (Q_Li > 0 or P['LA'] > P_LV) else 0.0
        dQ[1] = ((P['RA'] - P_RV - prms['K_Ri'] * Q_Ri * abs(Q_Ri)) / prms['L_Ri']) if (Q_Ri > 0 or P['RA'] > P_RV) else 0.0
        dQ[2] = (P['AA'] - P['AO1'] - prms['R_AA'] * Q_AA) / prms['L_AA']
        dQ[3] = (P['PA1'] - P['PA2'] - prms['R_PA'] * Q_PA) / prms['L_PA']
        dQ[4] = (P['PA2'] - P['AO2'] - prms['R_DA'] * Q_DA - prms['K_DA'] * Q_DA * abs(Q_DA)) / prms['L_DA']
        dQ[5] = (P['AO1'] - P['CA'] - prms['R_CARG'] * Q_CAR) / prms['L_CAR']
        
        return np.concatenate([dV, dQ])

    def run_simulation(self, duration=4.0):
        
        p = self.params
        c_list = [
            p['C_AA'], p['C_AO1'], p['C_AO2'], p['C_AO3'], p['C_AO4'],
            p['C_CA'], p['C_BR'], p['C_UB'], p['C_PA1'], p['C_PA2'],
            p['C_LUNG'], p['C_KID'], p['C_HE'], p['C_INTE'], p['C_LEG'],
            p['C_PLAC'], p['C_UV'], p['C_SVC'], p['C_IVC'], p['C_aR'], p['C_aL']
        ]
        
        # Stable blood volume using baseline fetal state
        x0 = np.zeros(29)
        P_init = 16.5  
        for i, comp_c in enumerate(c_list):
            x0[i] = comp_c * P_init
            
        x0[21], x0[22] = 6.0, 6.0  # Initial Ventricular Volumes
        
        sol = solve_ivp(self.compute_derivatives, [0, duration], x0, method='Radau', max_step=0.0005)
        return sol

# --- Eval & Plotting ---
if __name__ == '__main__':
    model = FetalCirculationModel()
    solution = model.run_simulation(4.3)
    
    t = solution.t
    mask = t >= 3.57  # one cardiac cycle
    t_cycle = t[mask] - 3.57
    
    ao1_pressure = solution.y[1, mask] / model.params['C_AO1']
    pa2_pressure = solution.y[9, mask] / model.params['C_PA2']
    lv_volume = solution.y[22, mask]
    rv_volume = solution.y[21, mask]
    da_flow = solution.y[27, mask]  # Ductus Arteriosus flow
    
    # Generate Visual Analysis Plots
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_cycle, ao1_pressure, 'r-', label='Aortic Arch Pressure (Systemic)')
    plt.plot(t_cycle, pa2_pressure, 'b--', label='Pulmonary Artery Pressure')
    plt.title('Fetal Hemodynamic Waveforms (Stabilized State Loop)')
    plt.ylabel('Pressure (mmHg)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t_cycle, lv_volume, 'g-', label='Left Ventricle Vol.')
    plt.plot(t_cycle, rv_volume, 'm--', label='Right Ventricle Vol.')
    plt.ylabel('Volume (ml)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_cycle, da_flow, 'k-', label='Ductus Arteriosus Shunt Flow')
    plt.xlabel('Time inside Cardiac Cycle (seconds)')
    plt.ylabel('Flow Rate (ml/s)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()