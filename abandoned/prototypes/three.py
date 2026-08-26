import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

class AdvancedFetalCirculationModel:
    def __init__(self, reflex_on=True):
        self.reflex_on = reflex_on
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
            
            # Circulation Resistances (Natural Placenta Replaced with Low-Resistance AP)
            'R_AA': 0.12, 'R_ISTHS': 0.4, 'R_THAO': 0.04, 'R_ADAO': 0.06,
            'R_PA': 0.084, 'R_LUNG': 5.832, 'R_DA': 0.3, 'R_CARG': 0.3,
            'R_SUCGA': 3.0, 'R_BR': 8.5, 'R_UB': 8.0, 'R_UBV': 4.9,
            'R_COEL': 81.0, 'R_MES': 34.0, 'R_POR': 7.0, 'R_REA': 3.5,
            'R_REV': 14.0, 'R_UA': 3.9, 'R_FA': 3.5,
            'R_LEGV': 0.6, 'R_HA': 0.5, 'R_HV': 0.16, 'R_DV': 1.3,
            'R_SVC': 0.2, 'R_IVC': 0.12, 'R_LA': 1.92,
            
            # Artificial Placenta (AP) Core Circuit Parameters
            'R_AP': 1.0,         # Low-resistance clinical oxygenator membrane (mmHg*s/ml)
            'C_AP': 2.0,         # Compliance of the external blood circuit (ml/mmHg)
            
            # Vascular Compliances C
            'C_AA': 0.05, 'C_AO1': 0.08, 'C_AO2': 0.07, 'C_AO3': 0.04, 'C_AO4': 0.05,
            'C_PA1': 0.08, 'C_PA2': 0.08, 'C_LUNG': 0.4, 'C_CA': 0.01, 'C_BR': 0.3,
            'C_UB': 0.85, 'C_INTE': 0.25, 'C_KID': 0.02, 'C_HE': 3.0,
            'C_LEG': 4.0, 'C_UV': 0.3, 'C_SVC': 1.0, 'C_IVC': 0.6
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
            'AA': V[0] / prms['C_AA'], 'AO1': V[1] / prms['C_AO1'], 'AO2': V[2] / prms['C_AO2'],
            'AO3': V[3] / prms['C_AO3'], 'AO4': V[4] / prms['C_AO4'], 'CA': V[5] / prms['C_CA'],
            'BR': V[6] / prms['C_BR'], 'UB': V[7] / prms['C_UB'], 'PA1': V[8] / prms['C_PA1'],
            'PA2': V[9] / prms['C_PA2'], 'LUNG': V[10] / prms['C_LUNG'], 'KID': V[11] / prms['C_KID'],
            'HE': V[12] / prms['C_HE'], 'INTE': V[13] / prms['C_INTE'], 'LEG': V[14] / prms['C_LEG'],
            'AP': V[15] / prms['C_AP'],   # Pressure in artificial placenta oxygenator
            'UV': V[16] / prms['C_UV'], 'SVC': V[17] / prms['C_SVC'], 'IVC': V[18] / prms['C_IVC'],
            'RA': E_RA * V[19], 'LA': E_LA * V[20], 'RV': E_R * V[21], 'LV': E_L * V[22]
        }
        
        # Dynamic Disturbance: Turn on Artificial Placenta Pump assist at t >= 4.0s
        P_pump = 15.0 if t >= 4.0 else 0.0
        
        # 4. Flows and Interconnections
        Q_Ao = max(0.0, (P['LV'] - P['AA']) / prms['R_aortic'])
        Q_Pu = max(0.0, (P['RV'] - P['PA1']) / prms['R_pulmonary'])
        Q_Li = max(0.0, (P['LA'] - P['LV']) / prms['R_mitral'])
        Q_Ri = max(0.0, (P['RA'] - P['RV']) / prms['R_tricuspid'])
        Q_FO = max(0.0, (P['RA'] - P['LA']) / prms['R_FO'])
        
        Q_AA = (P['AA'] - P['AO1']) / prms['R_AA']
        Q_CAR = (P['AO1'] - P['CA']) / prms['R_CARG']
        Q_UB = (P['AO1'] - P['UB']) / (prms['R_UB'] * r_factor)
        Q_ISTHS = (P['AO1'] - P['AO2']) / prms['R_ISTHS']
        
        Q_CA_BR = (P['CA'] - P['BR']) / prms['R_SUCGA']
        Q_BR_SVC = (P['BR'] - P['SVC']) / (prms['R_BR'] * r_factor)
        Q_UBV = (P['UB'] - P['SVC']) / prms['R_UBV']
        Q_SVC_RA = (P['SVC'] - P['RA']) / prms['R_SVC']
        
        Q_PA = (P['PA1'] - P['PA2']) / prms['R_PA']
        Q_DA = (P['PA2'] - P['AO2']) / prms['R_DA']  
        
        Q_THAO = (P['AO2'] - P['AO3']) / prms['R_THAO']
        Q_ADAO = (P['AO3'] - P['AO4']) / prms['R_ADAO']
        Q_COEL = (P['AO3'] - P['HE']) / (prms['R_COEL'] * r_factor)
        Q_MES = (P['AO3'] - P['INTE']) / (prms['R_MES'] * r_factor)
        Q_REA = (P['AO3'] - P['KID']) / (prms['R_REA'] * r_factor)
        
        Q_LUNG_IN = (P['PA2'] - P['LUNG']) / prms['R_LUNG']
        Q_LUNG_OUT = (P['LUNG'] - P['LA']) / prms['R_LA']
        
        Q_POR = (P['INTE'] - P['HE']) / prms['R_POR']
        Q_REV = (P['KID'] - P['IVC']) / prms['R_REV']
        
        # Umbilical lines routing to the Artificial Placenta System
        Q_UA = (P['AO4'] - P['AP']) / prms['R_UA']
        Q_FA = (P['AO4'] - P['LEG']) / (prms['R_FA'] * r_factor)
        
        # Outflow from AP oxygenator includes pump assist pressure
        Q_AP_UV = (P['AP'] + P_pump - P['UV']) / prms['R_AP']
        Q_DV = max(0.0, (P['UV'] - P['IVC']) / prms['R_DV'])
        Q_HA = (P['UV'] - P['HE']) / prms['R_HA']
        Q_LEG_IVC = (P['LEG'] - P['IVC']) / prms['R_LEGV']
        Q_HV = (P['HE'] - P['IVC']) / prms['R_HV']
        Q_IVC_RA = (P['IVC'] - P['RA']) / prms['R_IVC']
        
        # 5. Mass Balance Equations (dV/dt)
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
        dV[15] = Q_UA - Q_AP_UV 
        dV[16] = Q_AP_UV - Q_HA - Q_DV  
        dV[17] = Q_UBV + Q_BR_SVC - Q_SVC_RA   
        dV[18] = Q_REV + Q_LEG_IVC + Q_HV + Q_DV - Q_IVC_RA  
        dV[19] = Q_SVC_RA + Q_IVC_RA - Q_Ri - Q_FO 
        dV[20] = Q_LUNG_OUT + Q_FO - Q_Li  
        dV[21] = Q_Ri - Q_Pu
        dV[22] = Q_Li - Q_Ao  
        
        # First-order sensing filter state derivative
        dP_sens = (P['AA'] - P_sens) / self.tau_baro
        
        return np.append(dV, dP_sens)

    def run_simulation(self, duration=8.0):
        p = self.params
        c_list = [
            p['C_AA'], p['C_AO1'], p['C_AO2'], p['C_AO3'], p['C_AO4'],
            p['C_CA'], p['C_BR'], p['C_UB'], p['C_PA1'], p['C_PA2'],
            p['C_LUNG'], p['C_KID'], p['C_HE'], p['C_INTE'], p['C_LEG'],
            p['C_AP'], p['C_UV'], p['C_SVC'], p['C_IVC']
        ]
        x0 = np.zeros(24)
        P_init = 16.5  
        for i, comp_c in enumerate(c_list):
            x0[i] = comp_c * P_init
        x0[19], x0[20], x0[21], x0[22] = 4.0, 2.0, 10.4, 6.5  # Heart volumes
        x0[23] = 55.0                                         # Initial P_sens
        
        return solve_ivp(self.compute_derivatives, [0, duration], x0, method='Radau', max_step=0.002)

if __name__ == '__main__':
    sim_duration = 8.0
    
    # Run Reflex ON Scenario
    model_on = AdvancedFetalCirculationModel(reflex_on=True)
    sol_on = model_on.run_simulation(sim_duration)
    
    # Run Reflex OFF Scenario
    model_off = AdvancedFetalCirculationModel(reflex_on=False)
    sol_off = model_off.run_simulation(sim_duration)
    
    # Process Reflex ON Curves
    t_on = sol_on.t
    p_aa_on = sol_on.y[0] / model_on.params['C_AA']
    p_sens_on = sol_on.y[23]
    hr_on = [model_on.HR_baseline * (1.0 - model_on.G_HR * ((ps - model_on.P_sens_ref) / model_on.P_sens_ref)) * 60.0 for ps in p_sens_on]
    p_ap_on = sol_on.y[15] / model_on.params['C_AP']
    p_uv_on = sol_on.y[16] / model_on.params['C_UV']
    q_ap_on = [max(0.0, (pa + (15.0 if ti >= 4.0 else 0.0) - pu) / model_on.params['R_AP']) * 60.0 / 1000.0 for ti, pa, pu in zip(t_on, p_ap_on, p_uv_on)]

    # Process Reflex OFF Curves
    t_off = sol_off.t
    p_aa_off = sol_off.y[0] / model_off.params['C_AA']
    p_sens_off = sol_off.y[23]
    hr_off = [model_off.HR_baseline * 60.0] * len(t_off)
    
    # Save each plot separately into a `figures/` folder with captions
    os.makedirs('figures', exist_ok=True)

    # Figure 1: Ascending Aorta Pressure
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t_off, p_aa_off, 'r--', alpha=0.5, label='Reflex OFF')
    plt.plot(t_on, p_aa_on, 'r-', linewidth=1.5, label='Reflex ON')
    plt.axvline(x=4.0, color='k', linestyle=':', label='AP Pump Activated')
    plt.title('Ascending Aorta Pressure ($P_{AA}$)', fontweight='bold')
    plt.ylabel('Pressure (mmHg)')
    plt.xlabel('Time (s)')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':')
    caption1 = (
        "Compares ascending aorta pressure with baroreceptor reflex ON vs OFF. "
        "Vertical line marks artificial placenta (AP) pump activation at 4 s    "
    )
    plt.figtext(0.5, 0.01, caption1, wrap=True, ha='center', fontsize=9)
    fig.subplots_adjust(bottom=0.21)
    fig.savefig(os.path.join('figures', '01_aorta_pressure.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: Fetal Heart Rate (FHR)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t_off, hr_off, 'b--', alpha=0.5, label='Reflex OFF')
    plt.plot(t_on, hr_on, 'b-', linewidth=1.5, label='Reflex ON')
    plt.axvline(x=4.0, color='k', linestyle=':')
    plt.title('Fetal Heart Rate (FHR)', fontweight='bold')
    plt.ylabel('Heart Rate (BPM)')
    plt.xlabel('Time (s)')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':')
    caption2 = (
        "Shows baroreceptor-mediated chronotropic modulation. Reflex ON alters heart rate "
        "in response to sensed pressure changes while Reflex OFF remains at baseline."
    )
    plt.figtext(0.5, 0.01, caption2, wrap=True, ha='center', fontsize=9)
    fig.subplots_adjust(bottom=0.21)
    fig.savefig(os.path.join('figures', '02_fetal_hr.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure 3: Artificial Placenta Circuit Flow
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t_on, q_ap_on, 'g-', linewidth=1.5, label='AP Circuit Flow')
    plt.axvline(x=4.0, color='k', linestyle=':')
    plt.title('Artificial Placenta Circuit Flow ($Q_{AP}$)', fontweight='bold')
    plt.ylabel('Flow Rate (L/min)')
    plt.xlabel('Time (s)')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':')
    caption3 = (
        "Demonstrates artificial placenta (AP) circuit flow and the pump assist effect after 4 s"
    )
    plt.figtext(0.5, 0.01, caption3, wrap=True, ha='center', fontsize=9)
    fig.subplots_adjust(bottom=0.21)
    fig.savefig(os.path.join('figures', '03_ap_flow.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure 4: Pulsatile Pressures (Zoom 7-8s)
    fig = plt.figure(figsize=(6, 4))
    mask = t_on >= 7.0
    plt.plot(t_on[mask], p_aa_on[mask], 'r-', label='Aorta ($P_{AA}$)')
    plt.plot(t_on[mask], p_ap_on[mask], 'm-', label='Oxygenator ($P_{AP}$)')
    plt.title('Pulsatile Pressures (Zoomed 7-8s)', fontweight='bold')
    plt.ylabel('Pressure (mmHg)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':')
    caption4 = (
        "Zoomed view showing pulsatility and coupling between oxygenator and aortic pressures "
        "late in the simulation; confirms AP influence on pulsatile hemodynamics."
    )
    plt.figtext(0.5, 0.01, caption4, wrap=True, ha='center', fontsize=9)
    fig.subplots_adjust(bottom=0.21)
    fig.savefig(os.path.join('figures', '04_pulsatile_zoom.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)