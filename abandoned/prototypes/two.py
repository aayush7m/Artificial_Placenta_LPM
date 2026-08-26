import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class FetalCirculationModel:
    def __init__(self):
        self.params = {
            # Ventricular & Atrial Elastance Extremes (Yigit Paper Supplementary Data)
            'E_max_L': 12.5,     # LV maximum systolic elastance (mmHg/ml)
            'E_min_L': 0.333,    # LV minimum diastolic elastance (mmHg/ml)
            'E_max_R': 12.5,     # RV maximum systolic elastance (mmHg/ml)
            'E_min_R': 0.333,    # RV minimum diastolic elastance (mmHg/ml)
            'E_max_LA': 3.333,   # LA maximum systolic elastance (mmHg/ml)
            'E_min_LA': 1.0,     # LA minimum diastolic elastance (mmHg/ml)
            'E_max_RA': 2.223,   # RA maximum systolic elastance (mmHg/ml)
            'E_min_RA': 0.5,     # RA minimum diastolic elastance (mmHg/ml)
            
            # Linear Unidirectional Valve Resistances (Converted using x0.06 factor)
            'R_aortic': 0.0798,   # Aortic valve resistance (mmHg*s/ml)
            'R_pulmonary': 0.0798,# Pulmonary valve resistance (mmHg*s/ml)
            'R_mitral': 0.04002,  # Mitral valve resistance (mmHg*s/ml)
            'R_tricuspid': 0.04002,# Tricuspid valve resistance (mmHg*s/ml)
            'R_FO': 0.096,        # Foramen ovale resistance (mmHg*s/ml)
            
            # Circulation Resistances (Table 2 - Fetal State Baseline)
            'R_AA': 0.12, 'R_ISTHS': 0.4, 'R_THAO': 0.04, 'R_ADAO': 0.06,
            'R_PA': 0.084, 'R_LUNG': 5.832, 'R_DA': 0.3, 'R_CARG': 0.3,
            'R_SUCGA': 3.0, 'R_BR': 8.5, 'R_UB': 8.0, 'R_UBV': 4.9,
            'R_COEL': 81.0, 'R_MES': 34.0, 'R_POR': 7.0, 'R_REA': 3.5,
            'R_REV': 14.0, 'R_UA': 3.9, 'R_PLAC': 3.4, 'R_FA': 3.5,
            'R_LEGV': 0.6, 'R_HA': 0.5, 'R_HV': 0.16, 'R_DV': 1.3,
            'R_SVC': 0.2, 'R_IVC': 0.12, 'R_LA': 1.92,
            
            # Vascular Compliances C
            'C_AA': 0.05, 'C_AO1': 0.08, 'C_AO2': 0.07, 'C_AO3': 0.04, 'C_AO4': 0.05,
            'C_PA1': 0.08, 'C_PA2': 0.08, 'C_LUNG': 0.4, 'C_CA': 0.01, 'C_BR': 0.3,
            'C_UB': 0.85, 'C_INTE': 0.25, 'C_KID': 0.02, 'C_HE': 3.0, 'C_PLAC': 1.5,
            'C_LEG': 4.0, 'C_UV': 0.3, 'C_SVC': 1.0, 'C_IVC': 0.6
        }
        
        # Fetal heart rate baseline = 140 BPM
        self.HR = 140.0 / 60.0  # Hz
        self.T_cycle = 1.0 / self.HR  # ~0.4286 seconds
        
    def get_elastances(self, t):
        """Computes the time-varying elastances using Yigit Equations (3a) and (3b)."""
        t_mod = t % self.T_cycle
        t_n = t_mod * self.HR  # Normalized time inside the cardiac cycle [0, 1]
        
        # Ventricular normalized activation curve (Eq 3a)
        term1_v = (t_n / 0.303)**1.32
        en_v = 1.67 * (term1_v / (1.0 + term1_v)) * (1.0 / (1.0 + (t_n / 0.508)**21.9))
        
        # Atrial normalized activation curve (Eq 3b) with 0.275 phase advance shift
        t_n_a = (t_n + 0.275) % 1.0
        term1_a = (t_n_a / 0.303)**1.32
        en_a = 2.70 * (term1_a / (1.0 + term1_a)) * (1.0 / (1.0 + (t_n_a / 0.25)**21.9))
        
        p = self.params
        E_R = (p['E_max_R'] - p['E_min_R']) * en_v + p['E_min_R']
        E_L = (p['E_max_L'] - p['E_min_L']) * en_v + p['E_min_L']
        E_RA = (p['E_max_RA'] - p['E_min_RA']) * en_a + p['E_min_RA']
        E_LA = (p['E_max_LA'] - p['E_min_LA']) * en_a + p['E_min_LA']
        
        return E_R, E_L, E_RA, E_LA

    def compute_derivatives(self, t, x):
        """Pure RC Companion mass balance network states."""
        V = x[0:23]
        prms = self.params
        
        # Unpack active driving elastances
        E_R, E_L, E_RA, E_LA = self.get_elastances(t)
        
        # Linear Compliant Compartment Pressures (P = V / C)
        P = {
            'AA': V[0] / prms['C_AA'], 'AO1': V[1] / prms['C_AO1'], 'AO2': V[2] / prms['C_AO2'],
            'AO3': V[3] / prms['C_AO3'], 'AO4': V[4] / prms['C_AO4'], 'CA': V[5] / prms['C_CA'],
            'BR': V[6] / prms['C_BR'], 'UB': V[7] / prms['C_UB'], 'PA1': V[8] / prms['C_PA1'],
            'PA2': V[9] / prms['C_PA2'], 'LUNG': V[10] / prms['C_LUNG'], 'KID': V[11] / prms['C_KID'],
            'HE': V[12] / prms['C_HE'], 'INTE': V[13] / prms['C_INTE'], 'LEG': V[14] / prms['C_LEG'],
            'PLAC': V[15] / prms['C_PLAC'], 'UV': V[16] / prms['C_UV'], 'SVC': V[17] / prms['C_SVC'],
            'IVC': V[18] / prms['C_IVC'],
            'RA': E_RA * V[19],
            'LA': E_LA * V[20],
            'RV': E_R * V[21],
            'LV': E_L * V[22]
        }
        
        # Unidirectional Heart Valve and Shunt Flows
        Q_Ao = max(0.0, (P['LV'] - P['AA']) / prms['R_aortic'])
        Q_Pu = max(0.0, (P['RV'] - P['PA1']) / prms['R_pulmonary'])
        Q_Li = max(0.0, (P['LA'] - P['LV']) / prms['R_mitral'])
        Q_Ri = max(0.0, (P['RA'] - P['RV']) / prms['R_tricuspid'])
        Q_FO = max(0.0, (P['RA'] - P['LA']) / prms['R_FO'])
        
        # Linear Interconnection Flow Streams
        Q_AA = (P['AA'] - P['AO1']) / prms['R_AA']
        Q_CAR = (P['AO1'] - P['CA']) / prms['R_CARG']
        Q_UB = (P['AO1'] - P['UB']) / prms['R_UB']
        Q_ISTHS = (P['AO1'] - P['AO2']) / prms['R_ISTHS']
        
        Q_CA_BR = (P['CA'] - P['BR']) / prms['R_SUCGA']
        Q_BR_SVC = (P['BR'] - P['SVC']) / prms['R_BR']
        Q_UBV = (P['UB'] - P['SVC']) / prms['R_UBV']
        Q_SVC_RA = (P['SVC'] - P['RA']) / prms['R_SVC']
        
        Q_PA = (P['PA1'] - P['PA2']) / prms['R_PA']
        Q_DA = (P['PA2'] - P['AO2']) / prms['R_DA']  
        
        Q_THAO = (P['AO2'] - P['AO3']) / prms['R_THAO']
        Q_ADAO = (P['AO3'] - P['AO4']) / prms['R_ADAO']
        Q_COEL = (P['AO3'] - P['HE']) / prms['R_COEL']
        Q_MES = (P['AO3'] - P['INTE']) / prms['R_MES']
        Q_REA = (P['AO3'] - P['KID']) / prms['R_REA']
        
        Q_LUNG_IN = (P['PA2'] - P['LUNG']) / prms['R_LUNG']
        Q_LUNG_OUT = (P['LUNG'] - P['LA']) / prms['R_LA']
        
        Q_POR = (P['INTE'] - P['HE']) / prms['R_POR']
        Q_REV = (P['KID'] - P['IVC']) / prms['R_REV']
        Q_UA = (P['AO4'] - P['PLAC']) / prms['R_UA']
        Q_FA = (P['AO4'] - P['LEG']) / prms['R_FA']
        
        Q_PLAC_UV = (P['PLAC'] - P['UV']) / prms['R_PLAC']
        Q_DV = (P['UV'] - P['IVC']) / prms['R_DV']
        Q_HA = (P['UV'] - P['HE']) / prms['R_HA']
        Q_LEG_IVC = (P['LEG'] - P['IVC']) / prms['R_LEGV']
        Q_HV = (P['HE'] - P['IVC']) / prms['R_HV']
        Q_IVC_RA = (P['IVC'] - P['RA']) / prms['R_IVC']
        
        # Mass Balance State Derivatives (dV/dt)
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
        dV[18] = Q_REV + Q_LEG_IVC + Q_HV + Q_DV - Q_IVC_RA  
        dV[19] = Q_SVC_RA + Q_IVC_RA - Q_Ri - Q_FO          
        dV[20] = Q_LUNG_OUT + Q_FO - Q_Li                   
        dV[21] = Q_Ri - Q_Pu                                
        dV[22] = Q_Li - Q_Ao                                
        
        return dV

    def run_simulation(self, duration=4.3):
        p = self.params
        c_list = [
            p['C_AA'], p['C_AO1'], p['C_AO2'], p['C_AO3'], p['C_AO4'],
            p['C_CA'], p['C_BR'], p['C_UB'], p['C_PA1'], p['C_PA2'],
            p['C_LUNG'], p['C_KID'], p['C_HE'], p['C_INTE'], p['C_LEG'],
            p['C_PLAC'], p['C_UV'], p['C_SVC'], p['C_IVC']
        ]
        
        x0 = np.zeros(23)
        P_init = 16.5  
        for i, comp_c in enumerate(c_list):
            x0[i] = comp_c * P_init
            
        x0[19] = 4.0    # RA Vol
        x0[20] = 2.0    # LA Vol
        x0[21] = 10.4   # RV Volume (ml)
        x0[22] = 6.5    # LV Volume (ml)
        
        sol = solve_ivp(self.compute_derivatives, [0, duration], x0, method='Radau', max_step=0.0005)
        return sol

if __name__ == '__main__':
    model = FetalCirculationModel()
    sim_duration = 5.0 
    solution = model.run_simulation(sim_duration)
    
    t = solution.t
    prms = model.params
    
    two_cycles_duration = 2 * model.T_cycle
    mask = t >= (sim_duration - two_cycles_duration)
    
    t_raw = t[mask]
    t_start = sim_duration - two_cycles_duration
    t_normalized = (t_raw - t_start) / model.T_cycle 
    
    p_asc = solution.y[0, mask] / prms['C_AA']      
    p_mpa = solution.y[8, mask] / prms['C_PA1']     
    
    ml_s_to_l_min = 60.0 / 1000.0
    
    E_R, E_L, E_RA, E_LA = zip(*[model.get_elastances(ti) for ti in t_raw])
    E_R, E_L, E_RA, E_LA = np.array(E_R), np.array(E_L), np.array(E_RA), np.array(E_LA)
    
    p_lv = E_L * solution.y[22, mask]
    p_rv = E_R * solution.y[21, mask]
    p_la = E_LA * solution.y[20, mask]
    p_ra = E_RA * solution.y[19, mask]
    p_ao2 = solution.y[2, mask] / prms['C_AO2']
    p_pa2 = solution.y[9, mask] / prms['C_PA2']
    p_uv = solution.y[16, mask] / prms['C_UV']
    p_ivc = solution.y[18, mask] / prms['C_IVC']
    
    q_aortic = np.maximum(0.0, (p_lv - p_asc) / prms['R_aortic']) * ml_s_to_l_min
    q_pulmonary = np.maximum(0.0, (p_rv - p_mpa) / prms['R_pulmonary']) * ml_s_to_l_min
    q_mitral = np.maximum(0.0, (p_la - p_lv) / prms['R_mitral']) * ml_s_to_l_min
    q_tricuspid = np.maximum(0.0, (p_ra - p_rv) / prms['R_tricuspid']) * ml_s_to_l_min
    
    q_da = ((p_pa2 - p_ao2) / prms['R_DA']) * ml_s_to_l_min
    q_dv = np.maximum(0.0, (p_uv - p_ivc) / prms['R_DV']) * ml_s_to_l_min
    q_ivc = ((p_ivc - p_ra) / prms['R_IVC']) * ml_s_to_l_min

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle('Steady State Fetal Circ', fontsize=16, fontweight='bold')
    
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    ax3 = plt.subplot2grid((3, 2), (0, 1))
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax5 = plt.subplot2grid((3, 2), (2, 1))

    # 1. Arterial Blood Pressures
    ax1.plot(t_normalized, p_asc, 'r-', linewidth=2, label='Ascending Aorta (ASC)')
    ax1.plot(t_normalized, p_mpa, 'b-', linewidth=1.5, label='Main Pulmonary Artery (MPA)')
    ax1.set_title('Arterial Blood Pressures', fontweight='bold', loc='left')
    ax1.set_ylabel('pressure [mmHg]')
    ax1.set_xlabel('cycle time')
    ax1.set_xlim(0, 2)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), frameon=False)

    # 2. Valvular Flow Tracings
    ax2.plot(t_normalized, q_aortic, 'r-', linewidth=2, label='Aortic Valve')
    ax2.plot(t_normalized, q_pulmonary, 'b-', linewidth=2, label='Pulmonary Valve')
    ax2.plot(t_normalized, q_mitral, 'r--', linewidth=1.5, label='Mitral Valve')
    ax2.plot(t_normalized, q_tricuspid, 'b--', linewidth=1.5, label='Tricuspid Valve')
    ax2.set_title('Valvular Flow Tracings', fontweight='bold', loc='left')
    ax2.set_ylabel('flow rate [l/min]')
    ax2.set_xlabel('cycle time')
    ax2.set_xlim(0, 2)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), ncol=2, frameon=False)

    # 3. Ductus Arteriosus (DA) Flow
    ax3.plot(t_normalized, q_da, 'k-', linewidth=1.5)
    ax3.set_title('Ductus Arteriosus (DA)', fontweight='bold', loc='left')
    ax3.set_ylabel('flow rate [l/min]')
    ax3.set_xlabel('cycle time')
    ax3.set_xlim(0, 2)

    # 4. Ductus Venosus (DV) Flow
    ax4.plot(t_normalized, q_dv, 'k-', linewidth=1.5)
    ax4.set_title('Ductus Venosus (DV)', fontweight='bold', loc='left')
    ax4.set_ylabel('flow rate [l/min]')
    ax4.set_xlabel('cycle time')
    ax4.set_xlim(0, 2)

    # 5. Inferior Vena Cava (IVC) Flow
    ax5.plot(t_normalized, q_ivc, 'k-', linewidth=1.5)
    ax5.set_title('Inferior Vena Cava (IVC)', fontweight='bold', loc='left')
    ax5.set_ylabel('flow rate [l/min]')
    ax5.set_xlabel('cycle time')
    ax5.set_xlim(0, 2)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()