import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import old.params as pm
import math

# Reduced from 18 to 14 because 4 radius state variables are removed
pm_count = 14

def get_r_LV(V_LV):
    V_LV = max(1e-6, V_LV)
    return math.sqrt((6 * V_LV) / (4 * math.pi * pm.K_LV * pm.l_LV))

def get_r_RV(V_RV):
    V_RV = max(1e-6, V_RV)
    return math.sqrt((3 * V_RV) / (math.pi * pm.K_RV * pm.l_RV))

def get_r_LA(V_LA):
    V_LA = max(1e-6, V_LA)
    return math.sqrt((6 * V_LA) / (4 * math.pi * pm.K_LA * pm.l_LA))

def get_r_RA(V_RA):
    V_RA = max(1e-6, V_RA)
    return math.sqrt((6 * V_RA) / (4 * math.pi * pm.K_RA * pm.l_RA))

def f_act_ventricle(t, T1, T2):
    t = t % pm.T 
    if 0 <= t < T1:
        return (1 - math.cos((t / T1) * math.pi)) / 2
    elif T1 <= t < T2:
        return (1 + math.cos(((t - T1) / (T2 - T1)) * math.pi)) / 2
    else:
        return 0.0

def calc_p_LV(t, V_LV):
    r_LV = get_r_LV(V_LV)
    # Active pressure component
    vol_diff_lv = (4/6) * math.pi * pm.K_LV * pm.l_LV * (r_LV**2 - pm.r0_LV**2)
    p_lv_a = pm.Ees_LV * vol_diff_lv * f_act_ventricle(t, pm.T1_LV, pm.T2_LV)
    
    # Passive pressure component
    p_lv_p = pm.A_LV * math.exp(pm.B_LV * max(0, V_LV)) - 1
    return p_lv_a + p_lv_p

def calc_p_RV(t, V_RV):
    r_RV = get_r_RV(V_RV)
    # Active pressure component 
    vol_diff_rv = (1/3) * math.pi * pm.K_RV * pm.l_RV * (r_RV**2 - pm.r0_RV**2)
    p_rv_a = pm.Ees_RV * vol_diff_rv * f_act_ventricle(t, pm.T1_RV, pm.T2_RV)
    
    # Passive pressure component
    p_rv_p = pm.A_RV * math.exp(pm.B_RV * max(0, V_RV)) - 1
    return p_rv_a + p_rv_p

def f_act_atrium(t, T_a_start):
    t = t % pm.T
    if T_a_start <= t <= pm.T:
        return 1 - math.cos(2 * math.pi * ((t - T_a_start) / (pm.T - T_a_start)))
    else:
        return 0.0
    
def calc_p_LA(t, V_LA):
    r_LA = get_r_LA(V_LA)
    E_la_t = pm.Emin_LA + 0.5 * (pm.Emax_LA - pm.Emin_LA) * f_act_atrium(t - pm.D_LA, pm.Ta_LA)
    vol_diff_la = (4/6) * math.pi * pm.K_LA * pm.l_LA * (r_LA**2 - pm.r0_LA**2)
    return E_la_t * vol_diff_la

def calc_p_RA(t, V_RA):
    r_RA = get_r_RA(V_RA)
    E_ra_t = pm.Emin_RA + 0.5 * (pm.Emax_RA - pm.Emin_RA) * f_act_atrium(t - pm.D_RA, pm.Ta_RA)
    vol_diff_ra = (4/6) * math.pi * pm.K_RA * pm.l_RA * (r_RA**2 - pm.r0_RA**2)
    return E_ra_t * vol_diff_ra

def dQdt(delta_P, R, Q, L):
    return (delta_P - R*Q)/(L)

def derivative(t: float, y: np.ndarray) -> np.ndarray:
    dydt = np.zeros(pm_count, dtype=float)

    # State layout mapping:
    # y[0:6] -> Capacitors: Ao, SA, SV, PA, PArt, PVein
    # y[6:10] -> Inductors: Q_Ao, Q_SA, Q_PA, Q_PArt
    # y[10] -> V_RA, y[11] -> V_RV, y[12] -> V_LA, y[13] -> V_LV

    p_Ao = y[0]/pm.cap_Ao
    p_SA = y[1]/pm.cap_SArt
    p_SV = y[2]/pm.cap_SV
    
    p_RA = calc_p_RA(t, y[10])
    p_RV = calc_p_RV(t, y[11])
    p_PA = y[3]/pm.cap_PA
    p_Art = y[4]/pm.cap_PArt
    p_PVein = y[5]/pm.cap_PVein
    p_LA = calc_p_LA(t, y[12])
    p_LV = calc_p_LV(t, y[13])

    # Aorta to Systemic Circulation
    dydt[6] = dQdt(p_Ao - p_SA, pm.res_Ao, y[6], pm.ind_Ao)
    dydt[1] = y[6] - y[7]
    
    dydt[7] = dQdt(p_SA - p_SV, pm.res_SArt, y[7], pm.ind_SArt)
    Q_RA = (p_SV - p_RA) / pm.res_SV
    dydt[2] = y[7] - Q_RA
   
    # Right Atrium & Tricuspid Valve
    Q_TV = max(0, (p_RA - p_RV) / pm.res_TV)
    dydt[10] = Q_RA - Q_TV

    # Right Ventricle & Pulmonary Valve
    Q_PValve = max(0, (p_RV - p_PA) / pm.res_PValve)
    dydt[11] = Q_TV - Q_PValve

    # Pulmonary Circulation
    dydt[3] = Q_PValve - y[8]
    dydt[8] = dQdt(p_PA - p_Art, pm.res_PA, y[8], pm.ind_PA)

    dydt[4] = y[8] - y[9]
    dydt[9] = dQdt(p_Art - p_PVein, pm.res_PArt, y[9], pm.ind_PArt)

    # Pulmonary Veins to Left Atrium
    Q_LA = (p_PVein - p_LA) / pm.res_PVein

    # Left Atrium & Mitral Valve
    Q_MV = max(0, (p_LA - p_LV) / pm.res_MV)
    dydt[12] = Q_LA - Q_MV

    # Left Ventricle & Aortic Valve
    Q_AV = max(0, (p_LV - p_Ao) / pm.res_AV)
    dydt[13] = Q_MV - Q_AV

    dydt[0] = Q_AV - y[6]

    return dydt

if __name__ == "__main__":
    y0 = np.zeros(pm_count)
    
    # 1. Vascular Volume Capacitors
    y0[0] = pm.cap_Ao * 65 
    y0[1] = 55 
    y0[2] = 1264.6 
    y0[3] = pm.cap_PA * 15 
    y0[4] = 5 
    y0[5] = 388 

    # 2. Flow Rates Inductors
    y0[6] = 10 
    y0[7] = 10 
    y0[8] = 10 
    y0[9] = 10 

    # 3. Cardiac Chamber Volumes Only
    y0[10] = 20  # V_RA
    y0[11] = 96  # V_RV
    y0[12] = 20  # V_LA
    y0[13] = 93  # V_LV

    indices_to_scale = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
    y0[indices_to_scale] *= 1.0/2.5

    num_cycles = 15
    t0, t_end = 0.0, num_cycles * pm.T
    
    sol = solve_ivp(
        derivative, 
        (t0, t_end), 
        y0, 
        method='BDF', 
        rtol=1e-1, 
        max_step=1e-1
    )

    ts = sol.t
    y_vals = sol.y.T

    # Extracting integrated volume states
    V_RA, V_RV, V_LA, V_LV = y_vals[:, 10], y_vals[:, 11], y_vals[:, 12], y_vals[:, 13]

    # Dynamically tracking derived pressures and diameters algebraically
    p_Ao = y_vals[:, 0] / pm.cap_Ao
    p_PA = y_vals[:, 3] / pm.cap_PA
    p_LV = np.array([calc_p_LV(t, v) for t, v in zip(ts, V_LV)])
    p_LA = np.array([calc_p_LA(t, v) for t, v in zip(ts, V_LA)])
    p_RV = np.array([calc_p_RV(t, v) for t, v in zip(ts, V_RV)])
    p_RA = np.array([calc_p_RA(t, v) for t, v in zip(ts, V_RA)])

    D_RA = np.array([2 * get_r_RA(v) for v in V_RA])
    D_RV = np.array([ get_r_RV(v) for v in V_RV])  # Note: 2*r matching standard mid-diameter approximations
    D_LA = np.array([2 * get_r_LA(v) for v in V_LA])
    D_LV = np.array([2 * get_r_LV(v) for v in V_LV])

    # Get last cardiac cycle data slice
    last_cycle_start = (num_cycles - 1) * pm.T
    idx = np.where(ts >= last_cycle_start)[0]
    t_plot = ts[idx] - last_cycle_start

    # Plot Reconfiguration
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    def format_ax(ax, title, y_lim, y_label):
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlim([0, pm.T])
        ax.set_xticks([0, 0.37, 0.75])
        ax.set_ylim(y_lim)
        ax.set_xlabel('Time [s]', fontweight='bold', fontsize=12)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', frameon=False, fontsize=11, handlelength=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)

    # e: Left Heart Pressures
    axs[0].plot(t_plot, p_LV[idx], 'b-', linewidth=2, label='P_lv')
    axs[0].plot(t_plot, p_Ao[idx], 'r-', linewidth=2, label='P_ao')
    axs[0].plot(t_plot, p_LA[idx], 'g-', linewidth=2, label='P_la')
    format_ax(axs[0], 'e', [0, 140], 'Pressure [mmHg]')

    # f: Right Heart Pressures
    axs[1].plot(t_plot, p_RV[idx], 'b-', linewidth=2, label='P_rv')
    axs[1].plot(t_plot, p_PA[idx], 'r-', linewidth=2, label='P_po')
    axs[1].plot(t_plot, p_RA[idx], 'g-', linewidth=2, label='P_ra')
    format_ax(axs[1], 'f', [0, 40], 'Pressure [mmHg]')

    # g: Chamber Volumes
    axs[2].plot(t_plot, V_LV[idx], 'b-', linewidth=2, label='V_lv')
    axs[2].plot(t_plot, V_LA[idx], 'r-', linewidth=2, label='V_la')
    axs[2].plot(t_plot, V_RV[idx], 'g-', linewidth=2, label='V_rv')
    axs[2].plot(t_plot, V_RA[idx], 'purple', linewidth=2, label='V_ra')
    format_ax(axs[2], 'g', [0, 140], 'Volume [mL]')

    # h: Chamber Diameters
    axs[3].plot(t_plot, D_LV[idx], 'b-', linewidth=2, label='D_lv')
    axs[3].plot(t_plot, D_LA[idx], 'r-', linewidth=2, label='D_la')
    axs[3].plot(t_plot, D_RV[idx], 'g-', linewidth=2, label='D_rv')
    axs[3].plot(t_plot, D_RA[idx], 'purple', linewidth=2, label='D_ra')
    format_ax(axs[3], 'h', [1, 6], 'Diameter [cm]')

    plt.tight_layout()
    out = 'hemodynamic_cycles_fixed2.png'
    plt.savefig(out, dpi=300)
    print(f'Plot saved to {out}')