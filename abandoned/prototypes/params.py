import math

# Resistance [mmHg s/mL]
res_Ao   = 0.05
res_SArt   = 0.95    # systemic arterioles (DCM value: 1.4)
res_SV   = 0.05
res_PA   = 0.01
res_PArt = 0.15
res_PVein  = 0.05
res_MV   = 0.002
res_AV   = 0.002
res_TV   = 0.001
res_PValve   = 0.001

# Inductance [mmHg s^2/mL]
ind_Ao   = 1e-5
ind_SArt   = 1e-5
ind_PA   = 1e-5
ind_PArt = 1e-5

# Capacitance [mL/mmHg]
cap_Ao   = 0.13
cap_SArt   = 1.13
cap_SV   = 19.35
cap_PA   = 3.33
cap_PArt = 0.13
cap_PVein = 19.35

# --- Cardiac Chamber Parameters ---
# T = cardiac cycle period [s]
T = 0.75

# Left Ventricle
l_LV   = 7          # length [cm]
K_LV   = 1.50       # (DCM: 1.40)
V0_LV  = 10         # unstressed volume [mL] (DCM: 17)
Ees_LV = 3.5        # end-systolic elastance [mmHg/mL] (DCM: 1.3)
A_LV   = 1          # (DCM: 0.85)
B_LV   = 0.02
T1_LV  = 0.33 * T   
T2_LV  = 0.45 * T   

# Right Ventricle
l_RV   = 7
K_RV   = 3.25
V0_RV  = 25
Ees_RV = 1.4
A_RV   = 1
B_RV   = 0.02
T1_RV  = 0.33 * T
T2_RV  = 0.45 * T

# Left Atrium
l_LA   = 4.5
K_LA   = 2.50
V0_LA  = 3
Emax_LA = 0.4       # [mmHg/mL]
Emin_LA = 0.2       # [mmHg/mL]
Ta_LA  = 0.8 * T
D_LA   = 0.0375     # [s]

# Right Atrium
l_RA   = 4.5
K_RA   = 2.50
V0_RA  = 3
Emax_RA = 0.4
Emin_RA = 0.2
Ta_RA  = 0.8 * T
D_RA   = 0.0375

# Derived quantities

def _r0_half_ellipsoid(V0, K, l):
    return math.sqrt((6 * V0) / (4 * math.pi * K * l))

def _r0_quarter_ellipsoid(V0, K, l):
    """Invert Eq. 10/11 for Right Ventricle: V = (4/3)*pi*K*r^2*l/4"""
    return math.sqrt((3 * V0) / (math.pi * K * l))

r0_LV = _r0_half_ellipsoid(V0_LV, K_LV, l_LV)
r0_RV = _r0_quarter_ellipsoid(V0_RV, K_RV, l_RV)
r0_LA = _r0_half_ellipsoid(V0_LA, K_LA, l_LA)
r0_RA = _r0_half_ellipsoid(V0_RA, K_RA, l_RA)

def V_LV(r_LV):
    return (4.0 / 3.0) * math.pi * K_LV * (r_LV ** 2) * l_LV / 2.0

def V_RV(r_RV):
    return (4.0 / 3.0) * math.pi * K_RV * (r_RV ** 2) * l_RV / 4.0

def V_LA(r_LA):
    return (4.0 / 3.0) * math.pi * K_LA * (r_LA ** 2) * l_LA / 2.0

def V_RA(r_RA):
    return (4.0 / 3.0) * math.pi * K_RA * (r_RA ** 2) * l_RA / 2.0