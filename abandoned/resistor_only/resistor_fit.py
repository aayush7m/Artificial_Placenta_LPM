import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 1. Read the data
df = pd.read_csv('ECMO_Trial_2024-03-05_cleaned_v2.csv', header=0)

# The first row contains units (mL/min, °C, mmHg, etc.), so we drop it
df = df.drop(index=0).reset_index(drop=True)

# Note the exact names based on the file snippet, including the space in 'Post-ECMO Pressure '
cols_to_convert = [
    'Flow', 
    'Pre-ECMO Pressure', 
    'Post-ECMO Pressure ', 
    'Post-Restrictor Pressure'
]

# Convert columns to numeric, handling any errors/coercion
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any missing values in these columns to cleanly fit the curves
df = df.dropna(subset=cols_to_convert)

# Reject rows with Flow < 100 or > 250
# This removes measurements outside the valid flow range for analysis
df = df[(df['Flow'] >= 100) & (df['Flow'] <= 250)].reset_index(drop=True)

# Convert Flow from mL/min to mL/s for all subsequent calculations and plotting
# 1 mL/min = 1/60 mL/s
df['Flow_mL_s'] = df['Flow'] / 60.0

# 2. Extract values and calculate Pressure Drops
flow = df['Flow_mL_s'].values

# Increase default matplotlib font size by ~6 points for clearer plots
mpl.rcParams.update({'font.size': mpl.rcParams.get('font.size', 12) + 6})

# ECMO P drop = Post - Pre
ecmo_p_drop = df['Post-ECMO Pressure '].values - df['Pre-ECMO Pressure'].values

# Restrictor P drop = Post Restrictor - Post ECMO
restrictor_p_drop = df['Post-Restrictor Pressure'].values - df['Post-ECMO Pressure '].values

# 3. Fit a linear (degree 1 polynomial) to both sets of data
p_ecmo = np.polyfit(flow, ecmo_p_drop, 1)
poly_ecmo = np.poly1d(p_ecmo)

p_rest = np.polyfit(flow, restrictor_p_drop, 1)
poly_rest = np.poly1d(p_rest)

# Generate an array of flow values to draw smooth fit lines
flow_line = np.linspace(flow.min(), flow.max(), 100)

# 4. Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: ECMO Pressure Drop
ax1.scatter(flow, ecmo_p_drop, label='Data', alpha=0.5, color='blue', s=10)
ax1.plot(flow_line, poly_ecmo(flow_line), color='red', 
         label=f'Fit: y = {p_ecmo[0]:.4f}x + {p_ecmo[1]:.4f}')
ax1.set_xlabel('Flow Rate (mL/s)')
ax1.set_ylabel('ECMO P Drop (mmHg)')
ax1.set_title('ECMO Pressure Drop vs Flow Rate')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Restrictor Pressure Drop
ax2.scatter(flow, restrictor_p_drop, label='Data', alpha=0.5, color='green', s=10)
ax2.plot(flow_line, poly_rest(flow_line), color='orange', 
         label=f'Fit: y = {p_rest[0]:.4f}x + {p_rest[1]:.4f}')
ax2.set_xlabel('Flow Rate (mL/s)')
ax2.set_ylabel('Restrictor P Drop (mmHg)')
ax2.set_title('Restrictor Pressure Drop vs Flow Rate')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('pressure_drops_fit.png', dpi=300)
plt.show()