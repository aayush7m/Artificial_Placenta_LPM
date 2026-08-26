import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats

# ============================================================
# USER SETTING: Percentage of outliers to remove (per series)
# e.g. 0.05 removes the most extreme 5% of rows by |z-score|
# of the calculated pressure drop value. Set to 0 to disable.
# ============================================================
OUTLIER_REMOVAL_FRACTION = 0.05

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

# 2. Calculate Pressure Drops on the full dataset first
df['ecmo_p_drop']       = df['Post-ECMO Pressure '] - df['Pre-ECMO Pressure']
df['restrictor_p_drop'] = df['Post-Restrictor Pressure'] - df['Post-ECMO Pressure ']

# 3. Remove outliers based on z-score of each pressure drop series
def remove_outliers_by_zscore(dataframe, col, fraction):
    """
    Remove rows whose |z-score| on `col` falls in the top `fraction`
    of the distribution (i.e. the most extreme values are dropped).
    Z-scores are computed on the derived pressure-drop column, not
    on the individual raw pressure readings.
    Returns the filtered dataframe and the z-score threshold used.
    """
    if fraction <= 0:
        return dataframe, None

    z_scores = np.abs(stats.zscore(dataframe[col].values))
    # Determine the z-score threshold that cuts off the top `fraction`
    threshold = np.quantile(z_scores, 1.0 - fraction)
    mask = z_scores <= threshold
    n_removed = (~mask).sum()
    print(f"[{col}] z-score threshold: {threshold:.4f} | "
          f"rows removed: {n_removed} / {len(dataframe)}")
    return dataframe[mask].reset_index(drop=True), threshold

df_ecmo, z_thresh_ecmo = remove_outliers_by_zscore(
    df, 'ecmo_p_drop', OUTLIER_REMOVAL_FRACTION
)
df_rest, z_thresh_rest = remove_outliers_by_zscore(
    df, 'restrictor_p_drop', OUTLIER_REMOVAL_FRACTION
)

# 4. Extract arrays for fitting
flow_ecmo = df_ecmo['Flow_mL_s'].values
ecmo_p_drop = df_ecmo['ecmo_p_drop'].values

flow_rest = df_rest['Flow_mL_s'].values
restrictor_p_drop = df_rest['restrictor_p_drop'].values

# Increase default matplotlib font size by ~6 points for clearer plots
mpl.rcParams.update({'font.size': mpl.rcParams.get('font.size', 12) + 6})

# 5. Fit a linear (degree 1 polynomial) to both sets of data
p_ecmo = np.polyfit(flow_ecmo, ecmo_p_drop, 1)
poly_ecmo = np.poly1d(p_ecmo)

p_rest = np.polyfit(flow_rest, restrictor_p_drop, 1)
poly_rest = np.poly1d(p_rest)

# Generate arrays of flow values to draw smooth fit lines
flow_line_ecmo = np.linspace(flow_ecmo.min(), flow_ecmo.max(), 100)
flow_line_rest = np.linspace(flow_rest.min(), flow_rest.max(), 100)

# Build subtitle noting outlier removal setting
if OUTLIER_REMOVAL_FRACTION > 0:
    subtitle = f"Outliers removed: {OUTLIER_REMOVAL_FRACTION * 100:.1f}% per series (by pressure-drop z-score)"
else:
    subtitle = "No outlier removal applied"

# 6. Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(subtitle, fontsize=mpl.rcParams['font.size'] - 2, style='italic')

# Plot 1: ECMO Pressure Drop
ax1.scatter(flow_ecmo, ecmo_p_drop, label='Data', alpha=0.5, color='blue', s=10)
ax1.plot(flow_line_ecmo, poly_ecmo(flow_line_ecmo), color='red',
         label=f'Fit: y = {p_ecmo[0]:.4f}x + {p_ecmo[1]:.4f}')
ax1.set_xlabel('Flow Rate (mL/s)')
ax1.set_ylabel('ECMO P Drop (mmHg)')
ax1.set_title('ECMO Pressure Drop vs Flow Rate')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Restrictor Pressure Drop
ax2.scatter(flow_rest, restrictor_p_drop, label='Data', alpha=0.5, color='green', s=10)
ax2.plot(flow_line_rest, poly_rest(flow_line_rest), color='orange',
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