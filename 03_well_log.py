import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from lib.control_plot import PLOT_PARAMS  # shared global plot style

matplotlib.rcParams.update(PLOT_PARAMS)

# Output folder (one folder per script, named after the script)
OUTDIR = Path("03_well_log")
OUTDIR.mkdir(parents=True, exist_ok=True)
# Install matplotlib using:
# python3 -m pip install matplotlib
# .pyplot is a module in the matplotlib package that provides an interactive interface.
# as plt is an alias for the module, it is used to refer to the module in the code.

df = pd.read_csv('dataset/well_log.csv')
# print(df.columns) # Display the columns of the dataframe
# Ensure that well logs sort by depth from shallow to deep.
df = df.sort_values(by='Depth', ascending=True)
# Select one well log data.
one_log_name = df[df['Well Name'] == 'SHRIMPLIN']
# print(one_log_name)
# Select one well log data.
GR = one_log_name.GR # one_log_name.['GR'] --> same command
# print(GR)

# Data visualization
# Define frame size: figsize=(axis-X, axis-Y)
# Tall/narrow depth track -- intentionally not the global 16:9 default.
figure = plt.figure(figsize=(5, 18))
# plt.plot(value of axis-X, value of axis-Y)
plt.plot(GR, one_log_name.Depth, label='GR', color='salmon')
plt.gca().invert_yaxis()
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel('Amplitude')
plt.ylabel('Depth (m)')
plt.title('Gamma Ray Well Log')
# Save file into this script's output folder.
plt.tight_layout()
fig_log = OUTDIR / 'well_log.png'
plt.savefig(fig_log, format='png', bbox_inches='tight')
print('Saved figure:', fig_log)
plt.show()
