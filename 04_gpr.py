from readgssi import readgssi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from lib.control_plot import PLOT_PARAMS  # shared global plot style
# pip install readgssi

matplotlib.rcParams.update(PLOT_PARAMS)

# Output folder (one folder per script, named after the script)
OUTDIR = Path("04_gpr")
OUTDIR.mkdir(parents=True, exist_ok=True)

GPR_file = 'dataset/line_001.DZT'

# Read metadata using readgssi
metadata = readgssi.readgssi(infile=GPR_file, plotting=False)
# print(metadata)
# Select the data arrays from the metadata. The metadata is a list with two elements.
extracted_values = metadata[1][0]

print(extracted_values.shape)

# top = extracted_values[0:100, :]
# print(top.shape)
# plt.imshow(top, cmap='Greys')
# plt.show()
print(f'number of columns (traces): {extracted_values.shape[1]} and number of rows (time): {extracted_values.shape[0]}')
print(f'data type: {extracted_values.dtype}')

signal = np.zeros((extracted_values.shape[0], extracted_values.shape[1]))
signal[100:, :] = extracted_values[100:, :]

# Radargram (global figure.figsize default)
plt.figure()
plt.imshow(signal, cmap='Greys', aspect='auto')
plt.title('GPR radargram (top mute applied)')
plt.xlabel('Trace number')
plt.ylabel('Two-way travel time (sample)')
plt.tight_layout()
fig_section = OUTDIR / 'gpr_section.png'
plt.savefig(fig_section, format='png', bbox_inches='tight')
plt.show()

# Single trace
trace = signal[:, 400]
axis_x = np.arange(0, trace.shape[0])
# Tall/narrow single trace -- intentionally not the global 16:9 default
plt.figure(figsize=(5, 10))
plt.plot(trace, axis_x)
plt.gca().invert_yaxis()
plt.title('Single trace (#400)')
plt.xlabel('Amplitude')
plt.ylabel('Two-way travel time (sample)')
plt.tight_layout()
fig_trace = OUTDIR / 'gpr_trace.png'
plt.savefig(fig_trace, format='png', bbox_inches='tight')
plt.show()

print('Saved figures:')
print(' -', fig_section)
print(' -', fig_trace)
# # Plot the raw GPR data (no top mute)
# plt.figure()
# plt.imshow(extracted_values, cmap='Greys', aspect='auto')
# plt.title('GPR at Accounting Department')
# plt.xlabel('Trace number')
# plt.ylabel('Two-way travel time (ms)')
# plt.tight_layout()
# plt.savefig(OUTDIR / 'gpr_raw.png', format='png', bbox_inches='tight')
# plt.show()
