from readgssi import readgssi
import matplotlib.pyplot as plt
import numpy as np
import os
# pip install readgssi

GPR_file = 'dataset/line_001.DZT'

# Read metadata using readgssi
metadata = readgssi.readgssi(infile=GPR_file, plotting=False)
# print(metadata)
# Select the data arrays from the metadata. The metadata is a list with two elements.
extracted_values = metadata[1][0]
print(f'number of columns (traces): {extracted_values.shape[1]} and number of rows (time): {extracted_values.shape[0]}')
print(f'data type: {extracted_values.dtype}')

# Plot the GPR data
plt.figure(figsize=(15, 10))
plt.imshow(extracted_values, cmap='Greys')
plt.title('GPR at Accounting Department')
plt.xlabel('Trace number')
plt.ylabel('Two-way travel time (ms)')
os.makedirs('figure_plot', exist_ok=True)
plt.savefig('figure_plot/' + 'GPR' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()


