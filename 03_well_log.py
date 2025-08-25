import pandas as pd
import matplotlib.pyplot as plt
import os
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
figure = plt.figure(figsize=(5, 18))
# plt.plot(value of axis-X, value of axis-Y)
plt.plot(GR, one_log_name.Depth, label='GR', color='salmon')
plt.gca().invert_yaxis()
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel('Amplitude')
plt.ylabel('Depth (m)')
plt.title('Gamma Ray Well Log')
# Save file, ensure that folder 'data_out' is available.
os.makedirs('figure_plot', exist_ok=True)
plt.savefig('figure_plot/' + 'well_log' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()