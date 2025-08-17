import pandas as pd
import matplotlib.pyplot as plt
from lib.util import plot_classification

# Load CSV
df = pd.read_csv("dataset/well_log.csv")

# Filter well NEWBY
df_newby = df[df["Well Name"] == "NEWBY"]

# Extract Facies 2 and 4
df_f2 = df_newby[df_newby["Facies"] == 2].sort_values(by="Depth")
df_f4 = df_newby[df_newby["Facies"] == 4].sort_values(by="Depth")

# Plot
plot_classification(df_f2, df_f4)