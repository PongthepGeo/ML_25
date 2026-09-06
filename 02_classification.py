import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from lib.control_plot import PLOT_PARAMS  # shared global plot style
from lib.util import plot_classification

matplotlib.rcParams.update(PLOT_PARAMS)

# Output folder (one folder per script, named after the script)
OUTDIR = Path("02_classification")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load CSV
df = pd.read_csv("dataset/well_log.csv")

# Filter well NEWBY
df_newby = df[df["Well Name"] == "NEWBY"]

# Extract Facies 2 and 4
df_f2 = df_newby[df_newby["Facies"] == 2].sort_values(by="Depth")
df_f4 = df_newby[df_newby["Facies"] == 4].sort_values(by="Depth")

# Plot
plot_classification(df_f2, df_f4, OUTDIR)
