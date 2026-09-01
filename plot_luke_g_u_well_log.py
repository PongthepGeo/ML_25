import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "dataset/well_log.csv"
WELL_NAME = "LUKE G U"

df = pd.read_csv(INPUT_CSV)

well = (
    df[df["Well Name"].astype(str).str.strip().eq(WELL_NAME)]
    .copy()
    .sort_values("Depth")
)

# GR well log
fig, ax = plt.subplots(figsize=(4.8, 9))
ax.plot(well["GR"], well["Depth"], linewidth=1.2)
ax.set_xlabel("Gamma Ray (GR)")
ax.set_ylabel("Depth")
ax.invert_yaxis()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("LUKE_G_U_GR_vs_Depth.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# GR vs resistivity correlation
r = well["GR"].corr(well["ILD_log10"])

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(well["GR"], well["ILD_log10"], s=16, alpha=0.65)
ax.set_xlabel("Gamma Ray (GR)")
ax.set_ylabel("Resistivity (ILD_log10)")
ax.text(
    0.03, 0.97,
    f"Pearson r = {r:.3f}",
    transform=ax.transAxes,
    va="top"
)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("LUKE_G_U_GR_vs_Resistivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Well: {WELL_NAME}")
print(f"Samples: {len(well)}")
print(f"Depth range: {well['Depth'].min()} - {well['Depth'].max()}")
print(f"Pearson correlation GR vs ILD_log10: {r:.4f}")
