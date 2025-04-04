import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv("")

bold_combinations = {
    ("CYP2D6", "doxepin"),
    ("CYP2D6", "propranolol"),
    ("CYP2D6", "metoprolol"),
    ("CYP2C19", "amitriptyline"),
    ("CYP2C9", "warfarin"),
    ("CYP3A5", "tacrolimus"),
}

df["Label"] = df["CYP_gene"] + " - " + df["drug_name"]

# median
df_sorted = df.sort_values("median_dose", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 14))

# Plot lines and points
for i, row in df_sorted.iterrows():
    ax.plot([row["min_dose"], row["max_dose"]], [i, i], color='gray', lw=2)
    ax.plot(row["median_dose"], i, 'o', color='green')

# Set y-axis tick positions
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels([])  # Remove default tick labels

# Set xscale and draw so layout is known
ax.set_xscale("log")
ax.set_xlabel("Dose (log scale)")
ax.set_title("Drug Doses with Min/Max Range and Median by Gene", fontweight='bold')
ax.invert_yaxis()
ax.grid(False)
plt.tight_layout()

# Now we know the xlim, so we can place text safely
x_text = ax.get_xlim()[0] / 1.5  # slightly to the left of the axis

for i, row in df_sorted.iterrows():
    label = f"{row['CYP_gene']} - {row['drug_name']}"
    is_bold = (row["CYP_gene"], row["drug_name"]) in bold_combinations
    ax.text(
        x_text,
        i,
        label,
        va='center',
        ha='right',
        fontweight='bold' if is_bold else 'normal',
        fontsize=10,
        clip_on=False
    )

# Save and show
plt.savefig("", dpi=300, bbox_inches='tight')

