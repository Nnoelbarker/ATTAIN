import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pingouin import intraclass_corr
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_excel("Attain_data_2025.xlsx")

#Bland-Altman Plot Function
def bland_altman_plot(data1, data2, ax, title="Bland-Altman", ylabel="Difference"):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)            # mean difference
    sd = np.std(diff, axis=0)     # standard deviation of difference


    ax.scatter(mean, diff, color='black', alpha=0.6, edgecolor='black', s=50)
    ax.axhline(md, color='black', linestyle='--', linewidth=1.5, label="Mean difference")
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=1, label="±1.96 SD")
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=1)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Mean of Measurements", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(False)

#ICC
icc_data = pd.melt(
    df[["Participant", "M1", "M2", "M3"]],
    id_vars=["Participant"],
    var_name="Rater",
    value_name="Score"
)

# Intrarater ICC (M1 vs M3)
icc_intra = intraclass_corr(
    data=icc_data[icc_data["Rater"].isin(["M1", "M3"])],
    targets="Participant", raters="Rater", ratings="Score"
).round(3)

# Interrater ICC (M1 vs M2)
icc_inter = intraclass_corr(
    data=icc_data[icc_data["Rater"].isin(["M1", "M2"])],
    targets="Participant", raters="Rater", ratings="Score"
).round(3)

print("Intra-rater ICC (M1 vs M3):")
print(icc_intra)
print("\nInter-rater ICC (M1 vs M2):")
print(icc_inter)

#SEM and MDC
def calc_sem_mdc(data1, data2, icc_df, icc_type='ICC2'):
    """
    Calculate SEM and MDC from raw scores and ICC.
    icc_type: column in ICC DataFrame to use, e.g., 'ICC2'
    """
    icc_val = icc_df.loc[icc_df['Type'] == icc_type, 'ICC'].values[0]
    diff = data1 - data2
    sd_diff = diff.std(ddof=1)  # sample SD
    sem = sd_diff * np.sqrt(1 - icc_val)
    mdc = sem * 1.96 * np.sqrt(2)
    return sem, mdc

# Intra-rater (M1 vs M3)
sem_intra, mdc_intra = calc_sem_mdc(df['M1'], df['M3'], icc_intra)
print(f"Intra-rater SEM (M1 vs M3): {sem_intra:.3f}")
print(f"Intra-rater MDC (M1 vs M3): {mdc_intra:.3f}")

# Inter-rater (M1 vs M2)
sem_inter, mdc_inter = calc_sem_mdc(df['M1'], df['M2'], icc_inter)
print(f"Inter-rater SEM (M1 vs M2): {sem_inter:.3f}")
print(f"Inter-rater MDC (M1 vs M2): {mdc_inter:.3f}")


#Bland-Altman Figures
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Intrarater
bland_altman_plot(
    df["M1"], df["M3"], axes[0],
    title="A. Intra-rater (M1 vs M3)",
    ylabel="Difference between Examiner A's measurements (mm²)"
)

# Interrater
bland_altman_plot(
    df["M1"], df["M2"], axes[1],
    title="B. Inter-rater (M1 vs M2)",
    ylabel="Difference between Examiner A and Examiner B measurements (mm²)"
)

plt.tight_layout()
plt.savefig("figures/bland_altman.png", dpi=300)
plt.close()


#Histogram
data = df["Med_lat"].dropna()
plt.figure(figsize=(8, 5))

#histogram frequency
counts, bins, patches = plt.hist(data, bins=7, alpha=0.6, edgecolor="black", color='gray', label='Data')

#x=1 line
plt.axvline(1, color="black", linestyle="--", linewidth=1.5, label="x=1")

#normal distribution
mu, std = stats.norm.fit(data)
x = np.linspace(bins[0], bins[-1], 100)
bin_width = bins[1] - bins[0]
p = stats.norm.pdf(x, mu, std) * len(data) * bin_width  # scale to histogram counts
plt.plot(x, p, 'k', linewidth=2, label="Normal distribution")

#labels
plt.xlabel("Size of medial CSA as a factor compared to lateral CSA", fontsize=10)
plt.ylabel("Frequency", fontsize=10)
plt.legend()
plt.tight_layout()


plt.savefig("figures/Med_lat_hist.png", dpi=300)
plt.close()
