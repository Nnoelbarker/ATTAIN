import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# Load Excel Data
# -----------------------------
file_path = r"C:\Users\mbznn\PycharmProjects\ATTAIN\ATTAIN RAW DATA.xlsx"
df = pd.read_excel(file_path)


# -----------------------------
# Function to calculate SEM and MDC
# -----------------------------
def sem_mdc(data1, data2):
    # Calculate differences
    diff = data1 - data2
    # ICC for SEM calculation
    icc = pg.intraclass_corr(data=pd.DataFrame({'rater1': data1, 'rater2': data2}),
                             targets=np.arange(len(data1)),
                             raters=['rater1', 'rater2'],
                             ratings='rater1').iloc[2]['ICC']  # Use ICC2 for consistency
    # Standard deviation of differences
    sd_diff = np.std(diff, ddof=1)
    # SEM = SD * sqrt(1 - ICC)
    sem = sd_diff * np.sqrt(1 - icc)
    # MDC = SEM * 1.96 * sqrt(2)
    mdc = sem * 1.96 * np.sqrt(2)
    return sem, mdc


# -----------------------------
# Bland-Altman Plot Function
# -----------------------------
def bland_altman_plot(data1, data2, title):
    mean_vals = (data1 + data2) / 2
    diff_vals = data1 - data2
    md = np.mean(diff_vals)
    sd = np.std(diff_vals, ddof=1)

    plt.figure(figsize=(6, 4))
    plt.scatter(mean_vals, diff_vals, color='black')
    plt.axhline(md, color='black', linestyle='-')
    plt.axhline(md + 1.96 * sd, color='black', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='black', linestyle='--')
    plt.xlabel('Mean', fontname='Times New Roman')
    plt.ylabel('Difference', fontname='Times New Roman')
    plt.title(title, fontname='Times New Roman')
    plt.grid(False)
    plt.show()


# -----------------------------
# Intra-rater: M1 vs M3
# -----------------------------
m1 = df['M1']
m3 = df['M3']

icc_intra = pg.intraclass_corr(data=pd.DataFrame({'rater1': m1, 'rater2': m3}),
                               targets=np.arange(len(m1)),
                               raters=['rater1', 'rater2'],
                               ratings='rater1')
sem_intra, mdc_intra = sem_mdc(m1, m3)

print("Intra-rater ICC:\n", icc_intra[['Type', 'ICC', 'CI95%']])
print(f"Intra-rater SEM: {sem_intra:.3f}")
print(f"Intra-rater MDC: {mdc_intra:.3f}")

bland_altman_plot(m1, m3, "Intra-rater Bland-Altman (M1 vs M3)")

# -----------------------------
# Inter-rater: M1 vs M2
# -----------------------------
m2 = df['M2']

icc_inter = pg.intraclass_corr(data=pd.DataFrame({'rater1': m1, 'rater2': m2}),
                               targets=np.arange(len(m1)),
                               raters=['rater1', 'rater2'],
                               ratings='rater1')
sem_inter, mdc_inter = sem_mdc(m1, m2)

print("Inter-rater ICC:\n", icc_inter[['Type', 'ICC', 'CI95%']])
print(f"Inter-rater SEM: {sem_inter:.3f}")
print(f"Inter-rater MDC: {mdc_inter:.3f}")

bland_altman_plot(m1, m2, "Inter-rater Bland-Altman (M1 vs M2)")
