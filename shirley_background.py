import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import voigt_profile, erfc
from scipy.optimize import curve_fit

df = pd.read_csv("Nb3d-2_only_data.csv")

# Use first two columns: Binding Energy and Intensity
x_original = df.iloc[:, 0].to_numpy(dtype=float)
y_original = df.iloc[:, 1].to_numpy(dtype=float)

def shirley_background(x, y, fit_min=None, fit_max=None, max_iter=1000, tol=1e-6):
    """
    Classic iterative Shirley background for XPS.

    x: binding energy values
    y: intensity values
    fit_min, fit_max: optional binding energy limits for the active Shirley region.
        The returned background still spans every point in x. Points outside the
        active region are extended from the nearest solved background endpoint.

    Returns background in the same order as input.
    """

    # Sort x from low BE to high BE for consistent integration
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    if fit_min is None:
        fit_min = x_sorted[0]
    if fit_max is None:
        fit_max = x_sorted[-1]

    if fit_min > fit_max:
        fit_min, fit_max = fit_max, fit_min

    fit_mask = (x_sorted >= fit_min) & (x_sorted <= fit_max)
    if fit_mask.sum() < 2:
        raise ValueError("Shirley fit range must contain at least two data points.")

    x_fit = x_sorted[fit_mask]
    y_fit = y_sorted[fit_mask]

    # Endpoint intensities
    y_lowBE = y_fit[0]
    y_highBE = y_fit[-1]

    # Initial background: straight line between endpoints
    bg = np.linspace(y_lowBE, y_highBE, len(y_fit))

    for iteration in range(max_iter):
        bg_old = bg.copy()

        # Net peak intensity
        net = y_fit - bg
        net[net < 0] = 0

        # Cumulative integrated peak area from low BE to each point
        cumulative_area = np.zeros_like(net)

        for i in range(1, len(net)):
            cumulative_area[i] = cumulative_area[i-1] + 0.5 * (
                net[i] + net[i-1]
            ) * abs(x_fit[i] - x_fit[i-1])

        total_area = cumulative_area[-1]

        if total_area == 0:
            break

        # Shirley background:
        # low BE side = y_lowBE
        # high BE side = y_highBE
        bg = y_lowBE + (y_highBE - y_lowBE) * cumulative_area / total_area

        change = np.max(np.abs(bg - bg_old))

        if change < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

    # Extend the solved background across the full spectrum.
    bg_sorted = np.empty_like(y_sorted)
    bg_sorted[x_sorted < x_fit[0]] = bg[0]
    bg_sorted[fit_mask] = bg
    bg_sorted[x_sorted > x_fit[-1]] = bg[-1]

    # Restore original order.
    bg_original_order = np.empty_like(bg_sorted)
    bg_original_order[order] = bg_sorted

    return bg_original_order


SHIRLEY_MIN_BE = 200.0
SHIRLEY_MAX_BE = 212.0

shirley = shirley_background(
    x_original,
    y_original,
    fit_min=SHIRLEY_MIN_BE,
    fit_max=SHIRLEY_MAX_BE,
)
corrected = y_original - shirley

out = pd.DataFrame({
    "BindingEnergy": x_original,
    "RawIntensity": y_original,
    "ShirleyBackground": shirley,
    "CorrectedIntensity": corrected
})

out.to_csv("Nb3d_shirley_corrected.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(x_original, y_original, label="Raw data")
plt.plot(x_original, shirley, label="Shirley background")
plt.plot(x_original, corrected, label="Background-subtracted")

plt.xlabel("Binding Energy (eV)")
plt.ylabel("Intensity")
plt.title("Nb 3d Shirley Background")
plt.gca().invert_xaxis()
plt.legend()
plt.show()

print("Saved: Nb3d_shirley_corrected.csv")

# ======================================================
# PEAK FITTING SECTION — USE AFTER SHIRLEY SUBTRACTION
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("Nb3d_shirley_corrected.csv")

x = df["BindingEnergy"].to_numpy(dtype=float)
y = df["CorrectedIntensity"].to_numpy(dtype=float)

# Only fit Nb 3d region
mask = (x >= 201.0) & (x <= 212.5)
x = x[mask]
y = y[mask]

# Sort increasing
order = np.argsort(x)
x = x[order]
y = y[order]

# Remove negative values after background subtraction
y = y - np.min(y)

# -----------------------------
# Height-normalized pseudo-Voigt
# -----------------------------
def pseudo_voigt(x, height, center, fwhm, eta):
    """
    height = peak height
    center = binding energy
    fwhm = full width at half maximum
    eta = Lorentzian fraction, 0 = Gaussian, 1 = Lorentzian
    """
    sigma = fwhm / 2.35482

    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    lorentzian = 1 / (1 + 4 * ((x - center) / fwhm) ** 2)

    return height * ((1 - eta) * gaussian + eta * lorentzian)

def asymmetric_nb_metal(x, height, center, fwhm, eta, tail):
    """
    Simple asymmetric metallic Nb peak.
    Adds tailing toward higher binding energy.
    """
    peak = pseudo_voigt(x, height, center, fwhm, eta)

    asym = np.ones_like(x)
    high_be = x > center
    asym[high_be] = np.exp(-tail * (x[high_be] - center))

    return peak * asym

# -----------------------------
# Doublets
# -----------------------------
SPLIT = 2.72
RATIO = 2 / 3   # 3d3/2 relative to 3d5/2

def oxide_doublet(x, height, center_5_2, fwhm, eta):
    p_5_2 = pseudo_voigt(x, height, center_5_2, fwhm, eta)
    p_3_2 = pseudo_voigt(x, height * RATIO, center_5_2 + SPLIT, fwhm, eta)
    return p_5_2 + p_3_2

def metal_doublet(x, height, center_5_2, fwhm, eta, tail):
    p_5_2 = asymmetric_nb_metal(x, height, center_5_2, fwhm, eta, tail)
    p_3_2 = asymmetric_nb_metal(x, height * RATIO, center_5_2 + SPLIT, fwhm, eta, tail)
    return p_5_2 + p_3_2

# -----------------------------
# Total model
# -----------------------------
def total_model(x,
                h_m, be_m, fwhm_m, eta_m, tail_m,
                h_nbo, be_nbo, fwhm_nbo, eta_nbo,
                h_nbo2, be_nbo2, fwhm_nbo2, eta_nbo2,
                h_nb2o5, be_nb2o5, fwhm_nb2o5, eta_nb2o5):

    return (
        metal_doublet(x, h_m, be_m, fwhm_m, eta_m, tail_m) +
        oxide_doublet(x, h_nbo, be_nbo, fwhm_nbo, eta_nbo) +
        oxide_doublet(x, h_nbo2, be_nbo2, fwhm_nbo2, eta_nbo2) +
        oxide_doublet(x, h_nb2o5, be_nb2o5, fwhm_nb2o5, eta_nb2o5)
    )

# -----------------------------
# Initial guesses
# -----------------------------
p0 = [
    1000, 202.3, 0.9, 0.3, 0.15,    # Nb metal
    1000, 203.7, 1.1, 0.3,          # NbO
    5000, 206.2, 1.2, 0.3,          # NbO2
    45000, 207.4, 1.2, 0.3          # Nb2O5
]

lower = [
    0, 201.9, 0.3, 0.0, 0.0,
    0, 202.7, 0.3, 0.0,
    0, 205.2, 0.3, 0.0,
    0, 207.0, 0.3, 0.0
]

upper = [
    np.inf, 202.6, 2.5, 1.0, 1.0,
    np.inf, 204.7, 3.0, 1.0,
    np.inf, 207.2, 3.0, 1.0,
    np.inf, 207.8, 3.0, 1.0
]

# -----------------------------
# Fit
# -----------------------------
popt, pcov = curve_fit(
    total_model,
    x,
    y,
    p0=p0,
    bounds=(lower, upper),
    maxfev=200000
)

fit = total_model(x, *popt)

nb_m = metal_doublet(x, popt[0], popt[1], popt[2], popt[3], popt[4])
nbo = oxide_doublet(x, popt[5], popt[6], popt[7], popt[8])
nbo2 = oxide_doublet(x, popt[9], popt[10], popt[11], popt[12])
nb2o5 = oxide_doublet(x, popt[13], popt[14], popt[15], popt[16])

# -----------------------------
# Print results
# -----------------------------
print("\nFIT RESULTS\n")
print(f"Nb metal 3d5/2: {popt[1]:.3f} eV, FWHM = {popt[2]:.3f}")
print(f"NbO      3d5/2: {popt[6]:.3f} eV, FWHM = {popt[7]:.3f}")
print(f"NbO2     3d5/2: {popt[10]:.3f} eV, FWHM = {popt[11]:.3f}")
print(f"Nb2O5    3d5/2: {popt[14]:.3f} eV, FWHM = {popt[15]:.3f}")

print("\nFixed doublet separation = 2.72 eV")
print("Fixed area ratio 3d5/2 : 3d3/2 = 3 : 2")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(11, 7))

plt.plot(x, y, "ko", markersize=4, label="Shirley-corrected data")
plt.plot(x, fit, "r-", linewidth=2.5, label="Total fit")

plt.plot(x, nb_m, "--", linewidth=2, label="Nb metal")
plt.plot(x, nbo, "--", linewidth=2, label="NbO")
plt.plot(x, nbo2, "--", linewidth=2, label="NbO2")
plt.plot(x, nb2o5, "--", linewidth=2, label="Nb2O5")

plt.xlabel("Binding Energy (eV)")
plt.ylabel("Corrected Intensity")
plt.title("Nb 3d Peak Fit")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.show()
