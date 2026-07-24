import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load CSV
df = pd.read_csv("Nb3d-2_only_data.csv")

# Use first two columns
x = df.iloc[:, 0].to_numpy()
y = df.iloc[:, 1].to_numpy()

# Find major peaks
peaks, properties = find_peaks(
    y,
    prominence=5000,
    distance=5
)

print("\nPossible Peaks Detected:\n")

for i, peak_idx in enumerate(peaks):
    print(
        f"Peak {i+1}: "
        f"Binding Energy = {x[peak_idx]:.2f} eV, "
        f"Intensity = {y[peak_idx]:.2f}"
    )

# Exact peak values from original dataset
peak_table = pd.DataFrame({
    "Peak #": range(1, len(peaks) + 1),
    "x_value": x[peaks],
    "y_value": y[peaks],
    "prominence": properties["prominences"]
})

print(peak_table)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Data")
plt.scatter(x[peaks], y[peaks], color="red", label="Detected peaks")

for p in peaks:
    plt.text(x[p], y[p], f"({x[p]}, {y[p]:.0f})", ha="center", va="bottom")

plt.xlabel("Binding Energy")
plt.ylabel("Intensity")
plt.title("Detected Peaks")
plt.gca().invert_xaxis()  # XPS convention
plt.legend()
plt.show()