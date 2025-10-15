import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import os

image_path = r"C:\PATH\TO\IMAGE"  # << change for your folder with images
output_folder = os.path.join(folder_path, "output_results")
os.makedirs(output_folder, exist_ok=True)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found or invalid.")

# Find all white pixels (edges) MUST BE A WHITE BACKGROUND
ys, xs = np.where(image > 0)
if len(ys) == 0:
    raise ValueError("No white pixels found in edge image.")

# Baseline is the bottom of the image
flat_baseline_y = image.shape[0] - 1
print(f"Baseline Y (bottom of image) = {flat_baseline_y}")

# Extract bottom-most edge
bottom_edge = {}
for x, y in zip(xs, ys):
    if x not in bottom_edge or y > bottom_edge[x]:
        bottom_edge[x] = y

if len(bottom_edge) < 10:
    raise ValueError("Too few points detected for profile.")

x_vals = np.array(sorted(bottom_edge.keys()))
y_vals = np.array([bottom_edge[x] for x in x_vals])

# Convert to Z-height from baseline
z_vals = (flat_baseline_y - y_vals).astype(float)

# Median filter
z_vals = median_filter(z_vals, size=5)

# Remove sharp spikes
dz = np.abs(np.gradient(z_vals))
grad_thresh = 100
z_vals[dz > grad_thresh] = np.nan
valid = ~np.isnan(z_vals)
z_vals = np.interp(np.arange(len(z_vals)), np.flatnonzero(valid), z_vals[valid])
window = 15 if len(z_vals) > 15 else len(z_vals) // 2 * 2 + 1
smoothed_z = savgol_filter(z_vals, window_length=window, polyorder=3)

# Save CSV
csv_name = os.path.splitext(os.path.basename(image_path))[0] + "_zx_curve.csv"
csv_path = os.path.join(output_folder, csv_name)
np.savetxt(csv_path, np.column_stack((x_vals, smoothed_z)),
           delimiter=',', header='X_pixels,Z_pixels', comments='')

# Plot
plt.figure(figsize=(10, 4))
plt.plot(x_vals, smoothed_z, color='red', linewidth=2, label='Smoothed Bottom Curve (Z vs X)')
plt.axhline(0, color='blue', linestyle='--', linewidth=1.5, label='Flat Baseline (Z=0)')
plt.xlabel("X Position (pixels)")
plt.ylabel("Z Height from Baseline (pixels)")
plt.title(f"Z-X Profile from: {os.path.basename(image_path)}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
