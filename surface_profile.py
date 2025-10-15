import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


folder_path = "C:/PATH/TO/IMAGES"  # << change for your folder with images
output_folder = os.path.join(folder_path, "output_results")
os.makedirs(output_folder, exist_ok=True)


image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_file} — not a valid image.")
        continue

    height = image.shape[0]

    # Threshold for yellow-orange region in BGR
    lower_bgr = np.array([0, 100, 150])
    upper_bgr = np.array([120, 255, 255])
    mask = cv2.inRange(image, lower_bgr, upper_bgr)

    # Find bottom-most contour points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"No contours found for {image_file}")
        continue

    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :]  # shape (N, 2)

    bottom_edge = {}
    for x, y in points:
        if x not in bottom_edge or y > bottom_edge[x]:
            bottom_edge[x] = y

    x_vals = np.array(sorted(bottom_edge.keys()))
    y_vals = np.array([bottom_edge[x] for x in x_vals])

    z_vals = height - y_vals
    z_vals -= np.min(z_vals)

    # Smooth curve
    window = 31 if len(z_vals) > 31 else len(z_vals) // 2 * 2 + 1
    smoothed_z = savgol_filter(z_vals, window_length=window, polyorder=3)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, smoothed_z, color='red', linewidth=2, label='Smoothed Bottom Curve (Z vs X)')
    plt.axhline(0, color='gray', linestyle='--', label='Flat Baseline (Z=0)')
    plt.xlim(400,4000)
    plt.ylim(-10,1300)
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Z Height (pixels)")
    plt.title(f"Z-X Profile: {image_file}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    save_name = os.path.splitext(image_file)[0] + "_zx_curve.png"
    save_path = os.path.join(output_folder, save_name)
    plt.savefig(save_path, dpi=300)
    csv_name = os.path.splitext(image_file)[0] + "_zx_curve.csv"
    csv_path = os.path.join(output_folder, csv_name)
    np.savetxt(csv_path, np.column_stack((x_vals, smoothed_z)), delimiter=',', header='X_pixels,Z_pixels', comments='')

    plt.close()
