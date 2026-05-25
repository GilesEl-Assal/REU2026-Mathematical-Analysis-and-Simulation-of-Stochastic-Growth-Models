import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from generators import generate_lattice_dla
from runtime_analysis import benchmark_dla_scaling
from scipy.spatial.distance import pdist

grid_size=600
num_particles=6000
sticking_prob=1
#dla_result, final_rmax, xorigin, yorigin = generate_lattice_dla(num_particles=num_particles, size=grid_size,sticking_prob=sticking_prob)


def calculate_density_correlation(grid, max_radius):
    # 1. Extract coordinates (Y, X order for correct mapping)
    y_coords, x_coords = np.where(grid == True)
    coords = np.column_stack((x_coords, y_coords))
    N = len(coords)

    # 2. Calculate pairwise distances (returns 1D array of length N*(N-1)/2)
    distances = pdist(coords)

    # 3. Histogram binning (step size dr = 1)
    max_dist = int(np.max(distances))
    bins = np.arange(1, max_dist + 1)
    counts, bin_edges = np.histogram(distances, bins=bins)

    # Calculate the center of each bin
    r = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 4. Normalize to obtain C(r)
    C_r = counts / (N * 2 * np.pi * r)

    # 5. Establish valid bounds for linear regression
    # Ignore lattice artifacts (r < 5)
    # Ignore finite-size edge effects (r > 50% of the maximum fractal radius)
    min_fit_r = 5
    max_fit_r = max_radius * 0.5

    # Isolate the data within the valid bounds
    valid_mask = (C_r > 0) & (r >= min_fit_r) & (r <= max_fit_r)
    log_r_fit = np.log10(r[valid_mask])
    log_Cr_fit = np.log10(C_r[valid_mask])

    # 6. Perform linear regression
    slope, intercept = np.polyfit(log_r_fit, log_Cr_fit, 1)

    # Fractal dimension D_f = slope + d (where d=2 for 2D space)
    Df_calculated = slope + 2

    # Prepare full data arrays for plotting (filtering out empty bins)
    plot_mask = C_r > 0
    r_full = r[plot_mask]
    Cr_full = C_r[plot_mask]

    return r_full, Cr_full, log_r_fit, slope, intercept, Df_calculated, min_fit_r, max_fit_r

#r_data, Cr_data, log_r_fit, slope, intercept, Df, min_r, max_r = calculate_density_correlation(dla_result, final_rmax)





# Define the test intervals (e.g., from 1000 to 10000, stepping by 1000)
test_counts = np.arange(1000, 11000, 500)

N_array, time_array = benchmark_dla_scaling(test_counts, grid_size=400)

log_N = np.log10(N_array)
log_T = np.log10(time_array)

# Fit a line (degree 1 polyfit)
slope, intercept = np.polyfit(log_N, log_T, 1)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
fig.suptitle(f"DLA Runtime Scaling Analysis\nEmpirical Complexity: $mathcal{{O}}(N^{{{slope:.2f}}})$", color='white', fontsize=14)

# --- Left Subplot: Linear Scale ---
ax1.plot(N_array, time_array, marker='o', color='cyan', linewidth=2)
ax1.set_title("Linear Scale (Raw Runtime)", color='white')
ax1.set_xlabel("Particle Count ($N$)", color='white')
ax1.set_ylabel("Execution Time (seconds)", color='white')

# --- Right Subplot: Log-Log Scale ---
ax2.scatter(log_N, log_T, color='white', label='Benchmark Data')
fit_line = slope * log_N + intercept
ax2.plot(log_N, fit_line, color='red', linewidth=2, label=f'Fit: slope = {slope:.2f}')

ax2.set_title("Log-Log Scale (Power Law Fit)", color='white')
ax2.set_xlabel("$log_{10}(N)$", color='white')
ax2.set_ylabel("$log_{10}(Time)$", color='white')
ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')

# --- Global Formatting ---
for ax in [ax1, ax2]:
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()










'''
Fractal Plot


fig, ax = plt.subplots(figsize=(8,8), facecolor='black')
ax.imshow(dla_result, cmap='hot', interpolation='nearest',origin='lower'
          , extent=(0, grid_size, 0, grid_size))
spawn_circle = Circle((xorigin, yorigin), final_rmax+2, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Final Spawn Circle')
ax.add_patch(spawn_circle)

boundary_box = Rectangle((0, 0), grid_size, grid_size, edgecolor='red', facecolor='none', linewidth=3, label='Array Boundary')
ax.add_patch(boundary_box)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
total_particles = np.count_nonzero(dla_result)
param_text = f"Particles: {total_particles}\nSticking Prob: {sticking_prob}\nFinal Radius: {final_rmax}"
ax.text(0.03, 0.97, param_text,
        transform=ax.transAxes,       # Use relative axes coordinates (0 to 1)
        color='white',
        fontsize=10,
        verticalalignment='top',      # Anchor to the top
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))

ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

ax.legend(loc='upper right', facecolor='black', labelcolor='white')

plt.title("Lattice DLA with Boundaries", color='white')
plt.show()
'''





'''

Plot 2: Density-Density Correlation

fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor='black')

# Plot full raw data
ax2.scatter(np.log10(r_data), np.log10(Cr_data), color='white', s=8, alpha=0.6, label='Raw $C(r)$ Data')

# Plot the regression line over the valid bounds
fit_line = slope * log_r_fit + intercept
fit_label = f'Linear Fit (slope $\\approx$ {slope:.3f})\nExtracted $D_f \\approx$ {Df:.3f}'
ax2.plot(log_r_fit, fit_line, color='red', linewidth=2.5, label=fit_label)

# Add vertical markers for the fitting region
ax2.axvline(np.log10(min_r), color='gray', linestyle='--', alpha=0.5)
ax2.axvline(np.log10(max_r), color='gray', linestyle='--', alpha=0.5)

# Formatting
ax2.set_facecolor('black')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('white')
ax2.grid(True, alpha=0.2)

ax2.set_title("Density-Density Correlation $C(r)$", color='white')
ax2.set_xlabel("$log_{10}(r)$", color='white')
ax2.set_ylabel("$log_{10}(C(r))$", color='white')
ax2.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

plt.show()
'''