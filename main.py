import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from generator import generate_lattice_dla
from dimensional_analysis import fractal_density

grid_size=300
num_particles=1000
sticking_prob=0.1
dla_result, final_rmax, xorigin, yorigin = generate_lattice_dla(num_particles=num_particles, size=grid_size,sticking_prob=sticking_prob)

'''
Fractal Plot
'''


fig, ax = plt.subplots(figsize=(8,8), facecolor='black')
ax.imshow(dla_result, cmap='hot', interpolation='nearest',origin='lower'
          , extent=(0, grid_size, 0, grid_size))
spawn_circle = Circle((xorigin, yorigin), final_rmax, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Final Spawn Circle')
ax.add_patch(spawn_circle)

boundary_box = Rectangle((0, 0), grid_size, grid_size, edgecolor='red', facecolor='none', linewidth=3, label='Array Boundary')
ax.add_patch(boundary_box)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)

ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

ax.legend(loc='upper right', facecolor='black', labelcolor='white')
plt.title("Lattice DLA with Boundaries", color='white')
plt.show()
'''
Density Plot
'''
radius_array, density_array = fractal_density(num_particles, sticking_prob, grid_size, 50)


fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(np.log10(radius_array), np.log10(density_array), marker='o', linestyle='-', color='white')

ax.set_title("Fractal Density Profile", color='white')
ax.set_xlabel("$\log_{10}(Radius)$", color='white')
ax.set_ylabel("$\log_{10}(Density)$", color='white')

# Formatting for dark background
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
ax.grid(True, alpha=0.2)

plt.show()