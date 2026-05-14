from scipy.spatial.distance import pdist
import numpy as np
from generator import generate_lattice_dla
import matplotlib.pyplot as plt

def fractal_density(num_particles, sticking_prob, size, num_slices):
    # 1. Extract coordinates of all particles
    dla_result, rmax, cx,cy = generate_lattice_dla(num_particles, sticking_prob, size)
    x_coords,y_coords = np.where(dla_result == True)
    distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    density_list = []
    radius_list = []
    max_eval_radius = rmax*.9
    step_size = max_eval_radius / num_slices

    for i in range(1, num_slices+1):
        radius = i * step_size
        particles_in_radius = np.sum(distances <= radius)
        area = np.pi * (radius**2)
        density = particles_in_radius / area
        density_list.append(density)
        radius_list.append(radius)

    return np.array(radius_list), np.array(density_list)