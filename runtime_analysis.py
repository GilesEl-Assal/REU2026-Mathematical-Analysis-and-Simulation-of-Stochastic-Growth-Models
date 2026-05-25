import cProfile
import pstats
import io
import time
import numpy as np
from generators import generate_lattice_dla


def benchmark_dla_scaling(particle_counts, grid_size=400):
    runtimes = []
    
    
    for n in particle_counts:
        # Start the hardware clock
        start_time = time.perf_counter()
        
        # Execute the simulation
        
        _ = generate_lattice_dla(num_particles=n, size=grid_size, sticking_prob=1.0)
        
        # Stop the clock
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        runtimes.append(elapsed_time)
        
        
    return np.array(particle_counts), np.array(runtimes)


def generate_profile_file(particles=8000, grid_size=300):
    print(f"--- Generating cProfile data for N={particles} ---")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute the simulation
    _ = generate_lattice_dla(num_particles=particles, size=grid_size, sticking_prob=1.0)
    
    profiler.disable()
    
    # put the raw data into a file
    output_filename = "dla_compute.prof"
    profiler.dump_stats(output_filename)
    
    print(f"Profile saved to {output_filename}. Run 'snakeviz {output_filename}' in terminal.")
















'''
def fractal_density(num_particles, sticking_prob, size, num_slices):
    # 1. Extract coordinates of all particles
    dla_result, rmax, cx,cy = generate_lattice_dla(num_particles, sticking_prob, size)
    x_coords,y_coords = np.where(dla_result == True)
    distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    density_list = []
    radius_list = []
    min_eval_radius = rmax *.1
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
    '''