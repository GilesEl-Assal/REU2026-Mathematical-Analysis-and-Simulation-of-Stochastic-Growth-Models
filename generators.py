import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def generate_lattice_dla(num_particles, sticking_prob, size):
    grid = np.zeros((size, size), dtype=bool)
    cx, cy = size // 2, size // 2
    grid[cx, cy] = True  # seed

    rmax = 1

    for _ in range(num_particles):
        r_spawn = rmax + 2
        r_kill_sq = (r_spawn + 5) ** 2

        if r_spawn + 5 >= size // 2:
            break

        angle = np.random.uniform(0, 2 * np.pi)
        x = round(cx + r_spawn * np.cos(angle))
        y = round(cy + r_spawn * np.sin(angle))

        while True:
            # Check kill boundary (particle wandered too far)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            if x <= 0 or x >= size - 1 or y <= 0 or y >= size - 1 or dist_sq > r_kill_sq:
                angle = np.random.uniform(0, 2 * np.pi)
                x = round(cx + r_spawn * np.cos(angle))
                y = round(cy + r_spawn * np.sin(angle))
                continue

            # if particle is outside spawn circle, up the step size to get it back faster
            current_dist = np.sqrt(dist_sq)

            if current_dist > r_spawn + 1:
                leap_distance = current_dist - r_spawn
                leap_angle = np.random.uniform(0, 2 * np.pi)

                # can use angles because the pdf of random walk exit point converges to a uniform continuous distribution
                # as radius increases. This keeps time complexity O(1) while preserving statistical rigor
                x = round(x + leap_distance * np.cos(leap_angle))
                y = round(y + leap_distance * np.sin(leap_angle))
                continue

            # Collision check
            if grid[x + 1, y] or grid[x - 1, y] or grid[x, y + 1] or grid[x, y - 1]:
                if np.random.random() < sticking_prob:
                    grid[x, y] = True

                    # Update rmax if the new particle expands the cluster
                    current_r = round(current_dist)
                    if current_r > rmax:
                        rmax = current_r
                    break

            # Random walk when close to origin. Checks valid moves that arent currently occupied by a particle
            valid_moves = []
            if not grid[x + 1, y]: valid_moves.append((1, 0))
            if not grid[x - 1, y]: valid_moves.append((-1, 0))
            if not grid[x, y + 1]: valid_moves.append((0, 1))
            if not grid[x, y - 1]: valid_moves.append((0, -1))

            if valid_moves:

                idx = np.random.randint(len(valid_moves))
                dx, dy = valid_moves[idx]
                x += dx
                y += dy
            else:
                # Edge Case: If the particle wanders into a 1x1 hole and gets trapped,
                # force it to stick to prevent an infinite loop.
                grid[x, y] = True
                current_r = int(current_dist)
                if current_r > rmax:
                    rmax = current_r
                break

    return grid, rmax, cx, cy

'''
METHOD 2. Greens Function
'''

def initialize_fractal_mask(grid_size=101):
    #Initializes binary state matrix with a single central seed. Use an odd number grid size to guarantee a true center
    fractal_mask = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    fractal_mask[center, center] = 1.0
    return fractal_mask


def sparse_matrix_constructor(fractal_mask, grid_size=101, boundary_radius=45):
    # Calculates the steady-state probability field based on the CURRENT cluster.
    center = grid_size // 2

    def get_k(x, y):
        return y * grid_size + x

    # Initialize empty sparse matrix A and vector b
    A = sp.lil_matrix((grid_size ** 2, grid_size ** 2))
    b = np.zeros(grid_size ** 2)

    # Construct the matrix
    for y in range(grid_size):
        for x in range(grid_size):
            k = get_k(x, y)
            r_squared = (x - center) ** 2 + (y - center) ** 2

            # The Entire Fractal Cluster
            if fractal_mask[y, x] == 1.0:
                A[k, k] = 1.0
                b[k] = 0.0

            # The Source Boundary and Hard Grid Edges
            elif r_squared >= boundary_radius ** 2 or x == 0 or x == grid_size - 1 or y == 0 or y == grid_size - 1:
                A[k, k] = 1.0
                b[k] = 1.0

            # Empty Space (Laplacian)
            else:
                A[k, k] = -4.0
                A[k, get_k(x + 1, y)] = 1.0
                A[k, get_k(x - 1, y)] = 1.0
                A[k, get_k(x, y + 1)] = 1.0
                A[k, get_k(x, y - 1)] = 1.0

    # Convert to CSR format for fast algebraic solving
    A = A.tocsr()

    u_1D = spla.spsolve(A, b)
    u_2D = u_1D.reshape((grid_size, grid_size))
    return u_2D

def get_growth_probabilities(prob_field, fractal_mask):
    """
    Uses convolution to find empty adjacent sites and extracts their potential.
    Returns the coordinates of the perimeter and their raw probabilities.
    """
    kernel = np.array([0,1,0,1,0,1,0,1,0]).reshape((3, 3))

    neighbor_count = convolve2d(fractal_mask, kernel, mode='same')

    # Isolate the active perimeter layer
    perimeter_mask = (neighbor_count > 0) & (fractal_mask == 0)

    # Extract the actual (y, x) coordinates where the mask is True
    y_indices, x_indices = np.where(perimeter_mask)

    # Extract the potential u at those exact coordinates
    probabilities = prob_field[y_indices, x_indices]


    candidates = list(zip(y_indices, x_indices))

    return candidates, probabilities


def add_particle(candidates, raw_probabilities, fractal_mask, eta=1.0):
    """
    Normalizes the probabilities, selects a winning site via roulette wheel selection,
    and updates the binary state matrix.
    """
    # apply sticking probability
    weights = np.power(raw_probabilities, eta)

    # Normalize the weights, add tiny epsilon to prevent division by 0
    total_weight = np.sum(weights) + 1e-10
    normalized_probs = weights / total_weight


    indices = np.arange(len(candidates))
    chosen_index = np.random.choice(indices, p=normalized_probs)

    winning_y, winning_x = candidates[chosen_index]

    # Update the state matrix
    fractal_mask[winning_y, winning_x] = 1.0

    return fractal_mask


# 1. Initialization
grid_size = 101
fractal_mask = initialize_fractal_mask(grid_size)

# 2. The Growth Loop
num_particles = 500

for i in range(num_particles):
    # Pass fractal_mask into the solver so it can "see" the cluster
    u_2D = sparse_matrix_constructor(fractal_mask, grid_size=grid_size, boundary_radius=45)

    # Find the perimeter
    candidates, raw_probs = get_growth_probabilities(u_2D, fractal_mask)

    # Add one particle
    fractal_mask = add_particle(candidates, raw_probs, fractal_mask, eta=1.0)

    # Optional: Print progress so you know it hasn't frozen
    if i % 50 == 0:
        print(f"Added particle {i}...")

# 3. Visualize the Final Cluster
plt.imshow(fractal_mask, cmap='magma', origin='lower')
plt.title(f"DLA Cluster: {num_particles} Particles")
plt.show()