import numpy as np


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