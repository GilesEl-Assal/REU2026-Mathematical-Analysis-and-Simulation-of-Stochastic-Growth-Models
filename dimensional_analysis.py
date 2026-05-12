from scipy.spatial.distance import pdist
import numpy as np

def analyze_fractal_dimension(grid, rmax):
    y_coords, x_coords = np.where(grid==True)
    coords = np.column_stack((x_coords, y_coords))

    N = len(coords)
    distances = pdist(coords)