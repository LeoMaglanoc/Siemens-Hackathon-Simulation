"""
    This script is intended to make a comparison of all developed prediction methods
"""

import numpy as np
import knn_regression_algorithm as knnr

def main(X):
    """
        Estimate the effective stiffness with different machine learning methods
        X - Input Parameters
            Format: np array 1x5 vector like [[lattice_d_cell, lattice_d_rod, lattice_number_cells_x, scaling_factor_YZ, density]]
    """
    result_knn = knnr.predict_effective_stiffness(X)
    print("Knn: " + str(result_knn[0]) + " GPa")
    # random forrest
    # neural network
    
if __name__ == '__main__':
    X = np.array([[3, 4, 2, 1, 6]])
    main(X)
