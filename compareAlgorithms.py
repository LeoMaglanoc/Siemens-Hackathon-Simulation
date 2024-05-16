"""
    This script is intended to make a comparison of all developed prediction methods
"""

import numpy as np
import knn_regression_algorithm as knnr
import random_forest_algorithm as rf
import gaussian_process_regressian_algorithm as gpr
import mlp as mlp_module
import torch

def knn(X):
    result_knn = knnr.predict_effective_stiffness(X)
    print("Knn: " + str(result_knn[0]) + " MPa")
    return result_knn[0]

def random_forest(X):
    result_rf = rf.predict_effective_stiffness(X)
    print("Random Forest: " + str(result_rf[0]) + " MPa")
    return result_rf[0]

def mlp(X):
    result_mlp = mlp_module.predict_effective_stiffness(X)
    print("Mlp: " + str(result_mlp[0][0]) + " MPa")
    return result_mlp

def point_cloud_nn(X):
    return 0

def gp(X):
    result_gpr = gpr.predict_effective_stiffness(X)
    print("Gpr: " + str(result_gpr[0]) + " MPa")
    return result_gpr[0]

if __name__ == '__main__':
    """
    Estimate the effective stiffness with different machine learning methods
    X - Input Parameters
    Format: np array 1x4 vector like [[lattice_d_cell, lattice_d_rod, lattice_number_cells_x, scaling_factor_YZ]]
    """
    X = np.array([[2.375,0.7,2.0,6.0]])
    knn(X)
    random_forest(X)
    # TODO: mlp
    mlp(X)
    # TODO: point_cloud_nn
    point_cloud_nn(X)

    gp(X)

