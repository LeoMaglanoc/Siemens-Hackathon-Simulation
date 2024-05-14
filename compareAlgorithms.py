"""
    This script is intended to make a comparison of all developed prediction methods
"""

import numpy as np
import knn_regression_algorithm as knnr

def main(X):
    """
        Estimate the effective stiffness with different machine learning models
        X - Input Parameters
            Format: np array vector like
    """
    result_knn = knnr.predict_effective_stiffness(X)
    # random forrest
    # neural network
    
if __name__ == '__main__':
    X = np.array([[3, 4, 2, 1, 6]])
    main(X)
