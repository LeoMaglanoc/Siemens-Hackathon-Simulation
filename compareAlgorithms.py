"""
    This script is intended to make a comparison of all developed prediction methods
"""

import knn_regression_algorithm as knnr

# knn regression
X, y = knnr.random_test_data_generator(20, [[0, 100], [0, 1], [10, 18], [5.7, 9.2]], [0, 2])

effective_stiffness_knn_model = knnr.get_fitted_knn_regressor_model(3, X, y)
presentation_data = [[17, 0.3, 17, 6.1], [97, 0.5, 16, 7.9]]
predicted_output = knnr.predict_effective_stiffness(effective_stiffness_knn_model, presentation_data)
print(predicted_output)

# random forrest


# point cloud


# neural network