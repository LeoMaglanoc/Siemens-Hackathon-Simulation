import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def random_test_data_generator(num_data: int, ranges: list, is_int: list):
    '''
        Generate random data.

        args:
            num_data: Number of data sample.
            ranges: The numeric range for each parameter, e.g. [[0, 100], [0, 1], [10, 18], [5.7, 9.2]]
            is_int: Specify which parameters are integer. e.g. [0, 2] -> the 0 and 2nd parameters are integers
                    The rest that are not specified will be generated as float.
        output:
            Randomly generated input parameters and output effective stiffness ground truth pairs.
    '''
    X_list = []
    is_int_pointer = 0

    for i in range(len(ranges)):

        if (is_int_pointer < len(is_int)) and (i == is_int[is_int_pointer]):
            is_int_pointer = is_int_pointer + 1
            range_i = ranges[i]
            X_list.append(np.random.randint(range_i[0], range_i[1], num_data))
        else:
            range_i = ranges[i]
            X_list.append(np.random.uniform(range_i[0], range_i[1], num_data))

    X = np.vstack(X_list)
    X = np.transpose(X)

    y = np.random.uniform(0, 1, num_data)

    return X, y

def get_fitted_knn_regressor_model(k_neighbors: int, X: np.array, y: np.array):
    '''
        Given the GT data X, y, and number of neighbors k_neighbors,
        fit and return a knn regressor model.
    '''
    knn_model = KNeighborsRegressor(k_neighbors)
    knn_model.fit(X, y)

    return knn_model

def predict_effective_stiffness(model, X: np.array):
    '''
        Given the knn regressor model and simulation parameters X, predict the effective stiffness.
        X should be 2D.
    '''
    return model.predict(X)
