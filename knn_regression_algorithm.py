"""
    The main function creates and saves the knn model trained with provided data
    With an seperate function call you have access to the latest trained model and can predict values with it
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

KNNMODELNAME = 'knn_model.pth'
PATHTRAININGDATA = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'
PARAMETERS = ['id', 'lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ']
TARGET = 'effective_stiffness'

def get_data(path):
    a = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=1, names=PARAMETERS + [TARGET])
    X = np.array([a[PARAMETERS[1]],a[PARAMETERS[2]],a[PARAMETERS[3]],a[PARAMETERS[4]]]).T
    y = np.array(a[TARGET])
    return X, y


def train_knn_model(X: np.array, y: np.array, k_neighbors: int) -> KNeighborsRegressor:
    knn_model = KNeighborsRegressor(k_neighbors)
    knn_model.fit(X, y)
    return knn_model


def validate_knn_model(knn_model: KNeighborsRegressor) -> float:
    X_val, y_val = get_data(PATHVALIDATIONDATA)
    y_calc = knn_model.predict(X_val)
    mse = np.mean((y_calc - y_val) ** 2)
    return mse


def save_knn_model(knn_model: KNeighborsRegressor):
    knn_Pickle = open(KNNMODELNAME, 'wb') # open in binary mode!
    pickle.dump(knn_model, knn_Pickle)
    knn_Pickle.close()
    print("Knn model created and saved at ./" + KNNMODELNAME)


def create_knn_model():
    print("Creating knn model...")
    mse_errors = np.zeros(9)
    knn_models = []
    X_train, y_train = get_data(PATHTRAININGDATA)
    for k_neighbors in range(1, 10):
        knn_models.append(train_knn_model(X_train, y_train, k_neighbors))
        mse_errors[k_neighbors-1] = validate_knn_model(knn_models[k_neighbors-1])
        print("Neighbors: " + str(k_neighbors) + ", Error: " + str(mse_errors[k_neighbors-1]))
    optimal_neighbors = np.argmin(mse_errors) + 1
    print("Optimal neighbors: " + str(optimal_neighbors))
    save_knn_model(knn_models[optimal_neighbors-1])


def predict_effective_stiffness(X: np.array):
    knn_model = pickle.load(open(KNNMODELNAME, 'rb'))
    return knn_model.predict(X)

if __name__ == '__main__':
    create_knn_model()