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

def get_training_data():
    a = np.genfromtxt(PATHTRAININGDATA, dtype=None, delimiter=',', skip_header=1, names=['id', 'lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ']]).T
    y = np.array(a['effective_stiffness'])
    return X, y


def get_validation_data():
    a = np.genfromtxt(PATHVALIDATIONDATA, dtype=None, delimiter=',', skip_header=1, names=['id', 'lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ']]).T
    y = np.array(a['effective_stiffness'])
    return X, y


def train_knn_model(X: np.array, y: np.array, k_neighbors: int) -> KNeighborsRegressor:
    knn_model = KNeighborsRegressor(k_neighbors)
    knn_model.fit(X, y)
    return knn_model


def validate_knn_model(knn_model: KNeighborsRegressor) -> float:
    X_val, y_val = get_validation_data()
    y_calc = knn_model.predict(X_val)
    print([y_calc, y_val])
    diff = y_calc - y_val
    diff_squared = diff ** 2
    mse = np.mean(diff_squared)
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
    X_train, y_train = get_training_data()
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