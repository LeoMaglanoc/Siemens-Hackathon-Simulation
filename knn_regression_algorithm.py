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
    #X = np.array([lattice_d_cell,lattice_d_rod,lattice_number_cells_x,scaling_factor_YZ, density]).T
    #y = np.array(young_modulus)
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ']]).T
    y = np.array(a['effective_stiffness'])
    return X, y


def preprocess_data(X: np.array) -> np.array:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def train_knn_model(X: np.array, y: np.array, k_neighbors: int) -> KNeighborsRegressor:
    knn_model = KNeighborsRegressor(k_neighbors)
    knn_model.fit(X, y)
    return knn_model


def validate_knn_model(knn_model: KNeighborsRegressor) -> float:
    a = np.genfromtxt(PATHVALIDATIONDATA, dtype=None, delimiter=',', skip_header=1, names=['id', 'lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X_val = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ']]).T
    X_val = preprocess_data(X_val)
    y_val = np.array(a['effective_stiffness'])
    mse = np.mean((knn_model.predict(X_val) - y_val) ** 2)
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
    X_train = preprocess_data(X_train)
    for k_neighbors in range(1, 10):
        knn_models.append(train_knn_model(X_train, y_train, k_neighbors))
        mse_errors[k_neighbors-1] = validate_knn_model(knn_models[k_neighbors-1])
        print("Neighbors: " + str(k_neighbors) + ", Error: " + str(mse_errors[k_neighbors-1]))
    optimal_neighbors = np.argmin(mse_errors) + 1
    print("Optimal neighbors: " + str(optimal_neighbors))
    save_knn_model(knn_models[optimal_neighbors-1])


def predict_effective_stiffness(X: np.array):
    X = preprocess_data(X)
    knn_model = pickle.load(open(KNNMODELNAME, 'rb'))
    return knn_model.predict(X)

if __name__ == '__main__':
    create_knn_model()