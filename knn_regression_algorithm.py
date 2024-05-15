"""
    The main function creates and saves the knn model trained with provided data
    With an seperate function call you have access to the latest trained model and can predict values with it
"""

import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor

KNEIGHBORS = 2
KNNMODELNAME = 'knn_model.pth'
PATHTRAININGDATA = './data/training.csv'

def get_training_data():
    a = np.genfromtxt(PATHTRAININGDATA, dtype=None, delimiter=',', skip_header=1, names=['id', 'lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    #X = np.array([lattice_d_cell,lattice_d_rod,lattice_number_cells_x,scaling_factor_YZ, density]).T
    #y = np.array(young_modulus)
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ']]).T
    y = np.array(a['effective_stiffness'])
    return X, y

def create_knn_model(X, y):
    knn_model = KNeighborsRegressor(KNEIGHBORS)
    knn_model.fit(X, y)
    knn_Pickle = open(KNNMODELNAME, 'wb') # open in binary mode!
    pickle.dump(knn_model, knn_Pickle)
    knn_Pickle.close()
    print("Knn model created and saved at ./" + KNNMODELNAME + " with " + str(KNEIGHBORS) + " neighbors.")

def predict_effective_stiffness(X: np.array):
    knn_model = pickle.load(open(KNNMODELNAME, 'rb'))
    return knn_model.predict(X)

if __name__ == '__main__':
    X_train, y_train = get_training_data()
    create_knn_model(X_train, y_train)