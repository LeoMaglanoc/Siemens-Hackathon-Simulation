"""
    The main function creates and saves the gpr model trained with provided data
    With an seperate function call you have access to the latest trained model and can predict values with it
"""

import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler

GPRMODELNAME = 'gpr_model.pth'
PATHTRAININGDATA = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'

def preprocess_data(X: np.array) -> np.array:
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def get_training_data():
    a = np.genfromtxt(PATHTRAININGDATA, dtype=None, delimiter=',', skip_header=1, names=['id', 'lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ']]).T
    X = preprocess_data(X)
    y = np.array(a['effective_stiffness'])
    return X, y

def get_validation_data():
    a = np.genfromtxt(PATHVALIDATIONDATA, dtype=None, delimiter=',', skip_header=1, names=['id', 'lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ']]).T
    X = preprocess_data(X)
    y = np.array(a['effective_stiffness'])
    return X, y

def create_gpr_model():
    X, y = get_training_data()
    kernel = DotProduct() + WhiteKernel()
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
    gpr_Pickle = open(GPRMODELNAME, 'wb') # open in binary mode!
    pickle.dump(gpr_model, gpr_Pickle)
    gpr_Pickle.close()
    print("Knn model created and saved at ./" + GPRMODELNAME + ".")

def predict_effective_stiffness(X: np.array):
    gpr_model = pickle.load(open(GPRMODELNAME, 'rb'))
    return gpr_model.predict(X)

def mean_square_error(y_pred: np.array, y_gt: np.array):
    return np.sqrt(np.mean((y_pred-y_gt)**2))

def validate_gpr_model():
    X, y = get_validation_data()
    mse = mean_square_error(predict_effective_stiffness(X), y)
    return mse


if __name__ == '__main__':
    create_gpr_model()
    print(validate_gpr_model())