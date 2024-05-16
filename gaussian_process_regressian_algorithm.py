"""
    The main function creates and saves the gpr model trained with provided data
    With an seperate function call you have access to the latest trained model and can predict values with it
"""

import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler
import preprocess_data as pd
import matplotlib.pyplot as plt

GPRMODELNAME = 'gpr_model.pth'
PATHTRAININGDATA = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'
# KERNEL = DotProduct() + WhiteKernel()
# KERNEL = DotProduct()
KERNEL = None

def preprocess_data(X: np.array) -> np.array:
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def create_gpr_model():
    X, y = pd.get_data(pd.PATHTRAININGDATA)
    kernel = KERNEL
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
    gpr_Pickle = open(GPRMODELNAME, 'wb') # open in binary mode!
    pickle.dump(gpr_model, gpr_Pickle)
    gpr_Pickle.close()
    print("Knn model created and saved at ./" + GPRMODELNAME + ".")

def predict_effective_stiffness(X: np.array):
    gpr_model = pickle.load(open(GPRMODELNAME, 'rb'))
    return gpr_model.predict(X)

def mean_square_error(y_pred: np.array, y_gt: np.array):
    return np.mean((y_pred-y_gt)**2)

def validate_gpr_model():
    X, y = pd.get_data(pd.PATHVALIDATIONDATA)
    y_pred = predict_effective_stiffness(X)
    mse = mean_square_error(y_pred, y)
    plot_validation(X, y, X, y_pred)
    return mse

def plot_validation(X1: np.array, y1: np.array, X2: np.array, y2: np.array):
    num_plots = X1.shape[1]
    num_rows = (num_plots + 1) // 2
    fig, axs = plt.subplots(num_rows, 2)

    for i in range(num_rows):
        for j in range(2):
            training_data_handle = axs[i, j].scatter(X1[:,2*i+j], y1/1e3, label='Expected E_eff')
            validation_data_handle = axs[i, j].scatter(X2[:,2*i+j], y2/1e3, label='Predicted E_eff', marker="*")
            axs[i, j].set_xlabel(pd.PARAMETERS[2*i+j])
            axs[i, j].set_ylabel(pd.TARGET + f" in 1e3 MPa")

    fig.legend(loc='lower center', handles=[training_data_handle, validation_data_handle], ncol=2)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    create_gpr_model()
    mse = validate_gpr_model()
    print(mse)