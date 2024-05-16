"""
    The main function creates and saves the knn model trained with provided data
    With an seperate function call you have access to the latest trained model and can predict values with it
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


KNNMODELNAME = 'knn_model.pth'
PATHTRAININGDATA = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'
PARAMETERS = ['id', 'lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ']
TARGET = 'effective_stiffness'
MAXNEIGHBORS = 7 # Can be set hihger if more training data is available


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
    plot_validation(X_val, y_val, X_val, y_calc)
    return mse


def save_knn_model(knn_model: KNeighborsRegressor):
    knn_Pickle = open(KNNMODELNAME, 'wb') # open in binary mode!
    pickle.dump(knn_model, knn_Pickle)
    knn_Pickle.close()
    print("Knn model created and saved at ./" + KNNMODELNAME)


def plot_validation(X1: np.array, y1: np.array, X2: np.array, y2: np.array):
    num_plots = X1.shape[1]
    num_rows = (num_plots + 1) // 2
    fig, axs = plt.subplots(num_rows, 2)

    for i in range(num_rows):
        for j in range(2):
            training_data_handle = axs[i, j].scatter(X1[:,2*i+j], y1/1e3, label='Expected E_eff')
            validation_data_handle = axs[i, j].scatter(X2[:,2*i+j], y2/1e3, label='Predicted E_eff', marker="*")
            axs[i, j].set_xlabel(PARAMETERS[2*i+j])
            axs[i, j].set_ylabel(TARGET + f" in 1e3 MPa")

    fig.legend(loc='lower center', handles=[training_data_handle, validation_data_handle], ncol=2)
    plt.tight_layout()
    plt.show(block = False)
    

def plot_loss(loss: np.array):
    plt.figure()
    plt.scatter(range(1, MAXNEIGHBORS+1, 1), loss)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Mean squared error')
    plt.ioff()
    plt.show()


def create_knn_model():
    print("Creating knn model...")
    mse_errors = np.zeros(MAXNEIGHBORS)
    knn_models = []
    X_train, y_train = get_data(PATHTRAININGDATA)
    for k_neighbors in range(1, MAXNEIGHBORS+1, 1):
        knn_models.append(train_knn_model(X_train, y_train, k_neighbors))
        mse_errors[k_neighbors-1] = validate_knn_model(knn_models[k_neighbors-1])
        print("Neighbors: " + str(k_neighbors) + ", Error: " + str(mse_errors[k_neighbors-1]))
    optimal_neighbors = np.argmin(mse_errors) + 1
    print("Optimal neighbors: " + str(optimal_neighbors))
    save_knn_model(knn_models[optimal_neighbors-1])
    plot_loss(mse_errors)


def predict_effective_stiffness(X: np.array):
    knn_model = pickle.load(open(KNNMODELNAME, 'rb'))
    return knn_model.predict(X)

if __name__ == '__main__':
    create_knn_model()