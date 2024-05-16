"""
    This script is used to create the plots for the pitch presentation.
"""

import numpy as np
import matplotlib.pyplot as plt

PARAMETERS = ['lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ']
TARGET = 'effective_stiffness'

def get_data(path) -> tuple[np.array, np.array]:
    a = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=1, names=['id', 'lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X = np.array([a[PARAMETERS[0]],a[PARAMETERS[1]],a[PARAMETERS[2]],a[PARAMETERS[3]]]).T
    y = np.array(a[TARGET])
    return X, y


def plot_2d_data(X1: np.array, y1: np.array, X2: np.array, y2: np.array):
    num_plots = X1.shape[1]
    num_rows = (num_plots + 1) // 2
    fig, axs = plt.subplots(num_rows, 2)

    for i in range(num_rows):
        for j in range(2):
            training_data_handle = axs[i, j].scatter(X1[:,2*i+j], y1/1e3, label='Training data')
            validation_data_handle = axs[i, j].scatter(X2[:,2*i+j], y2/1e3, label='Validation data', marker="*")
            axs[i, j].set_xlabel(PARAMETERS[2*i+j])
            axs[i, j].set_ylabel(TARGET + f" in 1e3 MPa")

    fig.legend(loc='lower center', handles=[training_data_handle, validation_data_handle], ncol=2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_train, y_train = get_data('./data/training.csv')
    X_val, y_val = get_data('./data/validation.csv')
    plot_2d_data(X_train, y_train, X_val, y_val)