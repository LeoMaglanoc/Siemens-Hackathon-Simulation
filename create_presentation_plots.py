"""
    This script is used to create the plots for the pitch presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import preprocess_data as ppd

def plot_train_and_val_data():
    X1, y1 = ppd.get_data(ppd.PATHTRAININGDATA_SINGLE)
    X2, y2 = ppd.get_data(ppd.PATHVALIDATIONDATA_SINGLE)
    num_samples_train = X1.shape[0]
    num_samples_val = X2.shape[0]
    num_plots = X1.shape[1]
    num_rows = (num_plots + 1) // 2
    fig, axs = plt.subplots(num_rows, 2)

    for i in range(num_rows):
        for j in range(2):
            training_data_handle = axs[i, j].scatter(X1[:,2*i+j], y1/1e3, label='Training data')
            validation_data_handle = axs[i, j].scatter(X2[:,2*i+j], y2/1e3, label='Validation data', marker="*")
            axs[i, j].set_xlabel(ppd.PARAMETERS[2*i+j+1])
            axs[i, j].set_ylabel(ppd.TARGET + f" in 1e3 MPa")

    fig.legend(loc='lower center', handles=[training_data_handle, validation_data_handle], ncol=2)
    fig.suptitle('Simulation data - ' + str(num_samples_train) + ' training samples ' + str(num_samples_val) + ' validation samples')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_train_and_val_data()