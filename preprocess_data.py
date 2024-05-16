import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pad

PATHTRAININGDATA = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'
PARAMETERS = ['id', 'lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ']
TARGET = 'effective_stiffness'


def get_data(path):
    a = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=1, names=PARAMETERS + [TARGET])
    X = np.array([a[PARAMETERS[1]],a[PARAMETERS[2]],a[PARAMETERS[3]],a[PARAMETERS[4]]]).T
    y = np.array(a[TARGET])
    return X, y

def preprocess_data(X: np.array) -> np.array:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


if __name__ == '__main__':
    pass