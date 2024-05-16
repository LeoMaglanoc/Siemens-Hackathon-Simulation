import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

PATHTRAININGDATA = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'
PARAMETERS = ['lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ']
TARGET = 'effective_stiffness'


def get_data(path):
    a = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=1, names=PARAMETERS + [TARGET])
    X = np.array([a[PARAMETERS[0]],a[PARAMETERS[1]],a[PARAMETERS[2]],a[PARAMETERS[3]]]).T
    y = np.array(a[TARGET])
    return X, y

def preprocess_data(X: np.array):
    scaler = StandardScaler()
    scaler.fit(X)
    print(f"Mean: {scaler.mean_}, Variance: {scaler.var_}")
    return scaler.transform(X)


def scale_data(X: np.array, mean, std):
    return (X - mean) / std 

def check_double_rows() -> bool:
    train_data = pd.read_csv(PATHTRAININGDATA)
    val_data = pd.read_csv(PATHVALIDATIONDATA)
    train_data_without_first_column = train_data.iloc[:, 1:]
    val_data_without_first_column = val_data.iloc[:, 1:]
    equal_rows = val_data_without_first_column.isin(train_data_without_first_column).all(axis=1)
    print(f"Found {equal_rows.sum()} equal rows in the validation data.")
    return equal_rows.any()


if __name__ == '__main__':
    check_double_rows()