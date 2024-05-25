import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

PATHTRAININGDATA = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'
PATHTRAININGDATA_SINGLE = './data/training_single.csv'
PATHVALIDATIONDATA_SINGLE = './data/validation_single.csv'
PARAMETERS = ['ID', 'lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ']
TARGET = 'effective_stiffness'


def get_data(path):
    a = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=1, names=PARAMETERS + [TARGET])
    X = np.array([a[PARAMETERS[1]], a[PARAMETERS[2]], a[PARAMETERS[3]], a[PARAMETERS[4]]]).T
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
    train_data = pd.read_csv(PATHTRAININGDATA).drop('ID', axis=1)
    val_data = pd.read_csv(PATHVALIDATIONDATA).drop('ID', axis=1)
    equal_rows = val_data.isin(train_data).all(axis=1)
    print(f"Found {equal_rows.sum()} equal rows in train and validation data.")
    print(f"Found {train_data.duplicated().sum()} duplicates in training data.")
    print(f"Found {val_data.duplicated().sum()} duplicates in validation data.")
    return equal_rows.any() or train_data.duplicated().any() or val_data.duplicated().any()


def delete_double_rows():
    train_data = pd.read_csv(PATHTRAININGDATA).drop('ID', axis=1)
    val_data = pd.read_csv(PATHVALIDATIONDATA).drop('ID', axis=1)
    equal_rows = val_data.isin(train_data).all(axis=1)
    print(f"Found {equal_rows.sum()} equal rows in train and validation data.")
    print(f"Found {train_data.duplicated().sum()} duplicates in training data.")
    print(f"Found {val_data.duplicated().sum()} duplicates in validation data.")
    if equal_rows.any():
        val_data_single = val_data.drop(val_data.index[equal_rows])
    else:
        val_data_single = val_data
    if train_data.duplicated().any():
        train_data_single = train_data.drop_duplicates()
    else:
        train_data_single = train_data
    if val_data_single.duplicated().any():
        val_data_single.drop_duplicates(inplace=True)
    
    train_data_single.insert(0, 'ID', range(1,1 + len(train_data_single)))
    val_data_single.insert(0, 'ID', range(1,1 + len(val_data_single)))
    train_data_single.to_csv(PATHTRAININGDATA_SINGLE, index=False)
    val_data_single.to_csv(PATHVALIDATIONDATA_SINGLE, index=False)
    print("Deleted all duplicates and equal rows.")
    return



if __name__ == '__main__':
    check_double_rows()
    delete_double_rows()