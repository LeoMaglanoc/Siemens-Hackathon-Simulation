"""
    The main function creates and saves a random forest model trained with provided data
    With the function "predict_effective_stiffness" you have access to the latest trained model and can predict values with it
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle

RFMODELNAME        = 'rf_model.pth'
PATHTRAININGDATA   = './data/training.csv'
PATHVALIDATIONDATA = './data/validation.csv'


def get_training_data():
    a = np.genfromtxt(PATHTRAININGDATA, dtype=None, delimiter=',', skip_header=1, names=['id','lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ'],]).T
    y = np.array(a['effective_stiffness'])
    return X, y


def preprocess_data(X: np.array) -> np.array:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def train_rf_model(X_train, y_train, depth: int) -> RandomForestRegressor:
    rf_model = RandomForestRegressor(max_depth=depth, random_state=0)
    rf_model.fit(X_train, y_train)
    return rf_model


def validate_rf_model(rf_model: RandomForestRegressor) -> float:
    a = np.genfromtxt(PATHVALIDATIONDATA, dtype=None, delimiter=',', skip_header=1, names=['id','lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X_val = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ'],]).T
    X_val = preprocess_data(X_val)
    y_val = np.array(a['effective_stiffness'])
    mse = np.mean((rf_model.predict(X_val) - y_val) ** 2)
    return mse


def save_rf_model(rf_model: RandomForestRegressor):
    rf_Pickle = open(RFMODELNAME, 'wb') # open in binary mode!
    pickle.dump(rf_model, rf_Pickle)
    rf_Pickle.close()
    print("Random forest model created and saved at ./" + RFMODELNAME)


def create_rf_model():
    print("Creating random forest model...")
    mse_errors = np.zeros(9)
    rf_models = []
    X_train, y_train = get_training_data()
    X_train = preprocess_data(X_train)
    for depth in range(1, 10):
        rf_models.append(train_rf_model(X_train, y_train, depth))
        mse_errors[depth-1] = validate_rf_model(rf_models[depth-1])
        print("Depth: " + str(depth) + ", Error: " + str(mse_errors[depth-1]))
    optimal_depth = np.argmin(mse_errors) + 1
    print("Optimal depth: " + str(optimal_depth))
    save_rf_model(rf_models[optimal_depth-1])


def predict_effective_stiffness(X: np.array):
    X = preprocess_data(X)
    rf_model = pickle.load(open(RFMODELNAME, 'rb'))
    return rf_model.predict(X)


if __name__ == '__main__':
    create_rf_model()