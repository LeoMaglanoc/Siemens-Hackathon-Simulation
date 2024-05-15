"""
    The main function creates and saves a random forest model trained with provided data
    With the function "predict_effective_stiffness" you have access to the latest trained model and can predict values with it
"""

from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pickle

RFMODELNAME = 'rf_model.pth'
PATHTRAININGDATA = './data/training.csv'

def get_training_data():
    a = np.genfromtxt(PATHTRAININGDATA, dtype=None, delimiter=',', skip_header=1, names=['id','lattice_d_cell','lattice_d_rod','lattice_number_cells_x','scaling_factor_YZ','effective_stiffness'])
    X = np.array([a['lattice_d_cell'],a['lattice_d_rod'],a['lattice_number_cells_x'],a['scaling_factor_YZ'],]).T
    y = np.array(a['effective_stiffness'])
    return X, y

def create_rf_model(X, y):
    rf_model = RandomForestRegressor(max_depth=2, random_state=0)
    rf_model.fit(X, y)
    rf_Pickle = open(RFMODELNAME, 'wb') # open in binary mode!
    pickle.dump(rf_model, rf_Pickle)
    rf_Pickle.close()
    print("Random forest model created and saved at ./" + RFMODELNAME)

def predict_effective_stiffness(X: np.array):
    rf_model = pickle.load(open(RFMODELNAME, 'rb'))
    return rf_model.predict(X)

if __name__ == '__main__':
    X_train, y_train = get_training_data()
    create_rf_model(X_train, y_train)