"""
    The main function creates and saves a random forest model trained with provided data
    With the function "predict_effective_stiffness" you have access to the latest trained model and can predict values with it
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import preprocess_data as pd

RFMODELNAME        = 'rf_model.pth'


def train_rf_model(X_train, y_train, depth: int) -> RandomForestRegressor:
    rf_model = RandomForestRegressor(max_depth=depth, random_state=0)
    rf_model.fit(X_train, y_train)
    return rf_model


def validate_rf_model(rf_model: RandomForestRegressor) -> float:
    X_val, y_val = pd.get_data(pd.PATHVALIDATIONDATA)
    X_val = pd.preprocess_data(X_val)
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
    X_train, y_train = pd.get_data(pd.PATHTRAININGDATA)
    X_train = pd.preprocess_data(X_train)
    for depth in range(1, 10):
        rf_models.append(train_rf_model(X_train, y_train, depth))
        mse_errors[depth-1] = validate_rf_model(rf_models[depth-1])
        print("Depth: " + str(depth) + ", Error: " + str(mse_errors[depth-1]))
    optimal_depth = np.argmin(mse_errors) + 1
    print("Optimal depth: " + str(optimal_depth))
    save_rf_model(rf_models[optimal_depth-1])


def predict_effective_stiffness(X: np.array):
    X = pd.preprocess_data(X)
    rf_model = pickle.load(open(RFMODELNAME, 'rb'))
    return rf_model.predict(X)


if __name__ == '__main__':
    create_rf_model()