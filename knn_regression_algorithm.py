"""
    The main function creates and saves the knn model trained with provided data
    With an seperate function call you have access to the latest trained model and can predict values with it
"""

import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor

KNEIGHBORS = 2
KNNMODELNAME = 'knn_model.pth'

def getTrainingData():
    X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    y = np.array([1, 2, 3])
    return X, y

def createKnnModel(X, y):
    knn_model = KNeighborsRegressor(KNEIGHBORS)
    knn_model.fit(X, y)
    knn_Pickle = open(KNNMODELNAME, 'wb') # open in binary mode!
    pickle.dump(knn_model, knn_Pickle)
    knn_Pickle.close()

def predict_effective_stiffness(X: np.array):
    knn_model = pickle.load(open(KNNMODELNAME, 'rb'))
    return knn_model.predict(X)

if __name__ == '__main__':
    X_train, y_train = getTrainingData()
    createKnnModel(X_train, y_train)