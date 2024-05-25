"""
    The main function creates and saves a random forest model trained with provided data
    With the function "predict_effective_stiffness" you have access to the latest trained model and can predict values with it
"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
import matplotlib.pyplot as plt
import preprocess_data as ppd

RFMODELNAME = 'rf_model.pth'
MAX_DEPTH = 7


def train_rf_model(X_train, y_train, depth: int) -> RandomForestRegressor:
    rf_model = RandomForestRegressor(max_depth=depth, random_state=0)
    rf_model.fit(X_train, y_train)
    return rf_model


def validate_rf_model(rf_model: RandomForestRegressor) -> float:
    X_val, y_val = ppd.get_data(ppd.PATHVALIDATIONDATA_SINGLE)
    X_val = ppd.preprocess_data(X_val)
    y_calc = rf_model.predict(X_val)
    mse = np.mean((y_calc - y_val) ** 2)
    plot_validation(X_val, y_val, X_val, y_calc)
    return mse


def save_rf_model(rf_model: RandomForestRegressor):
    rf_Pickle = open(RFMODELNAME, 'wb') # open in binary mode!
    pickle.dump(rf_model, rf_Pickle)
    rf_Pickle.close()
    print("Random forest model created and saved at ./" + RFMODELNAME)


def plot_validation(X1: np.array, y1: np.array, X2: np.array, y2: np.array):
    num_samples = X1.shape[0]
    num_plots = X1.shape[1]
    num_rows = (num_plots + 1) // 2
    fig, axs = plt.subplots(num_rows, 2)

    for i in range(num_rows):
        for j in range(2):
            training_data_handle = axs[i, j].scatter(X1[:,2*i+j], y1/1e3, label='Expected E_eff')
            validation_data_handle = axs[i, j].scatter(X2[:,2*i+j], y2/1e3, label='Predicted E_eff', marker="*")
            axs[i, j].set_xlabel(ppd.PARAMETERS[2*i+j])
            axs[i, j].set_ylabel(ppd.TARGET + f" in 1e3 MPa")

    fig.legend(loc='lower center', handles=[training_data_handle, validation_data_handle], ncol=2)
    fig.suptitle('Random forest model validation - ' + str(num_samples) + ' validation samples')
    plt.tight_layout()
    plt.show(block = False)

def plot_loss(loss: np.array):
    plt.figure()
    plt.scatter(range(1, MAX_DEPTH+1, 1), loss)
    plt.xlabel('Max depth of the tree')
    plt.ylabel('Mean squared error')
    plt.title('Random forest model validation loss')
    plt.ioff()
    plt.show()


def create_rf_model():
    print("Creating random forest model...")
    mse_errors = np.zeros(MAX_DEPTH)
    rf_models = []
    X_train, y_train = ppd.get_data(ppd.PATHTRAININGDATA_SINGLE)
    print("Nr of training samples: " + str(X_train.shape[0]))
    X_train = ppd.preprocess_data(X_train)
    for depth in range(1, MAX_DEPTH+1):
        rf_models.append(train_rf_model(X_train, y_train, depth))
        mse_errors[depth-1] = validate_rf_model(rf_models[depth-1])
        print("Depth: " + str(depth) + ", Error: " + str(mse_errors[depth-1]))
    optimal_depth = np.argmin(mse_errors) + 1
    print("Optimal depth: " + str(optimal_depth))
    save_rf_model(rf_models[optimal_depth-1])
    plot_loss(mse_errors)


def predict_effective_stiffness(X: np.array):
    scaler_mean = np.array([2.51,0.616,2.4,4.6]) # np.array([2.375,0.6875,2.0,6.0])
    scaler_std = np.array([0.0324, 0.066344, 0.0384, 2.16]) # np.array([0, 0.07755208, 0, 0])
    X = ppd.scale_data(X, scaler_mean, scaler_std)
    rf_model = pickle.load(open(RFMODELNAME, 'rb'))
    return rf_model.predict(X)


if __name__ == '__main__':
    create_rf_model()