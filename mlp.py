import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np

# Define the Dataset class
class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, source, scale_data=True):
        # Read the CSV file or DataFrame
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source
        
        # Assume the first column is the ID, the last column is the target, and the rest are features
        self.ids = df.iloc[:, 0].values  # ID column
        X = df.iloc[:, 1:-1].values      # Feature columns
        y = df.iloc[:, -1].values        # Target column
        
        # Apply scaling if necessary
        if scale_data:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# Define the MLP class
class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 1000),  
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )
        self.train_scaler_X = None 
        self.train_scaler_y = None

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

# Function to load the model
def load_model(model_path):
    model = MLP()
    # TODO: add saving of scaler transform
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Inference function
def predict_effective_stiffness(X):
    model = load_model('mlp_model.pth')
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    X = model.train_scaler_X.transform(X)
    with torch.no_grad():
        prediction = model(X)
    return model.train_scaler_y.inverse_transform(prediction.numpy())

# Main script
if __name__ == '__main__':
    # Set fixed random number seed
    torch.manual_seed(42)

    # Create the dataset instances
    train_dataset = CSVDataset('data/training.csv')
    val_dataset = CSVDataset('data/validation.csv')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0)

    # Initialize the MLP
    mlp = MLP()
    mlp.train_scaler_X = train_dataset.scaler_X 
    mlp.train_scaler_y = train_dataset.scaler_y

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/')

    # Run the training loop
    for epoch in range(100):
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        print(f'Epoch {epoch+1} loss: {current_loss / len(train_loader)}')

    print('Training process has finished.')

    # Evaluate the model on validation data
    mlp.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            writer.add_scalar('Loss/val', loss.item(), epoch * len(val_loader) + i)

    average_val_loss = val_loss / len(val_loader.dataset)
    print(f'Average validation loss: {average_val_loss:.3f}')

    # Save the model
    torch.save(mlp.state_dict(), 'mlp_model.pth')

    # Close the TensorBoard writer
    writer.close()

# TODO: add data inverse normalization to MLP inference, json file?

# TODO: same scaling for validation and training

# TODO: am I correctly calculating validation loss (in the non-normalized range?)

# inference script which takes saved model and dummy inputs and outputs stiffness

# list of point coordinates as information of geometric structure? --> PointNet, but others do that
# --> encode different lattice structures as discrete classes without giving further information (one-hot encoding)?

# we only have around 100 data points

# number of model parameters roughly equal to number datapoints?
# what about number of input parameters?

# stiffness >= 0? --> ReLU/Softplus at the end or by some other method?
