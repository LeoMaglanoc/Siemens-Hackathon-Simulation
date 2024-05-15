import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

# Define the Dataset class
class Dataset(torch.utils.data.Dataset):
    '''
    Prepare the dataset for regression
    '''
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float().reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class CSVDataset(Dataset):
    def __init__(self, source, scale_data=True):
        # Read the CSV file or DataFrame
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source
        
        # Assume the last column is the target and the rest are features
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Apply scaling if necessary
        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


# Define the MLP class
class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

# Main script
if __name__ == '__main__':
    # Set fixed random number seed
    torch.manual_seed(42)

    # Create the dataset instances
    dataset = CSVDataset('data/output.csv')

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataset.X.numpy(), dataset.y.numpy(), test_size=0.2, random_state=42)

    # Create DataFrames from split data
    train_df = pd.DataFrame(data=np.hstack((X_train, y_train.reshape(-1, 1))))
    test_df = pd.DataFrame(data=np.hstack((X_test, y_test.reshape(-1, 1))))

    # Create datasets from split data
    train_dataset = CSVDataset(train_df)
    test_dataset = CSVDataset(test_df)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/')

    # Run the training loop
    for epoch in range(5):
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

    # Evaluate the model on test data
    mlp.eval()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, targets = data
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()
            writer.add_scalar('Loss/test', loss.item(), epoch * len(test_loader) + i)

    average_test_loss = test_loss / len(test_loader.dataset)
    print(f'Average test loss: {average_test_loss:.3f}')

    # Save the model
    torch.save(mlp.state_dict(), 'mlp_model.pth')

    # Close the TensorBoard writer
    writer.close()

# inference script which takes saved model and dummy inputs and outputs stiffness

# list of point coordinates as information of geometric structure? --> PointNet, but others do that
# --> encode different lattice structures as discrete classes without giving further information (one-hot encoding)?

# we only have around 100 data points

# number of model parameters roughly equal to number datapoints?
# what about number of input parameters?

# stiffness >= 0? --> ReLU/Softplus at the end or by some other method?