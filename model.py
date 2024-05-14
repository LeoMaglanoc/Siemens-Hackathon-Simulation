import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Define the Dataset class
class CSVDataset(Dataset):
    def __init__(self, X, y, scale_data=True):
        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

# Define the MLP class
class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 64),
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

if __name__ == '__main__':
    # Set fixed random number seed
    torch.manual_seed(42)

    # Load dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare datasets
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    # Prepare DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get and prepare inputs
            inputs, targets = data
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Accumulate loss
            current_loss += loss.item()

            # Log the loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
        
        # Print statistics
        print(f'Epoch {epoch+1} loss: {current_loss / len(train_loader)}')

    # Process is complete
    print('Training process has finished.')

    # Evaluate the model on test data
    mlp.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, targets = data
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()

            # Log the test loss to TensorBoard
            writer.add_scalar('Loss/test', loss.item(), epoch * len(test_loader) + i)

    average_test_loss = test_loss / len(test_loader.dataset)
    print(f'Average test loss: {average_test_loss:.3f}')

    # Save the model
    torch.save(mlp.state_dict(), 'mlp_model.pth')

    # Close the TensorBoard writer
    writer.close()

# list of point coordinates as information of geometric structure? --> PointNet, but others do that
# --> encode different lattice structures as discrete classes without giving further information (one-hot encoding)?

# we only have around 100 data points

# number of model parameters roughly equal to number datapoints?
# what about number of input parameters?

# stiffness >= 0? --> ReLU/Softplus at the end or by some other method?