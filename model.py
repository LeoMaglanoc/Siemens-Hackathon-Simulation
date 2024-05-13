import torch
from torch import nn
from torch.utils.data import DataLoader
# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# code adapted from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md

class Dataset(torch.utils.data.Dataset):
    '''
    Prepare the dataset for regression
    '''

    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    

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

    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
        
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
        
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
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
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')

    # Evaluate the model on test data
    mlp.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            
            test_loss += loss.item()

        average_test_loss = test_loss / len(test_loader.dataset)
        print(f'Average test loss: {average_test_loss:.3f}')

    # Save the model
    torch.save(mlp.state_dict(), 'mlp_model.pth')
