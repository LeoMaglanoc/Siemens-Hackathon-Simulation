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
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
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
            nn.Linear(4, 64),  # Adjust input size based on features
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

# Function to load the model
def load_model(model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Inference function
def predict_effective_stiffness(X):
    model = load_model('mlp_model.pth')
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(X)
    return prediction.numpy()

# Main script
if __name__ == '__main__':
    # Set fixed random number seed
    torch.manual_seed(42)

    # Create the dataset instances
    train_dataset = CSVDataset('data/training.csv')
    val_dataset = CSVDataset('data/validation.csv')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0)

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

# inference script which takes saved model and dummy inputs and outputs stiffness

# list of point coordinates as information of geometric structure? --> PointNet, but others do that
# --> encode different lattice structures as discrete classes without giving further information (one-hot encoding)?

# we only have around 100 data points

# number of model parameters roughly equal to number datapoints?
# what about number of input parameters?

# stiffness >= 0? --> ReLU/Softplus at the end or by some other method?
