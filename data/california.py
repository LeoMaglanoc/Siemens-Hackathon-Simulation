import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)

# Create a DataFrame from the dataset
df = pd.DataFrame(X, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
df['MedHouseVal'] = y

# Write the DataFrame to a CSV file
csv_file = 'california_housing.csv'
df.to_csv(csv_file, index=False)

print(f'Dataset written to {csv_file}')
