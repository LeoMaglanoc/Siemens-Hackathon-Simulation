import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'path_to_your_file.csv'
df = pd.read_csv('data/validation.csv')
df = df.drop('ID', axis=1)

# Display the DataFrame (optional)
print("DataFrame:")
print(df)

# Check for duplicate rows
duplicates = df.duplicated()

# Display the duplicates
if duplicates.any():
    print("Duplicate rows found:")
    print(df[duplicates])
else:
    print("No duplicate rows found.")

# Alternatively, to see the indices of duplicate rows
duplicate_indices = df.index[duplicates]
print("Indices of duplicate rows:", duplicate_indices.tolist())
