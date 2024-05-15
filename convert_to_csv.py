import os
import glob
import xml.etree.ElementTree as ET
import csv
import numpy as np
import random

def parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize a dictionary to store the values
    data = {
        'ID': None,
        'lattice_d_cell': None,
        'lattice_d_rod': None,
        'lattice_number_cells_x': None,
        'scaling_factor_YZ': None,
        'effective_stiffness': None
    }

    d_total_x = None
    h_total_z = None
    w_total_y = None
    F = 100
    mass = None
    volume = None

    # Extract CAD Expressions values
    for response in root.findall('.//Response[@type="CAD_Expressions"]'):
        paramname = response.get('paramname')
        value = float(response.get('value'))

        if paramname == "User_Defined_Lattice_d_cell":
            data['lattice_d_cell'] = value
        elif paramname == "User_Defined_Lattice_d_rod":
            data['lattice_d_rod'] = value
        elif paramname == "User_Defined_Lattice_number_cells_x":
            data['lattice_number_cells_x'] = value
        elif paramname == "User_Defined_scaling_factor_YZ":
            data['scaling_factor_YZ'] = value
        elif paramname == "User_Defined_d_total_x":
            d_total_x = value
        elif paramname == "User_Defined_h_total_z":
            h_total_z = value
        elif paramname == "User_Defined_w_total_y":
            w_total_y = value

    # Extract MassProperty values
    for response in root.findall('.//Response[@type="MassProperty"]'):
        paramname = response.get('paramname')
        value = float(response.get('value'))

        if paramname == "Strength_Test_Cube_V1_1_Volume":
            volume = value
        elif paramname == "Strength_Test_Cube_V1_1_Mass":
            mass = value

    '''
    if mass is not None and volume is not None:
        data['density'] = mass / volume
        data['porosity'] = 1 - (volume / (h_total_z * w_total_y * d_total_x)) 
    '''
        
    # Extract Displacement Nodal Magnitude values
    displacement_values = []
    for response in root.findall('.//Response[@type="Solutions"][@output="Displacement - Nodal"][@component="Magnitude"]'):
        values = response.text.strip().split()
        displacement_values.extend(map(float, values))

    if displacement_values:
        data['effective_stiffness'] = (F / (h_total_z * w_total_y)) * (d_total_x / np.mean(displacement_values))

    return data

def main(folder_path, train_csv, val_csv):
    xml_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))

    all_data = []

    # Process each XML file
    for i, xml_file in enumerate(xml_files):
        data = parse_xml_file(xml_file)
        data['ID'] = i + 1  # Assign a unique ID to each data point
        all_data.append(data)

    # Shuffle the data randomly
    random.shuffle(all_data)

    # Split the data into training (70%) and validation (30%) sets
    split_idx = int(len(all_data) * 0.7)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Define the CSV file headers
    headers = ['ID', 'lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ', 'effective_stiffness']

    # Write the training data to the CSV file
    with open(train_csv, mode='w', newline='') as train_csv_file:
        writer = csv.DictWriter(train_csv_file, fieldnames=headers)
        writer.writeheader()
        for row in train_data:
            writer.writerow(row)

    # Write the validation data to the CSV file
    with open(val_csv, mode='w', newline='') as val_csv_file:
        writer = csv.DictWriter(val_csv_file, fieldnames=headers)
        writer.writeheader()
        for row in val_data:
            writer.writerow(row)

    print(f"Training data written to {train_csv}")
    print(f"Validation data written to {val_csv}")

# Example usage
if __name__ == "__main__":
    folder_path = 'data'  # Replace with the path to your folder containing XML files
    train_csv = 'data/training.csv'  # Replace with the desired output training CSV file path
    val_csv = 'data/validation.csv'  # Replace with the desired output validation CSV file path
    main(folder_path, train_csv, val_csv)

# e_solid = 210.000
# dichte_solid = 7.85 * 10^-6
