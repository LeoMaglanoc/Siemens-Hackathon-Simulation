import os
import glob
import xml.etree.ElementTree as ET
import csv
import numpy as np

def parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize a dictionary to store the values
    data = {
        'lattice_d_cell': None,
        'lattice_d_rod': None,
        'lattice_number_cells_x': None,
        'scaling_factor_YZ': None,
        'young_modulus': None,
        'density': None
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

    if mass is not None and volume is not None:
        data['density'] = mass / volume

    # Extract Displacement Nodal Magnitude values
    displacement_values = []
    for response in root.findall('.//Response[@type="Solutions"][@output="Displacement - Nodal"][@component="Magnitude"]'):
        values = response.text.strip().split()
        displacement_values.extend(map(float, values))

    if displacement_values:
        data['young_modulus'] = (F / (h_total_z * w_total_y)) * (d_total_x / np.mean(displacement_values))

    return data

def main(folder_path, output_csv):
    # Get a list of all XML files in the folder
    xml_files = glob.glob(os.path.join(folder_path, '*.xml'))

    # Define the CSV file headers
    headers = ['lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ', 'young_modulus', 'density']

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)

        # Write the header
        writer.writeheader()

        # Process each XML file
        for xml_file in xml_files:
            data = parse_xml_file(xml_file)
            writer.writerow(data)

    print(f"Data written to {output_csv}")

# Example usage
if __name__ == "__main__":
    folder_path = 'data'  # Replace with the path to your folder containing XML files
    output_csv = 'data/output.csv'  # Replace with the desired output CSV file path
    main(folder_path, output_csv)

# e_solid = 210.000
# dichte_solid = 7.85 * 10^-6