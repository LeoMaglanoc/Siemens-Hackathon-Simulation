import pandas as pd

val_df = pd.read_csv('data\\validation.csv')

#print(val_df[['lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ']])

val_df_dict = dict(val_df.groupby(['lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ'])['effective_stiffness'].first())
print(dict(val_df.groupby(['lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ'])['effective_stiffness'].first()))


print('adfafa', val_df_dict[(2.375, 0.2, 2.0, 3.0)])