import numpy as np

def Porosity(d_cell, d_rod, scaling_factor, lattice_num):
    v_lattice = np.sqrt(2)*np.pi*d_cell*np.power(d_rod,2)*scaling_factor*lattice_num
    v_cube = np.power(d_cell,3)*scaling_factor*lattice_num
    v_plates = 1.5*np.power(d_cell,2)*scaling_factor*lattice_num
    v_mesh = v_lattice + 2*v_plates
    v_total = v_cube + 2*v_plates
    epsilon = 1-v_mesh/v_total
    return epsilon