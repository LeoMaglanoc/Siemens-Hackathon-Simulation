import numpy as np

def Porosity(d_cell, d_rod, scaling_factor, lattice_num):
    v_lattice = np.sqrt(3)*np.pi*d_cell*np.power(d_rod,2)*np.pow(scaling_factor,2)*lattice_num
    v_cube = np.power(d_cell,3)*np.pow(scaling_factor,2)*lattice_num
    v_plates = 1.5*np.power(d_cell,2)*np.power(scaling_factor,2)*lattice_num
    v_mesh = v_lattice + 2*v_plates
    v_total = v_cube + 2*v_plates
    epsilon = 1-v_mesh/v_total
    return epsilon

def Porosity_ML(Volume, x,y,z):
    epsilon = 1-Volume/(x*y*z)
    return epsilon
