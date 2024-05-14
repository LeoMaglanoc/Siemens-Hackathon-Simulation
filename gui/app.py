import streamlit as st

st.title('Effective Stiffness Prediction Platform')

st.write('**Please enter the parameters of the simulation:**')


Lattice_d_cell = st.slider("What is the Lattice_d_cell?", 1.9, 2.7, 2.3, 0.1)
st.write(f"Lattice_d_cell is {Lattice_d_cell}")

Lattice_d_rod = st.slider("What is the Lattice_d_rod?", 0.2, 1.2, 0.7, 0.05)
st.write(f"Lattice_d_rod is {Lattice_d_rod}")

Lattice_number_cells_x = st.slider("What is the Lattice_number_cells_x?", 0, 10, 5, 1)
st.write(f"Lattice_number_cells_x is {Lattice_number_cells_x}")

Scaling_factor_YZ = st.slider("What is the Scaling_factor_YZ?", 1, 6, 3, 1)
st.write(f"Scaling_factor_YZ is {Scaling_factor_YZ}")

Lattice_num = st.slider("What is the Lattice_num?", 1, 4, 2, 1)
st.write(f"Lattice_num is {Lattice_num}")

st.header(f'**Effective Stiffness is {Lattice_d_cell+Lattice_d_rod+Lattice_number_cells_x+Scaling_factor_YZ+Lattice_num}**')

