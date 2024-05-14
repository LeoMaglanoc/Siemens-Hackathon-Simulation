import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import tempfile

pv.global_theme.allow_empty_mesh = True

def delmodel():
    del st.session_state.fileuploader

st.title('Effective Stiffness Prediction Platform')

st.write('**Please enter the parameters of the simulation:**')

model_path = st.text_input('Enter the path for the 3D model:', '')

st.write('Shape:')
placeholder = st.empty()

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

if model_path != '':
    ## Initialize pyvista reader and plotter
    plotter = pv.Plotter(border=False, window_size=[500, 400])
    plotter.background_color = "#f0f8ff"

    ## Create a tempfile to keep the uploaded file as pyvista's API
    ## only supports file paths but not buffers
    with tempfile.NamedTemporaryFile(suffix="_streamlit") as f:
        #reader = pv.STLReader(f.name)

        ## Read data and send to plotter
        #mesh = reader.read()
        mesh = pv.read(model_path)
        #plotter.add_mesh(mesh, color="orange", specular=0.5)
        plotter.add_mesh(mesh, specular=0.5)
        plotter.view_xz()

    ## Show in webpage
    with placeholder.container():
        stpyvista(plotter)