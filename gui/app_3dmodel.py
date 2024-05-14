import streamlit as st
import pyvista as pv
from pyvista import examples
from stpyvista import stpyvista
import tempfile

pv.global_theme.allow_empty_mesh = True

placeholder = st.empty()

def delmodel():
    del st.session_state.fileuploader


## Initialize pyvista reader and plotter
plotter = pv.Plotter(border=False, window_size=[500, 400])
plotter.background_color = "#f0f8ff"

## Create a tempfile to keep the uploaded file as pyvista's API
## only supports file paths but not buffers
with tempfile.NamedTemporaryFile(suffix="_streamlit") as f:
    #reader = pv.STLReader(f.name)

    ## Read data and send to plotter
    #mesh = reader.read()
    mesh = pv.read('Strength_Test_Cube_V1-1.obj')
    plotter.add_mesh(mesh, color="orange", specular=0.5)
    plotter.view_xz()

## Show in webpage
with placeholder.container():
    stpyvista(plotter)