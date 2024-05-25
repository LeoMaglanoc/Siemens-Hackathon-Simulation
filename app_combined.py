import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import tempfile
import random_forest_algorithm as rf

import numpy as np
import knn_regression_algorithm as knnr
import mlp as mlp_module
import torch
import pandas as pd


st.set_page_config(layout="wide")

def typewrite(text:str):
    with open("assets/codepen.css") as f:
        # The CSS from the codepen was stored in codepen.css
        css = f.read()

    with open("assets/codepen.js") as f:
        # The JS from the codepen was stored in codepen.js
        js = f.read()

    html = f"""
    <!DOCTYPE html>
    <head>
    <style>
        {css}
    </style>
    </head>
    <body>
        <p id="typewrite" data-content="">{text}</p>
        <script>
            {js}
        </script>
    </body>
    </html>
    """
    return html

def knn(X):
    result_knn = knnr.predict_effective_stiffness(X)
    #print("Knn: " + str(result_knn[0]) + " MPa")
    return result_knn[0]    

def random_forest(X):
    result_rf = rf.predict_effective_stiffness(X)
    print("Random Forest: " + str(result_rf[0]) + " MPa")
    return result_rf[0]

def mlp(X):
    result_mlp, prediction_time = mlp_module.predict_effective_stiffness(X)
    #print("Mlp: " + str(result_mlp[0][0]) + " MPa")
    if result_mlp[0][0] < 0:
        result_mlp[0][0] = 0
    return result_mlp[0][0], prediction_time

def point_cloud_nn(X):
    return 0


def Porosity(d_cell, d_rod, lattice_num, scaling_factor):
    v_lattice = np.sqrt(3)*np.pi*d_cell*np.power(d_rod,2)*np.power(scaling_factor,2)*lattice_num
    v_cube = np.power(d_cell,3)*np.power(scaling_factor,2)*lattice_num
    v_plates = 1.5*np.power(d_cell,2)*np.power(scaling_factor,2)*lattice_num
    v_mesh = v_lattice + 2*v_plates
    v_total = v_cube + 2*v_plates
    epsilon = 1-v_mesh/v_total
    return epsilon


pv.global_theme.allow_empty_mesh = True

def delmodel():
    del st.session_state.fileuploader

#@st.cache_data(experimental_allow_widgets=True)
def showmodel(model_path):
    global placeholder
    if model_path != '':
    ## Initialize pyvista reader and plotter
        plotter = pv.Plotter(border=False, window_size=[500, 400])
        plotter.background_color = "#f0f8ff"

        ## Create a tempfile to keep the uploaded file as pyvista's API
        ## only supports file paths but not buffers
        #with tempfile.NamedTemporaryFile(suffix="_streamlit") as f:
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

try_text = """Welcome to the demo! We are the best team. 
 Our project is amazing.
 Team Smart 3D-Printing 
 Jiyoun Jo Leo Malaz Sebastian Yan-Ling Yuejie Umut"""

typewrited = typewrite(try_text)
st.components.v1.html(typewrited, height=400, width=1100, scrolling=True)

st.title('Effective Stiffness Prediction Platform')

st.write('**Please enter the parameters of the simulation:**')

model_path = st.text_input('Enter the path for the 3D model:', '')

st.write('Shape:')
placeholder = st.empty()

col1, col2 = st.columns(2)

col1.header('Parameters:')

Lattice_d_cell = col1.slider("What is the Lattice_d_cell?", 1.9, 2.75, 2.3, 0.025)
col1.write(f"Lattice_d_cell is {Lattice_d_cell}")

Lattice_d_rod = col1.slider("What is the Lattice_d_rod?", 0.2, 1.2, 0.7, 0.05)
col1.write(f"Lattice_d_rod is {Lattice_d_rod}")

Lattice_number_cells_x = col1.slider("What is the Lattice_number_cells_x?", 0, 10, 5, 1)
col1.write(f"Lattice_number_cells_x is {Lattice_number_cells_x}")

Scaling_factor_YZ = col1.slider("What is the Scaling_factor_YZ?", 1, 6, 3, 1)
col1.write(f"Scaling_factor_YZ is {Scaling_factor_YZ}")

X = np.array([[Lattice_d_cell,Lattice_d_rod,Lattice_number_cells_x,Scaling_factor_YZ]])
knn_pred = knn(X)
# TODO: random forest
rf_pred = random_forest(X)
# TODO: mlp
mlp_pred, prediction_time = mlp(X)
# TODO: point_cloud_nn
pcd_pred = point_cloud_nn(X)


porosity_value = Porosity(Lattice_d_cell,Lattice_d_rod,Lattice_number_cells_x,Scaling_factor_YZ)

col2.header(f'**Processing took :red[{1000*prediction_time:.6f} milliseconds].**')

#col2.header(f'**In the simulation, one experiment takes at least hours for a very simple material.**')

col2.header(f'Different Model Predictions:')
show_df = pd.DataFrame(np.array([[knn_pred, rf_pred, mlp_pred]]), columns = ['KNN Regressor', 'Random Forest Regressor', 'MLP Regressor'], index=['Prediction'])

col2.table(show_df)
#col2.header(f'**KNN Regressor Effective Stiffness Prediction is {knn_pred:.3f}**')

#col2.header(f'**Random Forest Regressor Effective Stiffness Prediction is {rf_pred:.3f}**')

#col2.header(f'**MLP Regressor Effective Stiffness Prediction is {mlp_pred:.3f}**')

col2.header('Material Details:')

col2.header(f'**Porosity: {porosity_value:.3f}**')

#st.header(f'**Point Cloud Processing Network Effective Stiffness Prediction is {pcd_pred:.3f}**')


val_df = pd.read_csv('data\\validation.csv')

val_df_dict = dict(val_df.groupby(['lattice_d_cell', 'lattice_d_rod', 'lattice_number_cells_x', 'scaling_factor_YZ'])['effective_stiffness'].first())

ground_truth = val_df_dict.get((Lattice_d_cell,Lattice_d_rod,Lattice_number_cells_x,Scaling_factor_YZ), -1)


#if ground_truth != -1:
#    col2.header(f':red[**Ground truth is {ground_truth:.3f}**]')
#else:
#    col2.header(f':red[**Ground truth does not exist**]')


showmodel(model_path)
