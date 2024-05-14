Physical Simulation Steps
-------------------------------
- make geometry/mesh/analysis of deformation in Simcenter 3D
- close every file in Simcenter 3D
- Heeds (Anaylsis Portal: Simcenter 3D_2306)
  1. Process -> Simcenter 3D
  2. **Input** sim file (named Hachathon on desktop)
- Select tags and variables
  1. press auto tag -> CAD expressions (lattice_d_cell, lattice_d_rod, scalingfactor_x_y, lattice_num, and all)
  2. select min/max intervals
- **Output** (response)
  1. select blue icon and everything
  2. 
- **Run**
  1. study->methods->design set-> add item -> put numbers in the variables) and experiment with it
  2. go to RUN-tab and run

| variables (CAD expressions)      | computer 1      |computer 2     |computer 3      |
| -------------------------------- | --------------- |---------------|----------------|
| lattice_d_cell [1.9,2.75]        | 1.9,2.375,2.75  |1.9,2.375,2.75 |1.9,2.375,2.75  |
| lattice_d_rod [0.2,1.2]          | step 0.05       |step 0.05      |step 0.05       |
| scalingfactor_x_y [1,6]          | 3               |4              |6               |

the lattice_d_rod has the highest impact on porosity
