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
| lattice_d_cell [1.9,2.75]        | 2.375,2.75      |1.9            |1.9,2.375,2.75  |
| lattice_d_rod [0.2,1.2]          | step 0.05       |step 0.05      |step 0.05       |
| scaling_factor_YZ [1,6]          | 3               |3              |6               |
| lattice_num [1,4]                | 2               |2              |2               |
| load                             | 100N            | 100N          | 100N           |

the lattice_d_rod has the highest impact on porosity

Extract data: .xml file
-----------------------------
- Young modulus

  The first a few lines are CAD expressions/parameters, then the displacement vector $U$.
We need to calculate the Young modulus/effective stiffness $E$ from the mean value of $U$, i.e. $\Delta U$
$$E=\frac{\sigma}{\epsilon}=\frac{FU_0}{A\Delta U}$$
where $\sigma = \frac{F}{A}$ is the stress, F is the fixed load on area $A =$ h_total_z $\times$ w_total_y (depending on scaling factor), and $U\_0$ is d_total_x 

- Porosity
  For presentation, we need to calculate porousity $P$
  $$1-P = \frac{V_{mesh}}{V_{total}}$$
  where $V_{total} = $ h_total_z $\times$ w_total_y $\times$ d_total_x, $V_{mesh}$ corresponds to the "Volume" in the data file

