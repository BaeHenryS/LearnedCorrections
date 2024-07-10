from navierstokes import NavierStokes

navier_stokes_simulation = NavierStokes(output='./simulation_output', viscosity=0.1)
navier_stokes_simulation.simulate_and_generate()