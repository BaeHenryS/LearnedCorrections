from phi.flow import *
import numpy as np
import os, argparse, sys, random, json
import matplotlib.pyplot as plt

# Parameters
params = {}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Model parameters
parser.add_argument('-res', '--resolution', type=int, default=128, help='Resolution')
parser.add_argument('-dt', '--dt', type=float, default=0.001, help='Time step')
parser.add_argument('-seed', '--seed', type=int, default=0, help='Random seed')
parser.add_argument('-output', type=str, default='./1d-burgers/output', help='Output directory')

pargs = parser.parse_args()
params.update(vars(pargs))

# Set random seed
random.seed(params['seed'])
np.random.seed(params['seed'])

# Burgers equation
class Burgers:
    def __init__(self, viscosity=0.01):
        self.viscosity = viscosity
        self.INITIAL = math.tensor(np.zeros([params['resolution']]), spatial('x'))
        self.domain = CenteredGrid(self.INITIAL, extrapolation.PERIODIC, x=int(params['resolution']), bounds=Box['x', slice(0,2*np.pi)])
        self.forcing = np.zeros_like(np.linspace(0, 2*np.pi, int(params['resolution'])))
        self.N_force = 20
        self.A_values = np.random.uniform(-0.5, 0.5, self.N_force)
        self.ω_values = np.random.uniform(-0.4, 0.4, self.N_force)
        self.φ_values = np.random.uniform(0, 2*np.pi, self.N_force)
        self.l_values = np.random.choice([3, 4, 5, 6], self.N_force)
        self.max_steps = 500
        

        # Create a scene for storing simulation data
        self.scene = Scene.create(parent_directory=params['output'])

        self.velocities = [self.domain]

        self.xgrid = np.linspace(0, 2*np.pi, int(params['resolution']))


    def generate_forcing(self, T=0, N_force=20):
        # Generate forcing field
        self.forcing = np.zeros_like(self.xgrid)
        for j in range(N_force):
            # Use pre-generated parameters
            A = self.A_values[j]
            ω = self.ω_values[j]
            φ = self.φ_values[j]
            l = self.l_values[j]

            # Add sinusoidal function to f
            self.forcing += A * np.sin(ω * T + 2 * np.pi * l * self.xgrid / (2 * np.pi) + φ)

        self.forcing = self.forcing.reshape(-1)
        self.forcing = math.tensor(self.forcing, spatial('x'))

    def simulate(self):
        for i in range(self.max_steps):
            # Generate forcing field
            self.generate_forcing(T=i * params['dt'])

            # Solve Burgers equation
            v1 = diffuse.explicit(self.velocities[-1], self.viscosity, params['dt']) + self.forcing
            v2 = advect.semi_lagrangian(v1, v1, params['dt'])

            self.velocities.append(v2)

            # Save data to the scene
            self.save_simulation_data(i)

    def save_simulation_data(self, frame):
        """Save velocity data to the scene for the given frame."""
        self.scene.write({'velocity': self.velocities[frame]}, frame=frame)
    
    def save_parameters_to_json(self):
        """Save parameters to a JSON file."""
        params = {
            'viscosity': self.viscosity,
            'N_force': self.N_force,
            'A_values': self.A_values.tolist(),
            'ω_values': self.ω_values.tolist(),
            'φ_values': self.φ_values.tolist(),
            'l_values': self.l_values.tolist()
        }
        with open(os.path.join(self.scene.path, 'params.json'), 'w') as f:
            json.dump(params, f)
    
    def plot_velocity(self):
        plt.figure(figsize=(10, 6))
        total_steps = len(self.velocities)
        for i, v in enumerate(self.velocities):
            if i == total_steps // 4 or i == total_steps // 2 or i == 3 * total_steps // 4 or i == total_steps - 1:
                plt.plot(v.values.numpy(), label=f'Time step {i}')
        plt.legend()
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Velocity over time')
        plt.show()

burgers = Burgers()
burgers.simulate()
burgers.save_parameters_to_json()
