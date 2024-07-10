from phi.tf.flow import *
import numpy as np
import os, argparse, sys, random, json, time
import matplotlib.pyplot as plt
import tensorflow as tf


from numba import jit
    

class Burgers_1d:
    def __init__(self, resolution=512,viscosity = 0.01, dt=0.001, seed=42, output='./output', parameters=None, num_iters=2001):
        self.resolution = resolution
        self.dt = dt
        self.seed = seed
        self.output = output
        self.parameters = parameters
        self.num_iters = num_iters

        # Set random seed
        # Disable for Train

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.viscosity = viscosity 
        self.INITIAL = math.tensor(np.zeros([self.resolution]), spatial('x'))
        self.velocity = CenteredGrid(self.INITIAL, extrapolation.PERIODIC, x=int(self.resolution), bounds=Box['x', slice(0,2*np.pi)])
        self.forcing = np.zeros_like(np.linspace(0, 2*np.pi, int(self.resolution)))
        self.N_force = 20

        if self.parameters is None:
            self.A_values = np.random.uniform(-0.5, 0.5, self.N_force)
            self.ω_values = np.random.uniform(-0.4, 0.4, self.N_force)
            self.φ_values = np.random.uniform(0, 2*np.pi, self.N_force)
            self.l_values = np.random.choice([3, 4, 5, 6], self.N_force)
        else: 
            with open(self.parameters) as f:
                data = json.load(f)
                self.A_values = np.array(data['A_values'])
                self.ω_values = np.array(data['ω_values'])
                self.φ_values = np.array(data['φ_values'])
                self.l_values = np.array(data['l_values'])

        self.max_steps = self.num_iters

        # Create a scene for storing simulation data
        for attempt in range(5):
            try:
                self.scene = Scene.create(parent_directory=self.output)
                break  # Exit the loop if successful
            except Exception as e:
                if attempt < 10:  # Wait and retry for the first 4 attempts
                    print(f"Attempt {attempt + 1} failed, retrying in 10 seconds...")
                    time.sleep(10)
                else:  # Terminate if all attempts fail
                    print("Failed to create a scene after 5 attempts. Terminating program.")
                    sys.exit(1)

        self.xgrid = np.linspace(0, 2*np.pi, int(self.resolution))

    def reset(self):
        self.velocity = CenteredGrid(self.INITIAL, extrapolation.PERIODIC, x=int(self.resolution), bounds=Box['x', slice(0,2*np.pi)])
        self.forcing = np.zeros_like(np.linspace(0, 2*np.pi, int(self.resolution)))

    def change_forcing_dir(self, new_dir):
        with open(new_dir) as f:
            data = json.load(f)
            self.A_values = np.array(data['A_values'])
            self.ω_values = np.array(data['ω_values'])
            self.φ_values = np.array(data['φ_values'])
            self.l_values = np.array(data['l_values'])


    # Numpy Version - Generated Data
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

    # def generate_forcing(self, T=0, N_force=20):
    #     # Generate forcing field
    #     self.forcing = np.zeros_like(self.xgrid, dtype=np.float32)  # Ensure xgrid is float32
    #     for j in range(N_force):
    #         # Use pre-generated parameters
    #         A = self.A_values[j]
    #         ω = self.ω_values[j]
    #         φ = self.φ_values[j]
    #         l = self.l_values[j]

    #         # Ensure all variables are cast to float32 before the operation
    #         T_float = tf.cast(T, tf.float32)
    #         ω_float = tf.cast(ω, tf.float32)
    #         l_float = tf.cast(l, tf.float32)
    #         φ_float = tf.cast(φ, tf.float32)
    #         xgrid_float = tf.cast(self.xgrid, tf.float32)

    #         # Add sinusoidal function to f
    #         self.forcing += A * tf.sin(ω_float * T_float + 2 * np.pi * l_float * xgrid_float / (2 * np.pi) + φ_float)

    #     self.forcing = tf.reshape(self.forcing, [-1])
    #     self.forcing = math.tensor(self.forcing, spatial('x'))  
    
    def simulate(self):
        for i in range(self.max_steps):
            self.velocity = self.step(velocity_in=self.velocity, t=i * self.dt, dt=self.dt)
            self.save_simulation_data(self.velocity, i)

        self.save_parameters_to_json()

   

    # def step(self, velocity_in, t = 0,dt = 0.001):
    #     # Generate forcing field
    #     self.generate_forcing(T=t)
    #     # Solve Burgers equation
    #     v1 = diffuse.explicit(velocity_in, self.viscosity, dt) + self.forcing * dt
    #     v2 = advect.semi_lagrangian(v1, v1, dt)
    #     return v2
    
    def step(self, velocity_in, t=0, dt=0.001):
        # k1 is the slope at the start of the interval
        k1 = self.equation(velocity_in, t)
        
        # k2 is the slope at the midpoint, using k1 to estimate y at t + dt/2
        y_temp = velocity_in + 0.5 * dt * k1
        k2 = self.equation(y_temp, t + 0.5 * dt)
        
        # k3 is another slope at the midpoint, using k2 for estimation
        y_temp = velocity_in + 0.5 * dt * k2
        k3 = self.equation(y_temp, t + 0.5 * dt)
        
        # k4 is the slope at the end of the interval, using k3 for estimation
        y_temp = velocity_in + dt * k3
        k4 = self.equation(y_temp, t + dt)
        
        # Combine the slopes to estimate the new state
        velocity_out = velocity_in + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return velocity_out
    
    

    def step_corrected(self, velocity_in, t=0, dt=0.001):
        # Same as the step, except using self.equation_corrected


    def equation_noforce(self, velocity):
        advection = advect.finite_difference(velocity, velocity, order=2, implicit=False)
        diffusion = diffuse.finite_difference(velocity, self.viscosity, order=2, implicit=False)
        return advection + diffusion

    def equation(self, velocity, t):
        self.generate_forcing(T=t)
        advection = advect.finite_difference(velocity, velocity, order=2, implicit=False)
        diffusion = diffuse.finite_difference(velocity, self.viscosity, order=2, implicit=False)
        # Ensure shape compatibility here, possibly adjusting self.forcing or using broadcasting
        return advection + diffusion + self.forcing# * self.dt
        
    def downsample4x(self, velocity):
        return CenteredGrid(math.downsample2x(math.downsample2x(velocity.values, velocity.extrapolation)), bounds=velocity.bounds, extrapolation=velocity.extrapolation)

    def save_simulation_data(self, velocity, frame):
        """Save downsampled velocity data to the scene for the given frame."""
        downsampled_velocity = self.downsample4x(velocity)
        advection_diffusion = self.equation_noforce(velocity)
        downsampled_advection_diffusion = self.downsample4x(advection_diffusion)
        self.scene.write({'velocity': downsampled_velocity, 'advection_diffusion': downsampled_advection_diffusion}, frame)

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
    