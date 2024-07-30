from phi.tf.flow import *
import numpy as np
import os, argparse, sys, random, json, time
import matplotlib.pyplot as plt
import tensorflow as tf


from numba import jit



def transformation_batch(velocity_functions, output_datas):
    """
    Apply a 1D convolution transformation to a batch of velocity functions with a batch of output data.
    
    Parameters:
    - velocity_functions: A batch of velocity functions, shape (batch_size, length, 1).
    - output_datas: A batch of output data for convolution filters, shape (batch_size, filter_length, 1, 1).
    
    Returns:
    - A batch of convolved data, each with shape (length,).
    """
    # Ensure velocity_functions is 4D for batched 1D convolution: (batch_size, width, in_channels, out_channels)
    velocity_functions = tf.reshape(velocity_functions, [velocity_functions.shape[0], -1, 1, 1])
    
    # Ensure output_datas is 4D for filters in batched 1D convolution: (batch_size, filter_width, in_channels, out_channels)
    output_datas = tf.reshape(output_datas, [output_datas.shape[0], -1, 1, 1])
    
    # Perform the batched 1D convolution
    convolved_datas = tf.nn.depthwise_conv2d(velocity_functions, output_datas, strides=[1, 1, 1, 1], padding='SAME')
    
    # Squeeze the output to remove unnecessary dimensions, resulting in a batch of shape (batch_size, length)
    return tf.squeeze(convolved_datas, axis=[2, 3])

def to_phiflow_format_batch(datas):
    """
    Convert a batch of data to a batch of PhiFlow CenteredGrid objects and stack them.
    
    Parameters:
    - datas: A batch of data, shape (batch_size, length).
    
    Returns:
    - A stacked CenteredGrid object representing the batch.
    """
    grids = []
    for data in datas:
        grid = CenteredGrid(math.tensor(data, spatial('x')), extrapolation.PERIODIC, x=int(128), bounds=Box['x', slice(0, 2 * np.pi)])
        grids.append(grid)
    
    # Stack the CenteredGrid objects to handle the batch dimension
    stacked_grid = stack(grids, batch('batch'))
    return stacked_grid







def transformation(velocity_function, output_data):
    # Reshape velocity_function to (1, length, 1) and output_data to (filter_length, 1, 1)
    velocity_function = tf.reshape(velocity_function, [1, -1, 1])  # (1, 128, 1)
    output_data = tf.reshape(output_data, [-1, 1, 1])  # (8, 1, 1)

    # Perform the 1D convolution
    convolved_data = tf.nn.conv1d(velocity_function, output_data, stride=1, padding='SAME')

    # Squeeze the output to remove unnecessary dimensions, resulting in (128,)
    return tf.squeeze(convolved_data, axis=[0, 2])


def to_phiflow_format(data):
    return CenteredGrid(math.tensor(data, spatial('x')), extrapolation.PERIODIC, x=int(128), bounds=Box['x', slice(0, 2 * np.pi)])

def to_numpy_format(data):
    return data.values.numpy('x')


def to_tensor_format(data):
    return tf.convert_to_tensor(data.values.numpy('x'))





class Burgers_1d:
    def __init__(self, resolution=512,viscosity = 0.01, dt=0.001, seed=42, output='./output_testing', parameters=None, num_iters=2001, save_original = False, batch_size = 32):
        self.resolution = resolution
        self.dt = dt
        self.seed = seed
        self.output = output
        self.parameters = parameters
        self.num_iters = num_iters
        self.save_original = save_original
        self.batch_size = batch_size

        # Set random seed
        # Disable for Train

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.viscosity = viscosity 
        self.INITIAL = math.tensor(np.zeros([self.resolution]), spatial('x'))
        self.velocity = CenteredGrid(self.INITIAL, extrapolation.PERIODIC, x=int(self.resolution), bounds=Box['x', slice(0,2*np.pi)])
        self.forcing = np.zeros_like(np.linspace(0, 2*np.pi, int(self.resolution)))
        self.N_force = 20


        self.forcing_batch = np.zeros((batch_size, self.N_force, 4))


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
    # def generate_forcing(self, T=0, N_force=20):
    #     # Generate forcing field
    #     self.forcing = np.zeros_like(self.xgrid)
    #     for j in range(N_force):
    #         # Use pre-generated parameters
    #         A = self.A_values[j]
    #         ω = self.ω_values[j]
    #         φ = self.φ_values[j]
    #         l = self.l_values[j]

    #         # Add sinusoidal function to f
    #         self.forcing += A * np.sin(ω * T + 2 * np.pi * l * self.xgrid / (2 * np.pi) + φ)

    #     self.forcing = self.forcing.reshape(-1)
    #     self.forcing = math.tensor(self.forcing, spatial('x'))

    def generate_forcing(self, T=0, N_force=20):
        # Generate forcing field
        self.forcing = np.zeros_like(self.xgrid, dtype=np.float32)  # Ensure xgrid is float32
        for j in range(N_force):
            # Use pre-generated parameters
            A = self.A_values[j]
            ω = self.ω_values[j]
            φ = self.φ_values[j]
            l = self.l_values[j]

            # Ensure all variables are cast to float32 before the operation
            T_float = tf.cast(T, tf.float32)
            ω_float = tf.cast(ω, tf.float32)
            l_float = tf.cast(l, tf.float32)
            φ_float = tf.cast(φ, tf.float32)
            xgrid_float = tf.cast(self.xgrid, tf.float32)

            # Add sinusoidal function to f
            self.forcing += A * tf.sin(ω_float * T_float + 2 * np.pi * l_float * xgrid_float / (2 * np.pi) + φ_float)

        self.forcing = tf.reshape(self.forcing, [-1])
        self.forcing = math.tensor(self.forcing, spatial('x'))  
    
    def simulate(self):
        self.save_parameters_to_json()
        for i in range(self.max_steps):
            self.velocity = self.step(velocity_in=self.velocity, t=i * self.dt, dt=self.dt)
            if self.save_original:
                self.save_original_simulation_data(self.velocity, i)
            else:
                self.save_simulation_data(self.velocity, i)

    def generate_forcing_with_params(self, T=0, params=None):
        # Generate forcing field
        forcing = np.zeros_like(self.xgrid, dtype=np.float32)  # Ensure xgrid is float32
        for j in range(len(params)):
            A = params[j][0]
            ω = params[j][1]
            φ = params[j][2]
            l = params[j][3]

            T_float = tf.cast(T, tf.float32)
            ω_float = tf.cast(ω, tf.float32)
            l_float = tf.cast(l, tf.float32)
            φ_float = tf.cast(φ, tf.float32)
            xgrid_float = tf.cast(self.xgrid, tf.float32)

            forcing += A * tf.sin(ω_float * T_float + 2 * np.pi * l_float * xgrid_float / (2 * np.pi) + φ_float)
        return forcing

    def generate_batch_forcing(self, T):
        """
        Generate a batch of forcing terms for an array of T values.

        Parameters:
        - T: An array of current times.

        Returns:
        - A tensor of shape (batch, 128) with the forcing term for each data point in the batch.
        """
        self.forcing_batch = phi.tf.flow.stack([to_phiflow_format(self.generate_forcing_with_params(T[i], self.batch_params[i])) for i in range(self.batch_size)], phi.tf.flow.batch('batch'))


        
    def set_batch_params(self, batch_params):
        self.batch_params = batch_params

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
    
    

    def step_corrected(self, velocity_in, t=0, dt=0.001, model = None):
        # Same as the step, except using self.equation_corrected
        
        k1 = self.equation_corrected(velocity_in, t, model)
        y_temp = velocity_in + 0.5 * dt * k1 
        k2 = self.equation_corrected(y_temp, t + 0.5 * dt, model)
        y_temp = velocity_in + 0.5 * dt * k2
        k3 = self.equation_corrected(y_temp, t + 0.5 * dt, model)
        y_temp = velocity_in + dt * k3
        k4 = self.equation_corrected(y_temp, t + dt, model)
        velocity_out = velocity_in + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return velocity_out

    def equation_corrected(self, velocity, t, model):
        
        self.generate_forcing(T=t)

        equation = self.equation(velocity, t)
        
        velocity_numpy = tf.expand_dims(tf.expand_dims(velocity.values, 0), -1)
        if model is not None:
            output = model(velocity_numpy)
            correction = transformation(velocity_numpy, output)
            return equation + to_phiflow_format(correction)

        return equation + self.forcing
    
    def equation_corrected_noforce(self, velocity, t, model):
        equation = self.equation(velocity, t)
        
        velocity_numpy = tf.expand_dims(tf.expand_dims(velocity.values, 0), -1)
        if model is not None:
            output = model(velocity_numpy)
            correction = transformation(velocity_numpy, output)
            return equation + to_phiflow_format(correction)

        return equation
    
    def equation_batch(self, velocities, t):
        """
        Compute the base equation for a batch of velocities, including forcing terms.

        Parameters:
        - velocities: A batch of velocity fields.
        - t: The current time.

        Returns:
        - The base equation for each velocity in the batch with the forcing term added.
        """
        # Generate forcing for the batch
        self.generate_batch_forcing(T=t)

        # Compute the base equation for each velocity in the batch
        advection = advect.finite_difference(velocities, velocities, order=2, implicit=False)
        diffusion = diffuse.finite_difference(velocities, self.viscosity, order=2, implicit=False)
        # Ensure shape compatibility here, possibly adjusting self.forcing or using broadcasting

        return advection + diffusion + self.forcing_batch
        

    def equation_corrected_batch(self, velocities, t, model):
        """
        Apply corrections to the equation for a batch of velocities, including forcing terms.
        
        Parameters:
        - velocities: A batch of velocity fields.
        - t: The current time.
        - model: The correction model to be applied.
        
        Returns:
        - The corrected equation for each velocity in the batch.
        """
        # Generate forcing for the batch
        self.generate_batch_forcing(T=t)

        # Compute the base equation for each velocity in the batch
        equations = self.equation(velocities, t)
        
        # Prepare velocities for the model
        velocities_numpy = tf.expand_dims(velocities, -1)
        
        if model is not None:
            # Apply the model to each velocity in the batch
            outputs = model(velocities_numpy)
            corrections = transformation_batch(velocities_numpy, outputs)
            corrected_equations = equations + to_phiflow_format_batch(corrections)
        else:
            corrected_equations = equations
        
        # Add the forcing term to each corrected equation
        return corrected_equations + self.forcing_batch

    def equation_corrected_noforce_batch(self, velocities, t, model):
        """
        Apply corrections to the equation for a batch of velocities, excluding forcing terms.
        
        Parameters:
        - velocities: A batch of velocity fields.
        - t: The current time.
        - model: The correction model to be applied.
        
        Returns:
        - The corrected equation for each velocity in the batch.
        """
        # Compute the base equation for each velocity in the batch
        equations = self.equation(velocities, t)
        
        # Prepare velocities for the model
        velocities_numpy = tf.expand_dims(velocities, -1)
        
        if model is not None:
            # Apply the model to each velocity in the batch
            outputs = model(velocities_numpy)
            corrections = transformation_batch(velocities_numpy, outputs)
            corrected_equations = equations + to_phiflow_format_batch(corrections)
        else:
            corrected_equations = equations
        
        return corrected_equations







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

    def save_original_simulation_data(self, velocity, frame):
        """Save original velocity data to the scene for the given frame."""
        advection_diffusion = self.equation_noforce(velocity)
        self.scene.write({'velocity': velocity, 'advection_diffusion': advection_diffusion}, frame) 

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
    