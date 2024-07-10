import numpy as np
import os
from phi.tf.flow import *
from numba import jit
import time

class NavierStokes():
    def __init__(self, output='./output', steps=1000, viscosity=0.1):
        self.output = output
        self.steps = steps
        self.viscosity = viscosity

        self.velocity = StaggeredGrid(Noise(), 'periodic', x=512, y=512)
        # Attempt to create the scene with retries
        for attempt in range(10):
            try:
                self.scene = Scene.create(parent_directory=self.output)
                break  # Break out of the loop if successful
            except Exception as e:  # Catch any exception
                if attempt < 4:  # Check if it's not the last attempt
                    print(f"Attempt {attempt + 1} failed, trying again in 10 seconds...")
                    time.sleep(10)  # Wait for 10 seconds before retrying
                else:
                    print("Failed to create scene after 10 attempts.")
                    raise  # Re-raise the last exception after all attempts fail

        self.velocity0, _ = fluid.make_incompressible(self.velocity)
        self.advection_diffusion_sum_combined = [self.downsample4x_staggered(self.momentum_equation(self.velocity0))]

        self.velocity_trj = None
        self.pressure_trj = None

    def downsample4x_centered(self, velocity):
        return CenteredGrid(math.downsample2x(math.downsample2x(velocity.values, velocity.extrapolation)), bounds=velocity.bounds, extrapolation=velocity.extrapolation)
    
    def downsample4x_staggered(self, velocity):
        return StaggeredGrid(math.downsample2x(math.downsample2x(velocity.values, velocity.extrapolation)), bounds=velocity.bounds, extrapolation=velocity.extrapolation)

    def momentum_equation(self, v):
        advection = advect.finite_difference(v, v, order=4, implicit=None)
        diffusion = diffuse.finite_difference(v, self.viscosity, order=4, implicit=None)
        return advection + diffusion

    def rk4_step(self, v, p, dt):
        advection_diffusion_sum = self.momentum_equation(v)
        advection_diffusion_sum_downsampled = self.downsample4x_staggered(advection_diffusion_sum)
        self.advection_diffusion_sum_combined.append(advection_diffusion_sum_downsampled)

        return fluid.incompressible_rk4(self.momentum_equation, v, p, dt, pressure_order=4)

    def simulate_and_generate(self, dt=0.01):
        velocity0, pressure0 = fluid.make_incompressible(self.velocity)
        self.scene.write({'velocity': velocity0, 'pressure': pressure0, 'advection_diffusion_sum': self.advection_diffusion_sum_combined[0]}, frame=0)
        self.velocity_trj, self.pressure_trj = iterate(self.rk4_step, batch(time=self.steps), velocity0, pressure0, dt=dt)
        
        for i, (velocity, pressure) in enumerate(zip(self.velocity_trj.time, self.pressure_trj.time)):
            velocity = self.downsample4x_staggered(velocity)
            pressure = self.downsample4x_centered(pressure)
            self.scene.write({'velocity': velocity, 'pressure': pressure, 'advection_diffusion_sum': self.advection_diffusion_sum_combined[i]}, frame=i)
        
        print('Simulation complete. Output saved to', self.output)