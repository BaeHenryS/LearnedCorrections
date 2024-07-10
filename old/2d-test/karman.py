from phi.tf.flow import *

class KarmanFlow():
    def __init__(self, domain):
        self.domain = domain

        self.vel_BcMask = self.domain.staggered_grid(HardGeometryMask( Box(y=(None, 5), x=None) ) )
    
        self.inflow = self.domain.scalar_grid(Box(y=(25,75), x=(5,10)) ) # scale with domain if necessary!
        self.obstacles = [Obstacle(Sphere(center=tensor([50, 50], channel(vector="y,x")), radius=10))] 
        #

    def step(self, density_in, velocity_in, re, res, buoyancy_factor=0, dt=1.0):
        velocity = velocity_in
        density = density_in

        # viscosity
        velocity = phi.flow.diffuse.explicit(field=velocity, diffusivity=1.0/re*dt*res*res, dt=dt)
        
        # inflow boundary conditions
        velocity = velocity*(1.0 - self.vel_BcMask) + self.vel_BcMask * (1,0)

        # advection 
        density = advect.semi_lagrangian(density+self.inflow, velocity, dt=dt)
        velocity = advected_velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)

        # mass conservation (pressure solve)
        pressure = None
        velocity, pressure = fluid.make_incompressible(velocity, self.obstacles)
        self.solve_info = { 'pressure': pressure, 'advected_velocity': advected_velocity }
        
        return [density, velocity]
