import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import os
import json
import argparse

logger = logging.getLogger(__name__)

# Parameters
params = {}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Model parameters
parser.add_argument('-res', '--resolution', type=int, default=1024, help='Resolution')
parser.add_argument('-dt', '--dt', type=float, default=1e-3, help='Time step')
parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed')
parser.add_argument('-output', type=str, default='./output', help='Output directory')
parser.add_argument('-parameters', type=str, default=None, help='Parameters file directory in JSON')
parser.add_argument('-num_iters', type=int, default = 200, help='Number of iterations')

pargs = parser.parse_args()
params.update(vars(pargs))

# Set random seed
np.random.seed(params['seed'])

class Burgers:
    def __init__(self, a=1e-2, b=0):
        self.num_iters = params['num_iters']

        self.params_file = params['parameters']
        self.output_dir = params['output']

        self.viscosity = a
        self.a = a
        self.b = b
        self.dt = params['dt']
        self.resolution = params['resolution']
        self.Lx = 2 * np.pi
        self.Nx = self.resolution
        # TODO: Fix Dealias - should have 3/2 and deal with grid
        self.dealias = 3/2
        # Number of iterations divided by the number of iterations per time unit
        self.stop_sim_time = self.num_iters * self.dt
        self.timestepper = d3.SBDF2
        self.dtype = np.float64
        self.N_force = 20

        # Bases
        self.xcoord = d3.Coordinate('x')
        self.dist = d3.Distributor(self.xcoord, dtype=self.dtype)
        self.xbasis = d3.RealFourier(self.xcoord, size=self.Nx, bounds=(0, self.Lx), dealias=self.dealias)

        # Fields
        self.u = self.dist.Field(name='u', bases=self.xbasis)
        self.F = self.dist.Field(name='F', bases=self.xbasis)

        # Substitutions
        self.dx = lambda A: d3.Differentiate(A, self.xcoord)

        # Problem
        self.problem = d3.IVP([self.u], namespace=locals())
        self.problem.add_equation("dt(self.u) - a*self.dx(self.dx(self.u)) - b*self.dx(self.dx(self.dx(self.u))) = - u*self.dx(self.u) + self.F")

        # Initial conditions
        self.x = self.dist.local_grid(self.xbasis)
        # Start just with zeros
        self.u['g'] = 0

        # Solver
        self.solver = self.problem.build_solver(self.timestepper)
        self.solver.stop_sim_time = self.stop_sim_time

        # Forcing parameters
        if self.params_file is None:
            print("Using random parameters")
            self.A_values = np.random.uniform(-0.5, 0.5, self.N_force)
            self.ω_values = np.random.uniform(-0.4, 0.4, self.N_force)
            self.φ_values = np.random.uniform(0, 2*np.pi, self.N_force)
            self.l_values = np.random.choice([3, 4, 5, 6], self.N_force)
        else: 
            with open(self.params_file) as f:
                data = json.load(f)
                self.A_values = np.array(data['A_values'])
                self.ω_values = np.array(data['ω_values'])
                self.φ_values = np.array(data['φ_values'])
                self.l_values = np.array(data['l_values'])

    def forcing(self, t):
        Force = np.zeros_like(self.x)
        for i in range(self.N_force):
            A = self.A_values[i]
            ω = self.ω_values[i]
            φ = self.φ_values[i]
            l = self.l_values[i]

            Force += A * np.sin(ω * t + 2 * np.pi * l * self.x / (2 * np.pi) + φ)
        return Force

    def simulate(self):
        self.u.change_scales(1)
        self.u_list = [np.copy(self.u['g'])]
        self.t_list = [self.solver.sim_time]
        while self.solver.proceed:
            self.F['g'] = self.forcing(self.solver.sim_time)
            self.solver.step(self.dt)
            if self.solver.iteration % 100 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(self.solver.iteration, self.solver.sim_time, self.dt))
            if self.solver.iteration % 1 == 0:
                self.u.change_scales(1)
                self.u_list.append(np.copy(self.u['g']))
                self.t_list.append(self.solver.sim_time)

    def save_data(self):
        
        os.makedirs(self.output_dir, exist_ok=True)
        np.savez(os.path.join(self.output_dir, 'data.npz'), u_list=self.u_list, t_list=self.t_list, x=self.x)

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
        with open(os.path.join(self.output_dir, 'params.json'), 'w') as f:
            json.dump(params, f)

    def plot(self):
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(self.x.ravel(), np.array(self.t_list), np.array(self.u_list), cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
        plt.xlim(0, self.Lx)
        plt.ylim(0, self.stop_sim_time)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'KdV-Burgers, (a,b)=({self.a},{self.b})')
        plt.tight_layout()
        plt.savefig('kdv_burgers.pdf')
        plt.savefig('kdv_burgers.png', dpi=200)

        # Plot u value at the end of the simulation
        plt.figure(figsize=(6, 4))
        plt.plot(self.x.ravel(), self.u_list[-1])
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('u value at the end of the simulation')
        plt.tight_layout()
        plt.savefig('u_end_simulation.png', dpi=200)

burgers = Burgers()
burgers.simulate()
burgers.save_data()
burgers.save_parameters_to_json()

# burgers.plot()