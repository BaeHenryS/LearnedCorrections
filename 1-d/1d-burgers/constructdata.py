from phi.flow import *
import numpy as np
import os, argparse, sys, random, json

from burgers_1d import Burgers_1d

def construct_hires_data(sim_num, resolution, viscosity, dt, output, parameters, num_iters):  
    sim = Burgers_1d(resolution, viscosity, dt, sim_num, output, parameters, num_iters)
    sim.simulate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate high-resolution Burgers_1d simulations.')
    parser.add_argument('--sim_num', type=int, default=0, help='Simulation number')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the simulation')
    parser.add_argument('--viscosity', type=float, default=0.01, help='Viscosity parameter')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--parameters', type=str, default=None, help='Parameters file')
    parser.add_argument('--num_iters', type=int, default=1000, help='Number of iterations')

    args = parser.parse_args()

    construct_hires_data(args.sim_num, args.resolution, args.viscosity, args.dt, args.output, args.parameters, args.num_iters)