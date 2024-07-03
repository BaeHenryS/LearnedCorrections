from phi.flow import *
import numpy as np
import os, argparse, sys, random, json

from burgers_1d import Burgers_1d

def construct_hires_data(num_sims, resolution, viscosity, dt, output, parameters, num_iters):
    for i in range(num_sims):
        sim = Burgers_1d(resolution, viscosity, dt, i, output, parameters, num_iters)
        sim.simulate()


construct_hires_data(10, 512, 0.01, 0.001, './output', None, 200)
