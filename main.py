"""
PHOTONIC LSTM SIMS V1.0 - main

Photonic & Analog Elelctronic Long-Short Term Memory System Simulations
Emma R Howard - May 2020

Run this to do integrated photonic/electronic sims

Packages required:
numpy
matplotlib

"""
# Import #######################################################################

# Import Helpers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Import functions from the Photonics helper library
from photonics import Phase, adddropMRR, allpassMRR

# Import functions from the Circuits helper library
# LATER PROBLEM 

# Constants ####################################################################

# Photonic Properties of the Microrings
a = 0.99 # Propogation loss
r = 0.99 # Self-Coupling coefficient
n = 3.42 # Index of refraction of silicon microring

# Other
lmbda = np.linspace(1500*10**-9, 1510*10**-9, 3) #System  laser wavelengths


# Code #########################################################################

# Simulation 1 - Simulate a MRR All-Pass notch filter with 3 wavelengths

R = 5*10**-6 # Microring resonator radius, m
detuning = np.linspace(-np.pi/2, np.pi/2, 100)
t = allpassMRR(detuning, a, r)

plt.figure()
plt.plot(detuning, t)
plt.savefig('detuning.png', bbox_inches='tight')
