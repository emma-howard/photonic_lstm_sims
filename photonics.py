"""
PHOTONIC LSTM SIMS V1.0 - photnic helper functions

These functions implement photonic transfer functions for various microring and
other devices

Packages required:
numpy

"""
# Import Helpers
import numpy as np

# Calculates the wavelength response of an all-pass microring resonator configuration
def allpassMRR (phase, a, r):
    T = (a**2 - 2*r*a*np.cos(phase) + r**2)/(1 - 2*r*a*np.cos(phase) + (a*r)**2)
    return T

# Calculates the wavelength response of an add-drop microring resonator configuration, assuming that r1 = r2
def adddropMRR (phase, a, r):
    Tp = ((a*r)**2 - 2*(r**2)*a*np.cos(phase)+r**2)/(1-2*(r**2)*a*np.cos(phase)+((r**2)*a)**2)
    Td = (a*(1-r**2)**2)/(1-2*a*(r**2)*np.cos(phase)+(a*(r**2))**2)
    return Tp, Td

# Calculates the detuning phase of a microring resonator given the refractive index and the wavelength in question
def Phase (lmbda, n, R):
    #c = 2.9979 * 10**8
    phase_shift = R*(2*np.pi)**2/(lmbda/n)
    phase = np.pi - np.remainder(phase_shift, 2*np.pi)
    return phase
