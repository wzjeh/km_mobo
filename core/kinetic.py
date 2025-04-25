

import numpy as np
import random
from scipy.integrate import solve_ivp

# Ideal gas constant [J/mol-K]
R = 8.314
# Reference temperature in Kelvin
T_REF = 25 + 273.15


def arrhenius_rate_constant(T):
    """Calculate rate constant k using the Arrhenius equation."""
    return np.exp(61.48 - 126813 / (R * T))


def mc_modifier(C, T):
    """Calculate the MC correction factor as a function of acid concentration and temperature."""
    return (-(2.16e-4 * C ** 5 - 1.27e-2 * C ** 4 + 0.28 * C ** 3 - 2.73 * C ** 2 + 10.6 * C)) * (200 / T + 0.3292)


def reaction(t, C, T_C):
    """
    Define the rate of change of concentrations during the reaction.

    Parameters:
    - t: time
    - C: array of concentrations [NB, NO2+, H2SO4]
    - T_C: temperature in Celsius

    Returns:
    - r: rate of change of concentrations
    """
    T_K = T_C + 273.15
    n = 0.85

    k = arrhenius_rate_constant(T_K)
    MC = mc_modifier(C[2], T_K)
    rate = -k * C[0] * C[1] * 10 ** (n * MC)

    return [rate, 0, 0]  # Only NB is consumed


def Experiment(**conditions):
    """
    Simulate the condensation amidation reaction of piperazine with benzoic acid.

    Input parameters in `conditions` dict:
    - 'Time(s)': reaction residence time (seconds)
    - 'NB(mol/l)': initial concentration of NB
    - 'NO2+(mol/l)': initial concentration of NP2+
    - 'H2SO4(mol/l)': mass concentration of sulfuric acid
    - 'Temperature(C)': reaction temperature in Celsius

    Returns:
    - e_factor: environmental factor
    - Conver_final: conversion of NB
    """
    # Unpack parameters
    Time = conditions['Time(s)']
    NB = conditions['NB(mol/l)']
    NO2 = conditions['NO2+(mol/l)']
    H2SO4 = conditions['H2SO4(mol/l)']
    Temp_C = conditions['Temperature(C)']

    # Initial concentrations vector
    C0 = [NB, NO2, H2SO4]

    # Integrate the ODE system
    sol = solve_ivp(reaction, [0, Time], C0, args=(Temp_C,))
    final = sol.y[:, -1]

    # Add Gaussian noise to results
    random.seed(525)
    final = [max(0, c + random.gauss(0, 0.01)) for c in final]

    # Calculate conversion of NB
    NB_final = final[0]
    Conver_final = max(0, (NB - NB_final) / NB)

    # Calculate e-factor
    e_factor = (
        Time / 200 + Temp_C / 82 + H2SO4 / 18.6 + abs(NO2 - NB) / NB
    ) / np.exp(NB * Conver_final)

    return e_factor, Conver_final
