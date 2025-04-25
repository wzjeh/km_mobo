import numpy as np 
import torch
from core.kinetic import Experiment
from core.doe import scale_data
from torch import Tensor

# Reaction function
def reaction(X):
    X = np.array(X)
    Conditions = {
        'Time(s)': X[0],
        'Temperature(C)': X[1],
        'NB(mol/l)': X[2],
        'NO2+(mol/l)': X[3],
        'H2SO4(mol/l)': X[4]
    }
    Y = Experiment(**Conditions)
    return Y  # e-factor and Conversion


# Design space
X_ranges = [
    [0, 200],    # Time
    [60, 82],    # Temperature
    [0, 1],      # NB concentration
    [0, 2],      # NO2+ concentration
    [15, 18.4]   # H2SO4 concentration
]

# Random sampling function with fixed seed
def random_search(X_ranges, X_suggest: int, seed: int = 44234):
    """
    Generate candidates using random sampling with a fixed random seed.

    Args:
        X_ranges (list): List of original scale bounds for each variable.
        X_suggest (int): Number of candidates to suggest.
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[Tensor, Tensor]: Real scale samples and unit scale samples.
    """

    torch.manual_seed(seed)  
    n_dim = len(X_ranges)
    unit_samples = torch.rand((X_suggest, n_dim))
    real_samples = scale_data(unit_samples, X_ranges, to_unit=0)
    return real_samples, unit_samples

def Y_lhs(lhs_unit: Tensor, objective_func: object, X_ranges: list, Y_obj: int): 
    """
    Evaluate the objective function on a set of input candidates.

    Args:
        lhs_unit (Tensor): Input samples in unit scale (normalized between 0 and 1).
        objective_func (object): Callable objective function to evaluate.
        X_ranges (list): List of original scale bounds for each variable (not used directly here, but relevant in some workflows).
        Y_obj (int): Index of the target objective (used internally by objective_func, if applicable).

    Returns:
        Tensor: Evaluation results in a 2D tensor format (N x 1 if single objective).
    """
    results = []
    for i in lhs_unit:
        y = objective_func(i)
        results.append(y)
    
    results = Tensor(results)
    if results.ndim == 1:
        results = results.reshape(len(results), 1)
    return results

# main
def run_random_sampling(samples: int):
    """
    Run random sampling and get initial evaluation results.

    Args:
        samples (int): Number of random samples to generate.

    Returns:
        Tensor: Evaluation results (random_Y)
    """
    random_X, _ = random_search(X_ranges=X_ranges, X_suggest=samples)
    random_Y = Y_lhs(random_X, reaction, X_ranges, Y_obj=2)
    return random_Y
