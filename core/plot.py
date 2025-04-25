import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib import pyplot as plt
import numpy as np
from core.doe import lhs, Y_lhs, scale_data 
from torch import torch, Tensor
from matplotlib.cm import Greens
import os

def max_observed(train_Y: torch.Tensor, X_suggest: int):
    """
    Compute the maximum observed values for both columns (Conversion and E-factor).

    Parameters
    ----------
    train_Y: Tensor
        Experimental results with two columns [Conversion, E-factor].
    X_suggest: int
        Number of candidate points per batch.

    Returns
    -------
    index: list
        Experiment batch indices.
    max_obs_conv: list
        Maximum observed values for Conversion at each batch.
    max_obs_ef: list
        Maximum observed values for E-factor at each batch.
    """
    
    index = []
    max_obs_conv = []
    max_obs_ef = []
    
    for i in range(round(len(train_Y) / X_suggest)):
        # Calculate the maximum value for the current batch (including the first i+1 batches)
        current_max_conv = train_Y[:X_suggest * (i + 1), 1].max()  # First column (Conversion)
        current_max_ef = train_Y[:X_suggest * (i + 1), 0].min()  # Second column (E-factor)
        
        max_obs_conv.append(current_max_conv.item())  
        max_obs_ef.append(current_max_ef.item())  
        index.append(i + 1)  
    
    return index, max_obs_conv, max_obs_ef


# Plot the Pareto Front
def pareto_front(train_Y:Tensor, samples:int, Iterate:int, X_suggest:int, save_path = None):
    """
    Plot the Pareto Front.

    Parameters
    ----------
    train_Y: Tensor
        All experimental results.
    samples: int
        Initial number of experiments.
    Iterate: int
        Number of iterations.
    X_suggest: int
        Number of candidate points.
    save_path: Optional
        The path where the picture is saved.
    """
    fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    cm = plt.cm.get_cmap('viridis')
    batch_number = torch.cat([torch.arange(1,Iterate+1).repeat(X_suggest, 1).t().reshape(-1)]).numpy()
    # Bayesian optimization recommended experimental data
    sc = axes.scatter(train_Y[:,1][samples:].cpu().numpy(), train_Y[:,0][samples:].cpu().numpy(), c=batch_number, cmap=cm, alpha=0.8, s=90)
    # Latin hypercube sampling data is represented by the symbol 'x'
    s_init = axes.scatter(train_Y[:,1][0:samples], train_Y[:,0][0:samples], color='black', alpha=0.8, s=90, marker='x')
    plt.legend([s_init, sc], ['LHS', 'qNEHVI'], fontsize=18, loc=0)
    axes.set_xlabel("Conversion", fontsize=18)
    axes.set_ylabel("E-factor", fontsize=18)
    norm = plt.Normalize(batch_number.min(), batch_number.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([1, 0.18, 0.04, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title("Iteration", fontsize=18)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')


## Plotting the Max Observed with dual y-axes
def plt_max_observed(train_Y: torch.Tensor, samples: int, Iterate: int, X_suggest: int, save_path=None):
    """
    Plot max observed values with dual y-axes.

    Parameters
    ----------
    train_Y: Tensor
        All experimental results.
    samples: int
        Initial number of experiments.
    Iterate: int
        Number of iterations.
    X_suggest: int
        Number of candidate points.
    save_path: Optional
        The path where the picture is saved.
    """
    max_yield = max_observed(train_Y[samples:], X_suggest)
    min_efactor = max_observed(train_Y[samples:], X_suggest)
    batch_number = torch.cat([torch.arange(1, Iterate + 1).repeat(X_suggest, 1).t().reshape(-1)]).numpy()
    x_inter = range(1, (X_suggest + Iterate), 1)  
    
    fig, ax1 = plt.subplots(figsize=(7, 7))
    
    # Create the second y-axis
    ax2 = ax1.twinx()
    
    # Generate color gradient (green)
    norm = plt.Normalize(batch_number.min(), batch_number.max())
    colors = Greens(norm(batch_number))
    
    # First y-axis (Conversion)
    ax1.scatter(batch_number, train_Y[samples:, 1], s=150, c='black', alpha=0.4, label="Sampled Conversion")
    ax1.plot(max_yield[0], max_yield[1], c='black', linewidth=5, label="Max Conversion")
    ax1.scatter(max_yield[0], max_yield[1], s=150, c='black', alpha=0.8)
    ax1.set_xlabel("Iteration", fontsize=18)
    ax1.set_ylabel("Conversion", fontsize=18)
    ax1.set_ylim(0, 1.05)
    
    # Second y-axis (E-factor)
    ax2.scatter(batch_number, train_Y[samples:, 0], s=150, c=colors, alpha=0.6, label="Sampled E-factor")
    ax2.plot(min_efactor[0], min_efactor[2], c='green', linewidth=5, label="Min E-factor")
    ax2.scatter(min_efactor[0], min_efactor[2], s=150, c='green', alpha=0.8)
    ax2.set_ylabel("E-factor", fontsize=18)
    
    # **Combine the legends**
    handles, labels = ax1.get_legend_handles_labels()  # Get ax1's legend
    handles2, labels2 = ax2.get_legend_handles_labels()  # Get ax2's legend
    handles += handles2  # Merge handles
    labels += labels2  # Merge labels
    ax1.legend(handles, labels, loc="lower left", fontsize=11)  # Place legend outside the plot

    plt.xticks(x_inter)
    # Show or save the plot
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
