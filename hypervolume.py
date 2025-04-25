import subprocess
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import _is_non_dominated_loop
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from core.random_search import run_random_sampling
import core.mobo

# === 1. Hypervolume Computation Function ===
def compute_hypervolume(Y, ref_point=None):
    """
    Compute the hypervolume (HV) of a set of multi-objective outcomes.

    Args:
        Y (Tensor): A 2D tensor of shape (N, 2) representing objective values.
        ref_point (Tensor, optional): Reference point for HV calculation. If None, 
                                      a slightly expanded bound based on current Y is used.

    Returns:
        float: Calculated hypervolume.
    """
    # Invert the second objective (assumed to be maximized)
    Y = torch.stack([Y[:, 0], -Y[:, 1]], dim=1)

    if ref_point is None:
        # Auto-generate a conservative reference point
        ref_point = torch.tensor([
            Y[:, 0].min() - 1e-2,
            Y[:, 1].min() - 1e-2
        ])
    else:
        ref_point = torch.tensor(ref_point)

    # Identify Pareto-efficient points
    pareto_mask = _is_non_dominated_loop(Y)
    pareto_front = Y[pareto_mask]

    # Compute HV using dominated partitioning
    partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_front)
    return partitioning.compute_hypervolume().item()

# === 2. Load Y_all from main.py and compute HV for BO ===
def bo_and_get_hv():
    """
    Run Bayesian Optimization via main.py and compute hypervolume per iteration.

    Returns:
        list: Hypervolume at each iteration.
    """
    subprocess.run(["python", "main.py"], check=True)  # Run main optimization script
    Y_all_list = torch.load("Y_all_iter.pt")  # Load saved results from main.py

    # Use initial evaluations to construct progressive subsets
    first_tensor = Y_all_list[0]
    accumulated_tensors = [first_tensor[:i+1] for i in range(first_tensor.shape[0])]
    Y_all_list = accumulated_tensors + Y_all_list[1:]

    hv_per_iter = []
    for Y in Y_all_list:
        hv = compute_hypervolume(Y)
        hv_per_iter.append(hv)
    return hv_per_iter

# === 3. Run Random Search and Compute HV ===
def random_and_get_hv(runs=20):
    """
    Run random sampling and compute hypervolume progression.

    Args:
        runs (int): Number of random samples.

    Returns:
        list: Hypervolume per accumulated step.
    """
    Y_all_random = run_random_sampling(runs)
    Y_accumulated = [Y_all_random[:i+1] for i in range(Y_all_random.shape[0])]

    hv_per_run = []
    for Y in Y_accumulated:
        hv = core.mobo.compute_hypervolume(Y)
        hv_per_run.append(hv)
    return hv_per_run

# === 4. Plot Hypervolume Progression for All Runs ===
def plot_compare_hypervolumes(all_hv_dict):
    """
    Plot the hypervolume progression curves for multiple optimization methods.

    Args:
        all_hv_dict (dict): Dictionary of the form 
                            {'BO': [[...], [...]], 'Random': [[...], [...]]}
                            where each value is a list of HV lists per run.

    Returns:
        Figure: Matplotlib figure object.
    """
    method_names = list(all_hv_dict.keys())
    n_runs = len(next(iter(all_hv_dict.values())))
    fig, axs = plt.subplots(4, 3, figsize=(12, 14), sharex=True, sharey=True)
    axs = axs.flatten()

    for run_idx in range(n_runs):
        ax = axs[run_idx]
        for method in method_names:
            hv_list = all_hv_dict[method][run_idx]
            ax.plot(range(1, len(hv_list) + 1), hv_list, label=method)
        ax.set_title(f'Run {run_idx + 1}')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Hypervolume")
        ax.legend()
    
    fig.suptitle("Hypervolume Comparison per Run", fontsize=16, y=0.92)
    return fig

# === 5. Main Script for Executing BO and Random Search Comparison ===
if __name__ == "__main__":
    n_runs = 12  # Number of independent repetitions for each method

    all_hv_bo = []
    print(f"Running BO {n_runs} times...")
    for i in range(n_runs):
        print(f"BO Run {i+1}")
        hv_list = bo_and_get_hv()
        all_hv_bo.append(hv_list)

    all_hv_random = []
    print(f"Running Random Search {n_runs} times...")
    for i in range(n_runs):
        print(f"Random Run {i+1}")
        hv_list = random_and_get_hv(runs=20)
        all_hv_random.append(hv_list)

    # Compare HV curves of the two methods
    hv_dict = {
        "Bayesian Optimization": all_hv_bo,
        "Random Search": all_hv_random
    }

    fig = plot_compare_hypervolumes(hv_dict)
    fig.savefig("hv_compare_BO_vs_Random.png", dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
