from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import _is_non_dominated_loop
from botorch.fit import fit_gpytorch_mll_torch
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor
import torch
from botorch.models.transforms import Normalize
from core.doe import StandardScaler, scale_data


def initialize_model(train_X: Tensor, train_Y: Tensor):
    """
    Create and fit a single-task Gaussian Process (GP) model.

    Args:
        train_X (Tensor): Training inputs (normalized).
        train_Y (Tensor): Training outputs (normalized).

    Returns:
        model: A trained SingleTaskGP model.
    """
    train_X = train_X.double()
    train_Y = train_Y.double()
    model = SingleTaskGP(train_X, train_Y)
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_mll_torch(mll)
    mll.eval()
    model.eval()
    return model

def compute_hypervolume(Y, ref_point=[3,-1]): 
    """
    Compute the hypervolume for a set of outcomes.

    Args:
        Y (Tensor): Objective values.
        ref_point (Tensor, optional): Reference point for hypervolume calculation.

    Returns:
        float: Hypervolume value.
    """
    Y = torch.stack([Y[:, 0], -Y[:, 1]], dim=1)

    if ref_point is None:

        ref_point = torch.tensor([
            Y[:, 0].min() - 1e-2,
            Y[:, 1].min() - 1e-2
        ])
        
    else:
        ref_point = torch.tensor(ref_point)

    pareto_mask = _is_non_dominated_loop(Y)
    pareto_front = Y[pareto_mask]
    partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_front)
    return partitioning.compute_hypervolume().item()

class bo:
    """
    Bayesian Optimization (BO) class using qNEI and qNEHVI acquisition functions.
    """

    def __init__(self, train_X: Tensor, train_Y: Tensor, model, X_ranges: list, **kwargs):
        """
        Initialize the BO instance.

        Args:
            train_X (Tensor): Normalized training inputs.
            train_Y (Tensor): Normalized training outputs.
            model: Trained GP model.
            X_ranges (list): List of original scale bounds for each feature.
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.model = model
        self.X_ranges = X_ranges
        self.hypervolume = []  # Will store tuples like (iteration_index, hypervolume)
        self.iteration = 1     # Track the current iteration

    def qNEHVI(self, X_suggest: int, ref_point=None, obj1=None, obj2=None, **kwargs):
        """
        Optimize the qNEHVI acquisition function.

        Args:
            X_suggest (int): Number of new candidates to suggest.
            ref_point (list or Tensor, optional): Reference point in objective space.
            obj1, obj2 (optional): Optional objective directions.

        Returns:
            Tuple[Tensor, Tensor]: Suggested points in real scale and unit scale.
        """
        n_dim = self.train_X.shape[1]
        bounds = torch.stack([torch.zeros(n_dim), torch.ones(n_dim)])

        # Normalize objectives, invert if minimizing
        y0 = -StandardScaler(self.train_Y[:, 0]) if obj1 else StandardScaler(self.train_Y[:, 0])
        y1 = -StandardScaler(self.train_Y[:, 1]) if obj2 else StandardScaler(self.train_Y[:, 1])
        train_Y_unit = torch.cat([y0, y1], dim=1)

        if ref_point is None:
            ref_point = Tensor([train_Y_unit[:, 0].min(), train_Y_unit[:, 1].min()])
        else:
            ref_point = Tensor(ref_point)

        model = initialize_model(self.train_X, train_Y_unit)
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            sampler=SobolQMCNormalSampler(512),
            ref_point=ref_point,
            X_baseline=self.train_X,
            prune_baseline=True
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=X_suggest,
            num_restarts=10,
            raw_samples=512,
            sequential=True
        )
        X_new = scale_data(candidates, self.X_ranges, to_unit=0)
        return X_new, candidates

    def qNEI(self, X_suggest: int, **kwargs):
        """
        Optimize the qNEI acquisition function for single-objective problems.

        Args:
            X_suggest (int): Number of new candidates to suggest.

        Returns:
            Tuple[Tensor, Tensor]: Suggested points in real scale and unit scale.
        """
        model = initialize_model(self.train_X, self.train_Y)
        n_dim = self.train_X.shape[1]
        bounds = torch.stack([torch.zeros(n_dim), torch.ones(n_dim)])

        acq_func = qNoisyExpectedImprovement(
            model=model,
            X_baseline=self.train_X,
            sampler=SobolQMCNormalSampler(torch.Size([100]))
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=X_suggest,
            num_restarts=10,
            raw_samples=100
        )
        X_new = scale_data(candidates, self.X_ranges, to_unit=0)
        return X_new, candidates
    
    def random_search(self, X_suggest: int):
        """
        Generate candidates using random sampling (Random Search baseline).

        Args:
            X_suggest (int): Number of candidates to suggest.

        Returns:
            Tuple[Tensor, Tensor]: Suggested points in real scale and unit scale.
        """
        n_dim = len(self.X_ranges)
        unit_samples = torch.rand((X_suggest, n_dim))
        real_samples = scale_data(unit_samples, self.X_ranges, to_unit=0)
        return real_samples, unit_samples

    def update_training_points(self, X_new: Tensor, candidates: Tensor, objective_func: object, Y_obj: int, **kwargs):
        """
        Evaluate new candidate points and update the training data.

        Args:
            X_new (Tensor): New candidates in real scale.
            candidates (Tensor): New candidates in unit scale.
            objective_func (callable): Function to evaluate the objectives.
            Y_obj (int): Number of objectives (1 or 2).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - New experimental results (Y_real)
                - All input points in real scale
                - All input points in unit scale
                - All experimental results
        """
        Y_real = [objective_func(x) for x in X_new]
        Y_real = torch.tensor(Y_real)
        if Y_real.ndim == 1:
            Y_real = Y_real.unsqueeze(1)

        self.train_Y = torch.cat([self.train_Y, Y_real])
        self.train_X = torch.cat([self.train_X, candidates])

        X_real = scale_data(self.train_X, self.X_ranges, to_unit=0)
        return Y_real, X_real, self.train_X, self.train_Y
