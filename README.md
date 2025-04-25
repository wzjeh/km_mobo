## File Overview

### `main.py`
Reproduces the Pareto front presented in the study using Bayesian optimization techniques. This script performs optimization over multiple objectives and visualizes the resulting trade-offs.

### `pareto_MTL.py`
Compares the performance between Multi-Objective Bayesian Optimization (MOBO) and the Pareto Multi-Task Learning (ParetoMTL) method. It evaluates how well each approach discovers the Pareto front across different iterations.

### `hypervolume.py`
Calculates the hypervolume indicator to assess the quality of the Pareto front. It also compares baseline strategies such as random search against MOBO, providing visualizations of hypervolume progression over multiple runs.
