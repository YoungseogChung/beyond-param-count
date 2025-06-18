# Soft MoE for MNIST Experiments

This is the codebase we used to run the MNIST experiments.

## Code Overview

We have included code to 1) train the suite of Soft MoE models, and 2) run the expert subset selection experiments. 


To train the suite of Soft MoE models, refer to the following file
- `train_mnist_model_suite.py`

The experiment code is in the following files:
- `experiment_scripts/run_heuristic_on_mnist_models.py`
- `experiment_scripts/run_random_dropping_on_mnist_models.py`

