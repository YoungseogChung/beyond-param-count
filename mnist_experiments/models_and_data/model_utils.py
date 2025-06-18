import torch


def construct_mlp_small(num_inputs: int, num_outputs: int, multiplier: int = 1):
    """2 layer MLP with ReLU activation

    Args:
        num_inputs (int): number of inputs
        num_outputs (int): number of outputs
        multiplier (int, optional): multiplier for number of units in hidden layer.
            Defaults to 1.

    Returns:
        torch model: 2 layer MLP
    """
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_inputs, int(num_outputs * multiplier), bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(int(num_outputs * multiplier), num_outputs, bias=False),
    )
    return model


def construct_linear_layer(num_inputs, num_outputs):
    """Linear layer without bias (just a matrix multiplication)

    Args:
        num_inputs (int): number of inputs
        num_outputs (int): number of outputs

    Returns:
        torch model: linear layer
    """
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_inputs, num_outputs, bias=False),
    )
    return model
