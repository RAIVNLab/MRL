import torch
import pytest
from MRL import Matryoshka_CE_Loss

def test_forward():
    # Create a Matryoshka_CE_Loss instance
    loss_fn = Matryoshka_CE_Loss()

    # Create some dummy input and target tensors
    # shape: [G, N batch size, C number of classes]
    output = torch.randn(2, 3, 5, requires_grad=True)
    # shape: [N batch size]
    target = torch.empty(3, dtype=torch.long).random_(5)

    # Calculate the loss
    loss = loss_fn.forward(output, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_relative_importance():
    # shape: [G]
    relative_importance = torch.empty(2, dtype=torch.long).random_(5)

    # Create a Matryoshka_CE_Loss instance with relative_importance
    loss_fn = Matryoshka_CE_Loss(relative_importance=relative_importance)

    # Create some dummy input and target tensors
    # shape: [G, N batch size, C number of classes]
    output = torch.randn(2, 3, 5, requires_grad=True)
    # shape: [N batch size]
    target = torch.empty(3, dtype=torch.long).random_(5)
    loss = loss_fn.forward(output, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
