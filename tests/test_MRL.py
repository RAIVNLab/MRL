import torch

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
    print(loss)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_relative_importance():
    # shape: [G]
    relative_importance = [0.1, 0.9]

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


def test_idempotency():
	"""Tests losses of newer implementations are equal to the original for-loop
	implementation.
	"""
	def forward_loop(self, output, target):
		"""Original implementation of forward() using for-loop
		"""
		loss=0
		N = len(output)
		for i in range(N):
			rel = 1.0 if self.relative_importance is None else self.relative_importance[i] 
			loss += rel*self.criterion(output[i], target)
		return loss

	relative_importance = [0.1, 0.9]

	# Current implementation
	torch.manual_seed(0)
	loss_fn = Matryoshka_CE_Loss(relative_importance=relative_importance)
	output_bc = torch.randn(2, 3, 5, requires_grad=True)
	# shape: [N batch size]
	target_bc = torch.empty(3, dtype=torch.long).random_(5)
	loss_broadcast = loss_fn(output_bc, target_bc)

	# Monkeypatching Original for-loop implementation
	torch.manual_seed(0)
	Matryoshka_CE_Loss.forward = forward_loop
	loss_org = Matryoshka_CE_Loss(relative_importance=relative_importance)
	output_org = torch.randn(2, 3, 5, requires_grad=True)
	# shape: [N batch size]
	target_org = torch.empty(3, dtype=torch.long).random_(5)
	loss_loop = loss_org(output_org, target_org)

	# Ensure the inputs to the loss fn are equal
	assert torch.equal(output_bc, output_org)
	assert torch.equal(target_bc, target_org)

	# Ensure the outputs are mostly equal
	assert torch.allclose(loss_loop, loss_broadcast)
