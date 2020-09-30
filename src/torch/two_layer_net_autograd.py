# -*-coding: utf-8 -*-

import torch

dtype = torch.float
device = torch.device("cpu")

# N is batch size
# D_in is input dimension
# H is hidden dimension
# D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute and predicted y using operations on Tensors;
    # these are exactly the same operations we used to compute the forward pass using
    # Tensors , but we dont need to keep references to intermediate values since we
    # are not implementing the backward pass by hand
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print the loss using operations on Tensors.
    # Now loss is a tensor of shape(1,)
    # loss.item() gets the scalar value held in the loss
    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
