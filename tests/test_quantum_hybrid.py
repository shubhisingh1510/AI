import numpy as np
import torch

from quantum_hybrid import HybridHead, build_quantum_layer


def test_quantum_layer_output_dim_matches_num_qubits():
    layer = build_quantum_layer(num_qubits=4, circuit_depth=1, diff_method="adjoint")
    x = torch.randn(3, 4)
    out = layer(x)
    assert out.shape == (3, 4)


def test_hybrid_head_forward_shape_matches_n_classes():
    head = HybridHead(num_qubits=4, circuit_depth=1, n_classes=3, diff_method="adjoint")
    x = torch.randn(2, 4)
    out = head(x)
    assert out.shape == (2, 3)


def test_hybrid_head_is_trainable_end_to_end():
    head = HybridHead(num_qubits=4, circuit_depth=1, n_classes=2, diff_method="adjoint")
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 0, 1])
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for _ in range(5):
        opt.zero_grad()
        out = head(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # not asserting convergence (5 steps on random data is noisy) -- just that gradients
    # actually flow through the quantum layer and the loss isn't frozen/NaN.
    assert all(np.isfinite(l) for l in losses)
    assert losses[0] != losses[-1]
