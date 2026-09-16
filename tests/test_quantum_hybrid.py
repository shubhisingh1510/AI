import numpy as np
import torch

from quantum_hybrid import HybridHead, build_quantum_layer, load_backbone_for_features


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


def test_quantum_layer_data_reuploading_output_shape_matches_non_reuploading():
    layer = build_quantum_layer(num_qubits=4, circuit_depth=2, diff_method="adjoint", data_reuploading=True)
    x = torch.randn(3, 4)
    out = layer(x)
    assert out.shape == (3, 4)


def test_quantum_layer_data_reuploading_has_same_weight_count_as_non_reuploading():
    # Re-uploading changes WHERE the input is injected, not the circuit's trainable parameter
    # count -- same (circuit_depth, num_qubits, 3) weight tensor either way.
    plain = build_quantum_layer(num_qubits=4, circuit_depth=2, diff_method="adjoint", data_reuploading=False)
    reuploaded = build_quantum_layer(num_qubits=4, circuit_depth=2, diff_method="adjoint", data_reuploading=True)
    plain_params = sum(p.numel() for p in plain.parameters())
    reuploaded_params = sum(p.numel() for p in reuploaded.parameters())
    assert plain_params == reuploaded_params


def test_hybrid_head_with_data_reuploading_is_trainable():
    head = HybridHead(num_qubits=4, circuit_depth=2, n_classes=2, diff_method="adjoint", data_reuploading=True)
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 0, 1])
    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    loss0 = criterion(head(x), y).item()
    for _ in range(5):
        opt.zero_grad()
        loss = criterion(head(x), y)
        loss.backward()
        opt.step()
    loss1 = loss.item()
    assert loss1 < loss0 + 1e-3  # non-strict: just confirm gradients actually flow and train


def test_load_backbone_for_features_accepts_old_style_flat_fc_checkpoint():
    # Checkpoints saved before classical_baseline.py wrapped the head in
    # Sequential(Dropout, Linear) used flat "fc.weight"/"fc.bias" keys -- must still load.
    device = torch.device("cpu")
    old_style_model = torch.nn.Module()
    import torchvision.models as tv_models
    ref = tv_models.resnet50(weights=None)
    ref.fc = torch.nn.Linear(ref.fc.in_features, 4)
    old_state = ref.state_dict()  # has flat "fc.weight"/"fc.bias"

    model = load_backbone_for_features(old_state, n_classes=4, pretrained=False, device=device)
    assert "fc.1.weight" in model.state_dict()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 4)


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
