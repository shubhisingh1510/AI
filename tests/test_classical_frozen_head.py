from classical_frozen_head import (
    find_matched_hidden_dim,
    mlp_param_count,
    quantum_circuit_param_count,
)


def test_quantum_circuit_param_count_matches_default_config():
    # 6 qubits, depth 3, 4 classes -- the project's documented default (82 trainable params).
    assert quantum_circuit_param_count(num_qubits=6, circuit_depth=3, n_classes=4) == 82


def test_find_matched_hidden_dim_gets_close_to_target():
    target = quantum_circuit_param_count(num_qubits=6, circuit_depth=3, n_classes=4)
    h = find_matched_hidden_dim(in_dim=6, n_classes=4, target_params=target)
    achieved = mlp_param_count(in_dim=6, hidden_dim=h, n_classes=4)
    assert abs(achieved - target) <= 5


def test_find_matched_hidden_dim_is_the_argmin_over_the_search_range():
    target = 82
    h = find_matched_hidden_dim(in_dim=6, n_classes=4, target_params=target, max_hidden=32)
    best_diff = abs(mlp_param_count(6, h, 4) - target)
    for other_h in range(1, 33):
        assert abs(mlp_param_count(6, other_h, 4) - target) >= best_diff
