import numpy as np

from learning.federated_sim import (
    TinyLogReg,
    federated_average,
    make_new_site_eval_data,
    make_site_train_data,
    run_federated_simulation,
)


def test_federated_average_is_weighted_mean():
    params_a = [np.array([1.0, 2.0]), np.array([0.0])]
    params_b = [np.array([3.0, 4.0]), np.array([2.0])]
    result = federated_average([params_a, params_b], [10, 10])
    assert np.allclose(result[0], [2.0, 3.0])
    assert np.allclose(result[1], [1.0])

    result_weighted = federated_average([params_a, params_b], [90, 10])
    assert np.allclose(result_weighted[0], [1.2, 2.2])


def test_tiny_logreg_learns_a_separable_boundary():
    X = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    X = np.hstack([X, np.zeros((4, 3))])
    y = np.array([1.0, 1.0, 0.0, 0.0])

    model = TinyLogReg()
    for _ in range(50):
        model.fit_one_round(X, y, lr=0.5, epochs=1)

    assert model.accuracy(X, y) == 1.0


def test_site_data_shapes():
    X_train, y_train = make_site_train_data(site_seed=0, n_train=24)
    assert X_train.shape == (24, 4)
    assert y_train.shape == (24,)
    assert set(np.unique(y_train)) <= {0.0, 1.0}


def test_federated_simulation_runs_and_improves_over_rounds():
    result = run_federated_simulation(num_sites=4, num_rounds=8, n_train=24, new_site_seed=999)
    accs = result["federated_accuracy_by_round_on_new_site"]
    assert len(accs) == 8
    assert accs[-1] >= accs[0]


def test_federation_beats_solo_on_average_across_new_sites():
    """Checks that, averaged across several unseen new sites, the federated
    model out-generalizes the mean of individual sites' solo models.
    """
    fed_scores = []
    solo_scores = []
    for new_seed in range(999, 1004):
        r = run_federated_simulation(num_sites=4, num_rounds=8, n_train=24, new_site_seed=new_seed)
        fed_scores.append(r["final_federated_accuracy_on_new_site"])
        solo_scores.append(r["mean_solo_accuracy_on_new_site"])

    assert sum(fed_scores) / len(fed_scores) > sum(solo_scores) / len(solo_scores)
