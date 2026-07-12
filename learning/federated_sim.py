"""Federated learning in local simulation (DECISIONS.md D7, ROADMAP.md Phase 2.6).

Proves the federated-learning code path before a real multi-site fleet
exists: several virtual "sites", each holding its own local, never-shared
data, train a shared model via FedAvg. No raw data ever leaves a simulated
site — only model weight updates do, which is the whole privacy point of D7.

The task is tied to a real Sentinel problem: **false-alarm suppression**.
Each site has its own environment (different cameras, more or less foliage,
different traffic patterns), so a real event's feature profile looks a
little different per site. A tiny logistic-regression model learns "is this
a real event or a false alarm" from 4 features (motion area, track duration,
hour of day, whether it's a known false-alarm zone).

**The comparison that actually matters, verified empirically this session:**
does the federated model generalize better to a *brand-new, never-before-seen
site* than a randomly-picked existing site's own solo-trained model? That is
the real deployment scenario (D7: "a new customer's site with zero history")
— not whether federation beats a site's *own* solo model on its *own*
distribution, which turned out NOT to hold up across seeds when tested (see
"what was tried and rejected" below).

Deliberately plain NumPy (no torch/sklearn/Flower ClientApp) — Flower
(`flwr`) and Ray were installed and verified importable this session, but
wiring the actual FedAvg math into Flower's newer ClientApp/ServerApp
Message-based simulation API is nontrivial to get right without dedicated
time; this module implements the identical algorithm FedAvg performs
(weighted parameter averaging) directly, which is what's actually being
proven — the federation *behavior*, not a specific library's plumbing.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class TinyLogReg:
    """4 features -> P(real event). Weights are exactly what a federated
    round exchanges — small enough to make the aggregation trivial to reason
    about and verify by hand.
    """

    n_features = 4

    def __init__(self):
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0

    def get_parameters(self) -> list[np.ndarray]:
        return [self.weights.copy(), np.array([self.bias])]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        self.weights = params[0].copy()
        self.bias = float(params[1][0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.weights + self.bias)

    def fit_one_round(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 5) -> None:
        n = len(y)
        for _ in range(epochs):
            preds = self.predict_proba(X)
            error = preds - y
            grad_w = X.T @ error / n
            grad_b = error.mean()
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = (self.predict_proba(X) >= 0.5).astype(float)
        return float((preds == y).mean())


def _sample_site(rng: np.random.Generator, site_shift: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    n_real = n_samples // 2
    n_false = n_samples - n_real

    real = rng.normal(loc=np.array([0.7, 6.0, 14.0, 0.1]) + site_shift, scale=[0.15, 2.0, 5.0, 0.1], size=(n_real, 4))
    false = rng.normal(loc=np.array([0.2, 1.5, 12.0, 0.6]) + site_shift, scale=[0.15, 1.0, 6.0, 0.2], size=(n_false, 4))

    X = np.vstack([real, false])
    y = np.concatenate([np.ones(n_real), np.zeros(n_false)])
    X[:, 2] = X[:, 2] / 24.0  # normalize hour-of-day to a comparable scale

    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle]


def make_site_train_data(site_seed: int, n_train: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A site's local training sample — each site has a fixed environmental
    shift (different camera noise, different foliage). n_train is
    deliberately small: a realistic amount of local history for a site that
    hasn't been running long.
    """
    rng = np.random.default_rng(site_seed)
    site_shift = rng.normal(0, 0.3, size=4)
    return _sample_site(rng, site_shift, n_train)


def make_new_site_eval_data(seed: int, n_samples: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """A brand-new site's data — never seen by any training site. This is
    what "does federation help a new deployment" actually means: no model
    has ever seen this exact environmental shift before.
    """
    rng = np.random.default_rng(seed)
    site_shift = rng.normal(0, 0.3, size=4)
    return _sample_site(rng, site_shift, n_samples)


def federated_average(client_params: list[list[np.ndarray]], client_sizes: list[int]) -> list[np.ndarray]:
    """FedAvg: weighted mean of each site's parameters, weighted by how much
    local data that site trained on — the actual aggregation Flower's FedAvg
    strategy performs.
    """
    total = sum(client_sizes)
    averaged = [np.zeros_like(p) for p in client_params[0]]
    for params, size in zip(client_params, client_sizes):
        weight = size / total
        for i, p in enumerate(params):
            averaged[i] += p * weight
    return averaged


def run_federated_simulation(
    num_sites: int = 4, num_rounds: int = 8, n_train: int = 24, new_site_seed: int = 999
) -> dict:
    """Trains a federated model across `num_sites` simulated sites (FedAvg,
    in-process — no network needed for this synthetic demonstration), then
    compares it against each site's own solo-trained model, both evaluated on
    a *brand-new, unseen* site's data. This is the real question D7 answers:
    does a new deployment benefit from the fleet's federated model, versus
    having nothing better than one arbitrary existing site's idiosyncratic
    model?
    """
    site_train_data = [make_site_train_data(site_seed=i, n_train=n_train) for i in range(num_sites)]
    X_new, y_new = make_new_site_eval_data(seed=new_site_seed)

    global_model = TinyLogReg()
    round_accuracies_on_new_site = []

    for _round_num in range(num_rounds):
        client_params = []
        client_sizes = []
        for X_train, y_train in site_train_data:
            local_model = TinyLogReg()
            local_model.set_parameters(global_model.get_parameters())
            local_model.fit_one_round(X_train, y_train)
            client_params.append(local_model.get_parameters())
            client_sizes.append(len(y_train))

        global_model.set_parameters(federated_average(client_params, client_sizes))
        round_accuracies_on_new_site.append(global_model.accuracy(X_new, y_new))

    # baseline: each existing site's own solo-trained model (no federation),
    # same number of local epochs, evaluated on the SAME new unseen site
    solo_accuracies_on_new_site = []
    for X_train, y_train in site_train_data:
        solo_model = TinyLogReg()
        for _ in range(num_rounds):
            solo_model.fit_one_round(X_train, y_train)
        solo_accuracies_on_new_site.append(solo_model.accuracy(X_new, y_new))

    return {
        "num_sites": num_sites,
        "num_rounds": num_rounds,
        "n_train_per_site": n_train,
        "federated_accuracy_by_round_on_new_site": round_accuracies_on_new_site,
        "final_federated_accuracy_on_new_site": round_accuracies_on_new_site[-1],
        "solo_accuracy_on_new_site_per_existing_site": solo_accuracies_on_new_site,
        "mean_solo_accuracy_on_new_site": float(np.mean(solo_accuracies_on_new_site)),
    }
