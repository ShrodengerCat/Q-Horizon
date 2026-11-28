"""Variational quantum classifier (VQC) used in the Q//Horizon 2026 project.

Both notebooks (Quantum.ipynb and Ablation.ipynb) import everything from here, so
the main experiment and the ablations run exactly the same code.

Model, in one picture (4 qubits, one per input feature):

    |0> --RY(x0)--[ trainable layers ]--<Z>   <- the only qubit we measure
    |0> --RY(x1)--[ trainable layers ]--
    |0> --RY(x2)--[ trainable layers ]--
    |0> --RY(x3)--[ trainable layers ]--

The 4 features come from Classical.ipynb already scaled to [0, pi], so each one is
used directly as a rotation angle. The simulator is exact (no shot noise), so a
given seed always gives the same result.
"""
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 4  # one qubit per input feature


def load_split(seed, splits_dir="splits"):
    """Load the exact train/test split that Classical.ipynb saved for this seed."""
    d = np.load(Path(splits_dir) / f"seed{seed}.npz")
    return d["Xtr"], d["Xte"], d["ytr"], d["yte"]


def make_circuit(n_layers=3, entangle=True, n_qubits=N_QUBITS):
    """Return a QNode f(x, weights) -> <Z> of qubit 0, a number in [-1, 1].

    x        : features, shape (n_qubits,) or a batch (n_samples, n_qubits)
    weights  : trainable angles, shape (n_layers, n_qubits, 3)
    entangle : True  -> StronglyEntanglingLayers (a rotation on every qubit, then a
                        ring of CNOTs, repeated n_layers times)
               False -> the same rotations but no CNOTs (ablation: qubits never talk)
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, diff_method="backprop")
    def circuit(x, weights):
        # 1) data encoding: feature i becomes an RY rotation angle on qubit i
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        # 2) trainable part
        if entangle:
            qml.StronglyEntanglingLayers(weights, wires=wires)
        else:
            for layer in range(n_layers):
                for w in wires:
                    qml.Rot(weights[layer, w, 0], weights[layer, w, 1],
                            weights[layer, w, 2], wires=w)
        # 3) read-out: expectation of Pauli-Z on qubit 0
        return qml.expval(qml.PauliZ(0))

    return circuit


def predict(circuit, weights, X):
    """Class 1 if <Z> > 0, else class 0."""
    return (np.asarray(circuit(X, weights)) > 0).astype(int)


def accuracy(circuit, weights, X, y):
    return float(np.mean(predict(circuit, weights, X) == np.asarray(y)))


def fit_vqc(Xtr, ytr, n_layers=3, entangle=True, steps=100, lr=0.1,
            init_scale=0.5, seed=0):
    """Train the VQC with full-batch Adam on the square loss.

    Labels {0,1} are mapped to {-1,+1} so they match the range of <Z>.
    Returns (circuit, trained weights, loss per step, fit time in seconds).
    """
    circuit = make_circuit(n_layers, entangle)
    rng = np.random.default_rng(seed)
    weights = pnp.array(init_scale * rng.standard_normal((n_layers, N_QUBITS, 3)),
                        requires_grad=True)
    y_pm = 2.0 * np.asarray(ytr) - 1.0

    def cost(w):
        return pnp.mean((y_pm - circuit(Xtr, w)) ** 2)

    opt = qml.AdamOptimizer(stepsize=lr)
    losses = []
    t0 = time.time()
    for _ in range(steps):
        weights, loss = opt.step_and_cost(cost, weights)
        losses.append(float(loss))
    return circuit, weights, losses, time.time() - t0


def run_seed(seed, n_layers=3, entangle=True, steps=100, lr=0.1, init_scale=0.5,
             splits_dir="splits"):
    """Train + evaluate on one saved split.

    The result dict has the same keys as the entries in results/classical.json,
    so the two files can be compared directly.
    """
    Xtr, Xte, ytr, yte = load_split(seed, splits_dir)
    circuit, weights, losses, fit_time = fit_vqc(
        Xtr, ytr, n_layers, entangle, steps, lr, init_scale, seed)
    result = {
        "seed": seed,
        "train_acc": accuracy(circuit, weights, Xtr, ytr),
        "test_acc": accuracy(circuit, weights, Xte, yte),
        "fit_time_s": float(fit_time),
        "n_params": int(np.asarray(weights).size),
    }
    return result, losses
