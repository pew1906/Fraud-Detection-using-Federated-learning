"""
Local Fraud Detection Neural Network
Pure NumPy implementation — no external DL framework dependency.
Architecture: Dense → BN → Dropout → Dense → BN → Dropout → Output
"""

import numpy as np
from typing import List, Optional, Tuple


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class LocalFraudModel:
    """
    Two-hidden-layer MLP for binary fraud classification.
    Supports FedProx proximal regularization during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.training = True

        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dims[0]) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dims[0])
        self.W2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * np.sqrt(2.0 / hidden_dims[0])
        self.b2 = np.zeros(hidden_dims[1])
        self.W3 = np.random.randn(hidden_dims[1], 1) * np.sqrt(2.0 / hidden_dims[1])
        self.b3 = np.zeros(1)

        # Batch Norm params
        self.bn1_gamma = np.ones(hidden_dims[0])
        self.bn1_beta = np.zeros(hidden_dims[0])
        self.bn2_gamma = np.ones(hidden_dims[1])
        self.bn2_beta = np.zeros(hidden_dims[1])

        # Running stats for BN inference
        self.bn1_running_mean = np.zeros(hidden_dims[0])
        self.bn1_running_var  = np.ones(hidden_dims[0])
        self.bn2_running_mean = np.zeros(hidden_dims[1])
        self.bn2_running_var  = np.ones(hidden_dims[1])

        # Adam optimizer state
        self._init_adam()

    # ── Forward ─────────────────────────────────────────────────────────────

    def _batch_norm(self, x, gamma, beta, running_mean, running_var, momentum=0.1, eps=1e-5):
        if self.training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            running_mean[:] = (1 - momentum) * running_mean + momentum * mu
            running_var[:]  = (1 - momentum) * running_var  + momentum * var
        else:
            mu, var = running_mean, running_var
        x_hat = (x - mu) / np.sqrt(var + eps)
        return gamma * x_hat + beta, mu, var, x_hat

    def _dropout(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.training or self.dropout_rate == 0:
            return x, np.ones_like(x)
        mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(float)
        return x * mask / (1 - self.dropout_rate), mask

    def forward(self, X: np.ndarray):
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1, mu1, var1, xh1 = self._batch_norm(
            z1, self.bn1_gamma, self.bn1_beta,
            self.bn1_running_mean, self.bn1_running_var
        )
        h1 = _relu(a1)
        h1d, mask1 = self._dropout(h1)

        # Layer 2
        z2 = h1d @ self.W2 + self.b2
        a2, mu2, var2, xh2 = self._batch_norm(
            z2, self.bn2_gamma, self.bn2_beta,
            self.bn2_running_mean, self.bn2_running_var
        )
        h2 = _relu(a2)
        h2d, mask2 = self._dropout(h2)

        # Output
        out = _sigmoid(h2d @ self.W3 + self.b3).squeeze(-1)

        cache = (X, z1, a1, h1, h1d, mask1, mu1, var1, xh1,
                 z2, a2, h2, h2d, mask2, mu2, var2, xh2)
        return out, cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.training = False
        proba, _ = self.forward(X)
        self.training = True
        return proba

    # ── Backward + Adam update ───────────────────────────────────────────────

    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.001,
        sample_weights: Optional[np.ndarray] = None,
        global_weights: Optional[List[np.ndarray]] = None,
        mu: float = 0.01,
    ) -> float:
        """One forward-backward pass with optional FedProx regularization."""
        n = len(y)
        out, cache = self.forward(X)
        (X_, z1, a1, h1, h1d, mask1, mu1, var1, xh1,
         z2, a2, h2, h2d, mask2, mu2, var2, xh2) = cache

        eps = 1e-7
        w = sample_weights if sample_weights is not None else np.ones(n)
        loss = -np.mean(w * (y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps)))

        # FedProx proximal term
        if global_weights is not None and mu > 0:
            local_w = self.get_weights()
            prox_loss = sum(np.sum((lw - gw) ** 2) for lw, gw in zip(local_w, global_weights))
            loss += (mu / 2) * prox_loss

        # Output gradient
        dout = (out - y) * w / n  # (n,)

        # Layer 3 grads
        dW3 = (h2d.T @ dout[:, None])
        db3 = dout.sum(keepdims=True)
        dh2d = (dout[:, None] @ self.W3.T)

        # Dropout 2
        dh2 = dh2d * mask2 / (1 - self.dropout_rate + 1e-8) if self.dropout_rate > 0 else dh2d

        # ReLU 2
        da2 = dh2 * _relu_grad(a2)

        # BN 2 (simplified)
        dz2 = self._bn_backward(da2, var2, xh2, self.bn2_gamma)
        dbn2_gamma = (da2 * xh2).sum(axis=0)
        dbn2_beta  = da2.sum(axis=0)

        dW2 = h1d.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1d = dz2 @ self.W2.T

        # Dropout 1
        dh1 = dh1d * mask1 / (1 - self.dropout_rate + 1e-8) if self.dropout_rate > 0 else dh1d

        # ReLU 1
        da1 = dh1 * _relu_grad(a1)

        # BN 1
        dz1 = self._bn_backward(da1, var1, xh1, self.bn1_gamma)
        dbn1_gamma = (da1 * xh1).sum(axis=0)
        dbn1_beta  = da1.sum(axis=0)

        dW1 = X_.T @ dz1
        db1 = dz1.sum(axis=0)

        # FedProx gradient addition
        if global_weights is not None and mu > 0:
            gW1, gb1, gW2, gb2, gW3, gb3 = global_weights[:6]
            dW1 += mu * (self.W1 - gW1)
            db1 += mu * (self.b1 - gb1)
            dW2 += mu * (self.W2 - gW2)
            db2 += mu * (self.b2 - gb2)
            dW3 += mu * (self.W3 - gW3)
            db3 += mu * (self.b3 - gb3)

        grads = [dW1, db1, dW2, db2, dW3, db3, dbn1_gamma, dbn1_beta, dbn2_gamma, dbn2_beta]
        self._adam_update(grads, lr)
        return float(loss)

    def _bn_backward(self, dout, var, x_hat, gamma, eps=1e-5):
        """Simplified batch norm backward."""
        N = dout.shape[0]
        std_inv = 1.0 / np.sqrt(var + eps)
        dx_hat = dout * gamma
        dvar = (-0.5 * (dx_hat * x_hat).sum(axis=0)) * (std_inv ** 3)
        dmean = (-dx_hat * std_inv).sum(axis=0) + dvar * (-2 * x_hat.mean(axis=0))
        dx = dx_hat * std_inv + dvar * 2 * x_hat / N + dmean / N
        return dx

    # ── Weight Management ────────────────────────────────────────────────────

    def get_weights(self) -> List[np.ndarray]:
        return [
            self.W1.copy(), self.b1.copy(),
            self.W2.copy(), self.b2.copy(),
            self.W3.copy(), self.b3.copy(),
            self.bn1_gamma.copy(), self.bn1_beta.copy(),
            self.bn2_gamma.copy(), self.bn2_beta.copy(),
        ]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
         self.bn1_gamma, self.bn1_beta, self.bn2_gamma, self.bn2_beta) = [
            w.copy() for w in weights
        ]

    # ── Adam Optimizer ───────────────────────────────────────────────────────

    def _init_adam(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_beta1 = beta1
        self._adam_beta2 = beta2
        self._adam_eps   = eps
        self._adam_t     = 0
        n_params = 10
        self._adam_m = [np.zeros_like(p) for p in self.get_weights()]
        self._adam_v = [np.zeros_like(p) for p in self.get_weights()]

    def _adam_update(self, grads: List[np.ndarray], lr: float):
        self._adam_t += 1
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                  self.bn1_gamma, self.bn1_beta, self.bn2_gamma, self.bn2_beta]
        for i, (p, g) in enumerate(zip(params, grads)):
            self._adam_m[i] = self._adam_beta1 * self._adam_m[i] + (1 - self._adam_beta1) * g
            self._adam_v[i] = self._adam_beta2 * self._adam_v[i] + (1 - self._adam_beta2) * g**2
            m_hat = self._adam_m[i] / (1 - self._adam_beta1 ** self._adam_t)
            v_hat = self._adam_v[i] / (1 - self._adam_beta2 ** self._adam_t)
            p -= lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)


