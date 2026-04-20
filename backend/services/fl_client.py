"""
Federated Learning Bank Client
Handles local model training with FedProx regularization and early stopping.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    bank_id: str
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    mu: float = 0.01          # FedProx proximal term coefficient
    early_stopping_patience: int = 3
    class_weight_auto: bool = True  # handle class imbalance automatically
    dropout_rate: float = 0.3
    seed: int = 42


class BankClient:
    """
    Represents a single bank participant in the FL system.
    Trains a local neural network on private transaction data.
    """

    def __init__(self, config: ClientConfig, model: "LocalFraudModel"):
        self.config = config
        self.model = model
        self.training_history: List[Dict] = []
        np.random.seed(config.seed)
        logger.info(f"Client [{config.bank_id}] initialized")

    def set_global_weights(self, global_weights: List[np.ndarray]) -> None:
        """Download global model weights from server."""
        self.model.set_weights(global_weights)

    def get_weights(self) -> List[np.ndarray]:
        """Return current local model weights."""
        return self.model.get_weights()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        Perform local training with optional FedProx regularization.
        Returns training metrics and updated weights.
        """
        # Compute class weights for imbalanced fraud data
        class_weights = None
        if self.config.class_weight_auto:
            class_weights = self._compute_class_weights(y_train)

        # Store global weights reference for FedProx
        w0 = [w.copy() for w in global_weights] if global_weights else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = self.model.get_weights()

        for epoch in range(self.config.local_epochs):
            # Mini-batch SGD
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(X_train), self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                sample_weights = (
                    class_weights[y_batch.astype(int)] if class_weights is not None else None
                )

                loss = self.model.train_step(
                    X_batch, y_batch,
                    lr=self.config.learning_rate,
                    sample_weights=sample_weights,
                    global_weights=w0,
                    mu=self.config.mu,
                )
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            # Early stopping on validation set
            if X_val is not None and y_val is not None:
                val_preds = self.model.predict_proba(X_val)
                val_loss = log_loss(y_val, val_preds)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_weights = self.model.get_weights()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.debug(
                            f"[{self.config.bank_id}] Early stop at epoch {epoch+1}"
                        )
                        self.model.set_weights(best_weights)
                        break

        # Compute final metrics on validation set
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train
        metrics = self._evaluate(eval_X, eval_y)

        self.training_history.append({
            "num_samples": len(X_train),
            "metrics": metrics,
        })

        logger.info(
            f"[{self.config.bank_id}] Training done | "
            f"F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f} | n={len(X_train)}"
        )

        return {
            "bank_id": self.config.bank_id,
            "weights": self.model.get_weights(),
            "num_samples": len(X_train),
            "metrics": metrics,
        }

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        proba = self.model.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        return {
            "loss": float(log_loss(y, proba)),
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "auc": float(roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5),
        }

    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Inverse-frequency class weighting for imbalanced datasets."""
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weights.get(cls, 1.0) for cls in range(int(y.max()) + 1)])
