"""
Federated Learning Server
Supports FedAvg, FedProx, and FedAdam aggregation strategies.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAM = "fedadam"


@dataclass
class RoundMetrics:
    round_num: int
    global_loss: float
    global_accuracy: float
    global_precision: float
    global_recall: float
    global_f1: float
    global_auc: float
    num_clients: int
    client_metrics: List[Dict] = field(default_factory=list)


class FederatedServer:
    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        min_clients: int = 2,
        dp_noise_multiplier: float = 0.0,
        dp_max_grad_norm: float = 1.0,
        fedadam_beta1: float = 0.9,
        fedadam_beta2: float = 0.999,
        fedadam_lr: float = 0.01,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.min_clients = min_clients
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_max_grad_norm = dp_max_grad_norm
        self.fedadam_beta1 = fedadam_beta1
        self.fedadam_beta2 = fedadam_beta2
        self.fedadam_lr = fedadam_lr
        self.seed = seed

        self.global_weights: Optional[List[np.ndarray]] = None
        self.round_history: List[RoundMetrics] = []
        self._m: Optional[List[np.ndarray]] = None
        self._v: Optional[List[np.ndarray]] = None
        self._adam_t: int = 0

        np.random.seed(seed)

    def initialize_global_model(self, weights: List[np.ndarray]) -> None:
        self.global_weights = [w.copy() for w in weights]
        if self.strategy == AggregationStrategy.FEDADAM:
            self._m = [np.zeros_like(w) for w in weights]
            self._v = [np.zeros_like(w) for w in weights]

    def aggregate(self, client_updates: List[Dict], round_num: int) -> Tuple[List[np.ndarray], RoundMetrics]:
        if len(client_updates) < self.min_clients:
            raise ValueError(f"Need at least {self.min_clients} clients, got {len(client_updates)}")

        if self.dp_noise_multiplier > 0:
            client_updates = self._apply_dp_noise(client_updates)

        if self.strategy == AggregationStrategy.FEDADAM:
            new_weights = self._fedadam(client_updates)
        else:
            new_weights = self._fedavg(client_updates)

        self.global_weights = new_weights
        total_samples = sum(u["num_samples"] for u in client_updates)
        metrics = self._aggregate_metrics(client_updates, total_samples)
        client_metric_list = [{"bank_id": u["bank_id"], **u["metrics"]} for u in client_updates]

        round_metrics = RoundMetrics(
            round_num=round_num,
            global_loss=metrics["loss"],
            global_accuracy=metrics["accuracy"],
            global_precision=metrics["precision"],
            global_recall=metrics["recall"],
            global_f1=metrics["f1"],
            global_auc=metrics["auc"],
            num_clients=len(client_updates),
            client_metrics=client_metric_list,
        )
        self.round_history.append(round_metrics)
        return new_weights, round_metrics

    def _fedavg(self, client_updates):
        total_samples = sum(u["num_samples"] for u in client_updates)
        new_weights = [np.zeros_like(w) for w in client_updates[0]["weights"]]
        for update in client_updates:
            wf = update["num_samples"] / total_samples
            for i, lw in enumerate(update["weights"]):
                new_weights[i] += wf * lw
        return new_weights

    def _fedadam(self, client_updates):
        self._adam_t += 1
        avg_weights = self._fedavg(client_updates)
        new_weights = []
        for i, (gw, aw) in enumerate(zip(self.global_weights, avg_weights)):
            pg = aw - gw
            self._m[i] = self.fedadam_beta1 * self._m[i] + (1 - self.fedadam_beta1) * pg
            self._v[i] = self.fedadam_beta2 * self._v[i] + (1 - self.fedadam_beta2) * pg**2
            mh = self._m[i] / (1 - self.fedadam_beta1**self._adam_t)
            vh = self._v[i] / (1 - self.fedadam_beta2**self._adam_t)
            new_weights.append(gw + self.fedadam_lr * mh / (np.sqrt(vh) + 1e-8))
        return new_weights

    def _apply_dp_noise(self, client_updates):
        noisy = []
        for update in client_updates:
            nw = []
            for lw in update["weights"]:
                norm = np.linalg.norm(lw)
                clipped = lw * min(1, self.dp_max_grad_norm / (norm + 1e-8))
                noise = np.random.normal(0, self.dp_noise_multiplier * self.dp_max_grad_norm, clipped.shape)
                nw.append(clipped + noise)
            noisy.append({**update, "weights": nw})
        return noisy

    def _aggregate_metrics(self, client_updates, total_samples):
        keys = ["loss", "accuracy", "precision", "recall", "f1", "auc"]
        agg = {k: 0.0 for k in keys}
        for u in client_updates:
            w = u["num_samples"] / total_samples
            for k in keys:
                agg[k] += w * u["metrics"].get(k, 0.0)
        return agg
