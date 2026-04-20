"""
Federated Learning Service Layer
Orchestrates training loop, emits real-time updates via WebSocket callback.
"""

import numpy as np
import logging
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Awaitable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
)

from services.fl_server import FederatedServer, AggregationStrategy
from services.fl_client import BankClient, ClientConfig
from services.generator import TransactionDataGenerator, FEATURES
from models.local_model import LocalFraudModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    num_rounds: int = 15
    local_epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 64
    mu: float = 0.01
    strategy: str = "fedavg"
    dp_noise_multiplier: float = 0.0
    dp_max_grad_norm: float = 1.0
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


# In-memory experiment state
class ExperimentState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.status: str = "idle"          # idle | running | completed | error
        self.current_round: int = 0
        self.total_rounds: int = 0
        self.history: List[Dict] = []
        self.config: Optional[Dict] = None
        self.error: Optional[str] = None
        self.baseline: Optional[Dict] = None
        self.final: Optional[Dict] = None
        self.bank_profiles: Optional[Dict] = None

    def to_dict(self):
        return {
            "status": self.status,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "progress_pct": round(self.current_round / max(self.total_rounds, 1) * 100, 1),
            "config": self.config,
            "error": self.error,
            "baseline": self.baseline,
            "final": self.final,
            "bank_profiles": self.bank_profiles,
        }


# Global singleton state
experiment_state = ExperimentState()


async def run_training(
    config: TrainingConfig,
    emit_update: Optional[Callable[[Dict], Awaitable[None]]] = None,
    db=None,
) -> Dict:
    """
    Run federated learning training loop.
    Calls emit_update(payload) after each round for real-time streaming.
    Optionally persists round metrics to MongoDB if db is provided.
    """
    global experiment_state
    experiment_state.reset()
    experiment_state.status = "running"
    experiment_state.total_rounds = config.num_rounds
    experiment_state.config = {
        "strategy": config.strategy,
        "num_rounds": config.num_rounds,
        "local_epochs": config.local_epochs,
        "dp_noise_multiplier": config.dp_noise_multiplier,
        "mu": config.mu,
    }

    try:
        strategy_enum = AggregationStrategy(config.strategy)
        np.random.seed(config.seed)

        # Generate data
        gen = TransactionDataGenerator(seed=config.seed)
        bank_data = gen.generate_all()
        X_global_test, y_global_test = gen.generate_global_test()
        bank_ids = list(bank_data.keys())
        input_dim = len(FEATURES)

        # Build bank profiles summary
        experiment_state.bank_profiles = {
            bid: {
                "n_transactions": d["profile"].n_transactions,
                "fraud_rate": d["profile"].fraud_rate,
                "geography": d["profile"].geography,
                "train_size": len(d["y_train"]),
                "fraud_in_train": int(d["y_train"].sum()),
            }
            for bid, d in bank_data.items()
        }

        # Initialize clients
        clients: Dict[str, BankClient] = {}
        for bid in bank_ids:
            model = LocalFraudModel(
                input_dim=input_dim,
                hidden_dims=config.hidden_dims,
                dropout_rate=config.dropout_rate,
                seed=config.seed,
            )
            cfg = ClientConfig(
                bank_id=bid,
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                mu=config.mu,
                seed=config.seed,
            )
            clients[bid] = BankClient(cfg, model)

        # Initialize server
        server = FederatedServer(
            strategy=strategy_enum,
            min_clients=2,
            dp_noise_multiplier=config.dp_noise_multiplier,
            dp_max_grad_norm=config.dp_max_grad_norm,
        )
        init_weights = clients[bank_ids[0]].get_weights()
        server.initialize_global_model(init_weights)

        # Baseline
        baseline = _evaluate_global(server.global_weights, X_global_test, y_global_test, config)
        experiment_state.baseline = baseline
        logger.info(f"Baseline → F1={baseline['f1']:.4f} | AUC={baseline['auc']:.4f}")

        # Training rounds
        for rnd in range(1, config.num_rounds + 1):
            experiment_state.current_round = rnd

            client_updates = []
            for bid in bank_ids:
                client = clients[bid]
                client.set_global_weights(server.global_weights)
                d = bank_data[bid]
                update = client.train(
                    d["X_train"], d["y_train"],
                    d["X_val"], d["y_val"],
                    global_weights=server.global_weights,
                )
                client_updates.append(update)

            _, round_metrics = server.aggregate(client_updates, rnd)

            global_eval = _evaluate_global(server.global_weights, X_global_test, y_global_test, config)

            round_data = {
                "round": rnd,
                "loss": round(global_eval["loss"], 4),
                "accuracy": round(global_eval["accuracy"], 4),
                "precision": round(global_eval["precision"], 4),
                "recall": round(global_eval["recall"], 4),
                "f1": round(global_eval["f1"], 4),
                "auc": round(global_eval["auc"], 4),
                "num_clients": len(client_updates),
                "client_metrics": [
                    {"bank_id": u["bank_id"], **{k: round(v, 4) for k, v in u["metrics"].items()}}
                    for u in client_updates
                ],
            }

            experiment_state.history.append(round_data)

            # Persist to MongoDB if available
            if db is not None:
                try:
                    await db.rounds.insert_one({**round_data, "config": experiment_state.config})
                except Exception as e:
                    logger.warning(f"MongoDB write failed: {e}")

            # Emit real-time update
            if emit_update:
                await emit_update({
                    "type": "round_update",
                    "data": round_data,
                    "state": experiment_state.to_dict(),
                })

            logger.info(f"Round {rnd:02d} | F1={global_eval['f1']:.4f} | AUC={global_eval['auc']:.4f}")

            # Yield control so WebSocket messages can be sent
            await asyncio.sleep(0)

        # Final evaluation
        final = _evaluate_global(server.global_weights, X_global_test, y_global_test, config)
        experiment_state.final = final
        experiment_state.status = "completed"

        if emit_update:
            await emit_update({
                "type": "training_complete",
                "data": {"final": final, "baseline": baseline},
                "state": experiment_state.to_dict(),
            })

        return experiment_state.to_dict()

    except Exception as e:
        experiment_state.status = "error"
        experiment_state.error = str(e)
        logger.error(f"Training error: {e}", exc_info=True)
        if emit_update:
            await emit_update({"type": "error", "message": str(e)})
        raise


def _evaluate_global(weights, X, y, config: TrainingConfig) -> Dict:
    model = LocalFraudModel(len(FEATURES), config.hidden_dims, config.dropout_rate)
    model.set_weights(weights)
    proba = model.predict_proba(X)
    preds = (proba >= 0.5).astype(int)
    return {
        "loss": float(log_loss(y, proba)),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc": float(roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5),
    }
