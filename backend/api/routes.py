"""
REST API routes for FedFraud backend.
"""

import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from services.fl_service import TrainingConfig, experiment_state, run_training
from database.mongo import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Schemas ─────────────────────────────────────────────────────────────────

class StartTrainingRequest(BaseModel):
    num_rounds: int = Field(15, ge=1, le=50)
    local_epochs: int = Field(5, ge=1, le=20)
    learning_rate: float = Field(0.001, gt=0)
    batch_size: int = Field(64, ge=8, le=512)
    mu: float = Field(0.01, ge=0)
    strategy: str = Field("fedavg", pattern="^(fedavg|fedprox|fedadam)$")
    dp_noise_multiplier: float = Field(0.0, ge=0)
    dp_max_grad_norm: float = Field(1.0, gt=0)
    dropout_rate: float = Field(0.3, ge=0, lt=1)
    seed: int = Field(42)


# Active WebSocket broadcast function (set by main.py)
_broadcast_fn = None


def set_broadcast_fn(fn):
    global _broadcast_fn
    _broadcast_fn = fn


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/start-training")
async def start_training(req: StartTrainingRequest, background_tasks: BackgroundTasks):
    """Start federated learning training in the background."""
    if experiment_state.status == "running":
        raise HTTPException(status_code=409, detail="Training is already in progress.")

    config = TrainingConfig(
        num_rounds=req.num_rounds,
        local_epochs=req.local_epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        mu=req.mu,
        strategy=req.strategy,
        dp_noise_multiplier=req.dp_noise_multiplier,
        dp_max_grad_norm=req.dp_max_grad_norm,
        dropout_rate=req.dropout_rate,
        seed=req.seed,
    )

    async def _run():
        db = await get_db()
        await run_training(config, emit_update=_broadcast_fn, db=db)

    background_tasks.add_task(_run)

    return {"status": "started", "message": "Training started in background.", "config": req.dict()}


@router.get("/status")
async def get_status():
    """Return current training status and progress."""
    return experiment_state.to_dict()


@router.get("/metrics")
async def get_metrics():
    """Return latest round metrics."""
    if not experiment_state.history:
        return {"message": "No metrics yet.", "data": None}
    return {"data": experiment_state.history[-1]}


@router.get("/history")
async def get_history():
    """Return full round-by-round metrics history."""
    return {
        "rounds": len(experiment_state.history),
        "data": experiment_state.history,
        "baseline": experiment_state.baseline,
        "final": experiment_state.final,
    }


@router.get("/bank-profiles")
async def get_bank_profiles():
    """Return participating bank profiles."""
    return {"data": experiment_state.bank_profiles}


@router.post("/reset")
async def reset_experiment():
    """Reset experiment state (only when not running)."""
    if experiment_state.status == "running":
        raise HTTPException(status_code=409, detail="Cannot reset while training is running.")
    experiment_state.reset()
    return {"status": "reset", "message": "Experiment state cleared."}
