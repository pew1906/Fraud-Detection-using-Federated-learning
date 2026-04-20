"""
Non-IID Bank Transaction Data Generator
Simulates realistic fraud patterns across heterogeneous banking institutions.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class BankProfile:
    bank_id: str
    n_transactions: int
    fraud_rate: float        # e.g. 0.02 = 2% fraud
    avg_transaction: float   # average transaction amount
    std_transaction: float
    primary_channels: List[str]  # e.g. ['online', 'atm']
    geography: str               # e.g. 'urban', 'rural', 'international'


FEATURES = [
    "amount", "amount_log", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "channel_online", "channel_atm",
    "channel_pos", "channel_wire", "geo_urban", "geo_rural",
    "geo_international", "velocity_1h", "velocity_24h",
    "distance_from_home", "foreign_transaction", "new_merchant",
    "high_risk_country", "card_present",
]


class TransactionDataGenerator:
    """
    Generates synthetic, non-IID transaction datasets for multiple banks.
    Each bank has unique fraud patterns, amounts, and channel distributions.
    """

    PREDEFINED_BANKS = [
        BankProfile("NationalBank",   10000, 0.025, 500,  800,  ["pos", "atm"],     "urban"),
        BankProfile("DigitalFirst",    8000, 0.035, 300,  400,  ["online"],          "international"),
        BankProfile("RuralCoopBank",   5000, 0.010, 200,  300,  ["pos", "atm"],     "rural"),
        BankProfile("PremiumWealth",   3000, 0.015, 5000, 8000, ["wire", "online"],  "urban"),
        BankProfile("FinTechNeo",      7000, 0.045, 150,  200,  ["online"],          "international"),
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_all(
        self,
        bank_profiles: Optional[List[BankProfile]] = None,
        val_split: float = 0.2,
        test_split: float = 0.1,
    ) -> Dict[str, Dict]:
        """
        Generate datasets for all banks.
        Returns dict: bank_id → {'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'profile'}
        """
        profiles = bank_profiles or self.PREDEFINED_BANKS
        all_data = {}
        for profile in profiles:
            X, y = self._generate_bank_data(profile)
            X_scaled = self._scale_features(X)
            splits = self._split(X_scaled, y, val_split, test_split)
            all_data[profile.bank_id] = {**splits, "profile": profile}
        return all_data

    def generate_global_test(
        self,
        bank_profiles: Optional[List[BankProfile]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a unified test set combining samples from all banks."""
        profiles = bank_profiles or self.PREDEFINED_BANKS
        Xs, ys = [], []
        for p in profiles:
            np.random.seed(self.seed + 999)
            test_profile = BankProfile(
                p.bank_id, max(500, p.n_transactions // 5),
                p.fraud_rate, p.avg_transaction, p.std_transaction,
                p.primary_channels, p.geography
            )
            X, y = self._generate_bank_data(test_profile)
            Xs.append(self._scale_features(X))
            ys.append(y)
        X_all = np.vstack(Xs)
        y_all = np.concatenate(ys)
        idx = np.random.permutation(len(y_all))
        return X_all[idx], y_all[idx]

    # ── Private ──────────────────────────────────────────────────────────────

    def _generate_bank_data(self, profile: BankProfile) -> Tuple[np.ndarray, np.ndarray]:
        n = profile.n_transactions
        n_fraud = int(n * profile.fraud_rate)
        n_legit = n - n_fraud

        # Legitimate transactions
        legit = self._generate_transactions(n_legit, profile, is_fraud=False)
        # Fraudulent transactions
        fraud = self._generate_transactions(n_fraud, profile, is_fraud=True)

        X = np.vstack([legit, fraud])
        y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

        # Shuffle
        idx = np.random.permutation(n)
        return X[idx], y[idx]

    def _generate_transactions(
        self, n: int, profile: BankProfile, is_fraud: bool
    ) -> np.ndarray:
        if n == 0:
            return np.zeros((0, len(FEATURES)))

        fraud_mult = 3.0 if is_fraud else 1.0

        amount = np.abs(np.random.normal(
            profile.avg_transaction * fraud_mult,
            profile.std_transaction * fraud_mult,
            n
        ))
        amount_log = np.log1p(amount)

        hour = np.random.randint(0 if is_fraud else 6, 5 if is_fraud else 23, n)
        dow = np.random.randint(0, 7, n)
        is_weekend = (dow >= 5).astype(float)
        is_night = ((hour < 6) | (hour > 22)).astype(float)

        # Channel encoding
        channels = ["online", "atm", "pos", "wire"]
        ch_probs = {
            "online": 0.0, "atm": 0.0, "pos": 0.0, "wire": 0.0
        }
        for ch in profile.primary_channels:
            ch_probs[ch] = 1.0 / len(profile.primary_channels)
        probs = np.array([ch_probs[c] for c in channels])
        probs /= probs.sum()

        ch_idx = np.random.choice(len(channels), n, p=probs)
        ch_online = (ch_idx == 0).astype(float)
        ch_atm    = (ch_idx == 1).astype(float)
        ch_pos    = (ch_idx == 2).astype(float)
        ch_wire   = (ch_idx == 3).astype(float)

        # Geography
        geo_probs = {
            "urban": [0.8, 0.1, 0.1],
            "rural": [0.1, 0.8, 0.1],
            "international": [0.3, 0.1, 0.6],
        }
        gp = geo_probs.get(profile.geography, [0.33, 0.33, 0.34])
        geo_idx = np.random.choice(3, n, p=gp)
        geo_urban = (geo_idx == 0).astype(float)
        geo_rural = (geo_idx == 1).astype(float)
        geo_intl  = (geo_idx == 2).astype(float)

        # Risk features (frauds have higher risk scores)
        base_vel = 5.0 if is_fraud else 1.0
        velocity_1h  = np.abs(np.random.exponential(base_vel, n))
        velocity_24h = np.abs(np.random.exponential(base_vel * 3, n))

        distance = np.abs(np.random.exponential(200 if is_fraud else 20, n))
        foreign  = (np.random.rand(n) < (0.6 if is_fraud else 0.05)).astype(float)
        new_merch = (np.random.rand(n) < (0.7 if is_fraud else 0.1)).astype(float)
        high_risk = (np.random.rand(n) < (0.4 if is_fraud else 0.02)).astype(float)
        card_present = (np.random.rand(n) < (0.3 if is_fraud else 0.8)).astype(float)

        return np.column_stack([
            amount, amount_log, hour, dow, is_weekend, is_night,
            ch_online, ch_atm, ch_pos, ch_wire,
            geo_urban, geo_rural, geo_intl,
            velocity_1h, velocity_24h, distance, foreign,
            new_merch, high_risk, card_present,
        ])

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Min-max + log normalization for numerical stability."""
        X = X.copy().astype(float)
        for col in [0, 1, 13, 14, 15]:  # amount columns + velocities + distance
            col_max = X[:, col].max() + 1e-8
            X[:, col] = X[:, col] / col_max
        return X

    def _split(
        self, X: np.ndarray, y: np.ndarray, val_frac: float, test_frac: float
    ) -> Dict:
        n = len(y)
        n_test = max(50, int(n * test_frac))
        n_val  = max(50, int(n * val_frac))
        idx = np.random.permutation(n)
        test_idx = idx[:n_test]
        val_idx  = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        return {
            "X_train": X[train_idx], "y_train": y[train_idx],
            "X_val":   X[val_idx],   "y_val":   y[val_idx],
            "X_test":  X[test_idx],  "y_test":  y[test_idx],
        }
