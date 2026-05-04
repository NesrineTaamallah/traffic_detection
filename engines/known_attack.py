"""
engines/known_attack.py
=======================
Pipeline hiérarchique XGBoost :
  1. Modèle binaire   : Normal vs Attack
  2. Modèle multi-classe : type d'attaque (si attack détecté)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_COLS = [
    'dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',
    'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'ct_state_ttl','ct_flw_http_mthd','ct_srv_src','ct_srv_dst',
    'ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm',
    'ct_dst_sport_ltm','ct_dst_src_ltm'
]
HIGH_SKEW = ['sbytes','dbytes','Sload','Dload','Sjit','smeansz','dmeansz']


class KnownAttackEngine:
    """Wrapper du pipeline XGBoost hiérarchique."""

    def __init__(self, models_dir: str = "./models"):
        base = Path(models_dir)
        self.binary     = joblib.load(base / "best_binary_model.pkl")
        self.multiclass = joblib.load(base / "xgb_hierarchical_multiclass.pkl")
        self.scaler     = joblib.load(base / "scaler_hierarchical.pkl")
        self.label_enc  = joblib.load(base / "label_encoder_hierarchical.pkl")
        self.pt         = None
        pt_path = base / "powertransformer_hierarchical.pkl"
        if pt_path.exists():
            self.pt = joblib.load(pt_path)
        self.hsk = [f for f in HIGH_SKEW if f in FEATURE_COLS]
        print(f"[KnownAttack] Modèles chargés depuis {models_dir}")

    def _preprocess(self, raw: dict) -> pd.DataFrame:
        X = pd.DataFrame([{col: raw.get(col, 0) for col in FEATURE_COLS}])
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        X_sc = pd.DataFrame(self.scaler.transform(X), columns=FEATURE_COLS)
        if self.pt and self.hsk:
            X_sc[self.hsk] = self.pt.transform(X_sc[self.hsk])
        return X_sc

    def predict(self, features: dict) -> dict:
        X = self._preprocess(features)

        is_attack = int(self.binary.predict(X)[0])
        bin_proba = float(self.binary.predict_proba(X)[0][1])

        if not is_attack:
            return {
                "is_attack":   False,
                "label":       "Normal",
                "attack_type": None,
                "confidence":  round(1 - bin_proba, 4),
                "bin_proba":   round(bin_proba, 4),
            }

        class_idx   = self.multiclass.predict(X)[0]
        class_prob  = self.multiclass.predict_proba(X)[0]
        attack_type = self.label_enc.inverse_transform([class_idx])[0]

        return {
            "is_attack":   True,
            "label":       "Attack",
            "attack_type": attack_type,
            "confidence":  round(float(class_prob.max()), 4),
            "bin_proba":   round(bin_proba, 4),
            "all_probs": {
                cls: round(float(p), 4)
                for cls, p in zip(self.label_enc.classes_, class_prob)
            },
        }