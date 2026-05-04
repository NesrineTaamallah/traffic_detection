"""
engines/zero_day.py
===================
Wrapper KitNET pour la détection d'anomalies zero-day.

Mode normal   : reçoit vecteurs AfterImage (~115 features) via process_vector()
Mode dégradé  : reçoit features UNSW-NB15 (13 colonnes) via process()
Mode unavailable : KitNET-py non cloné
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Cherche KitNET.py dans Kitsune-py ou KitNET-py
_PROJECT_ROOT = Path(__file__).parent.parent
_CANDIDATES = [
    _PROJECT_ROOT / "Kitsune-py",
    _PROJECT_ROOT / "KitNET-py",
    _PROJECT_ROOT / "kitsune-py",
    _PROJECT_ROOT / "kitnet-py",
]
for _c in _CANDIDATES:
    if _c.exists():
        sys.path.insert(0, str(_c))

try:
    import KitNET as kit
    KITNET_AVAILABLE = True
    print("[OK] KitNET disponible")
except ImportError:
    KITNET_AVAILABLE = False
    print("[WARN] KitNET non trouvé")
    print("       git clone https://github.com/ymirsky/Kitsune-py.git")

FALLBACK_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'smeansz', 'dmeansz',
    'Sjit', 'Sintpkt', 'tcprtt', 'synack'
]


class ZeroDayEngine:
    """
    Détection d'anomalies via KitNET.

    Paramètres
    ----------
    fm_grace      : paquets pour la phase Feature Mapping
    ad_grace      : paquets pour la phase Anomaly Detection
    max_ae_size   : taille max de chaque autoencoder
    learning_rate : taux d'apprentissage
    n_features    : 0 = auto-détecté depuis AfterImage
    """

    def __init__(
        self,
        fm_grace:      int   = 5_000,
        ad_grace:      int   = 50_000,
        max_ae_size:   int   = 10,
        learning_rate: float = 0.1,
        n_features:    int   = 0,
    ):
        self.fm_grace    = fm_grace
        self.ad_grace    = ad_grace
        self.grace_total = fm_grace + ad_grace
        self._n          = n_features if n_features > 0 else len(FALLBACK_FEATURES)
        self._kitnet     = None

        if KITNET_AVAILABLE:
            self._kitnet = kit.KitNET(
                n                    = self._n,
                max_autoencoder_size = max_ae_size,
                FM_grace_period      = fm_grace,
                AD_grace_period      = ad_grace,
                learning_rate        = learning_rate,
                hidden_ratio         = 0.75,
            )

        self.rmse_history: list[float] = []
        self.threshold:    float       = 0.1
        self.packet_count: int         = 0
        self.trained:      bool        = False
        self._mode = "afterimage" if KITNET_AVAILABLE else "unavailable"

    def configure_n_features(self, n: int):
        """Reconfigure KitNET si le nombre réel de features diffère."""
        if n == self._n or not KITNET_AVAILABLE or self.packet_count > 0:
            return
        self._n = n
        max_ae = self._kitnet.FM.maxAE if self._kitnet else 10
        self._kitnet = kit.KitNET(
            n=n, max_autoencoder_size=max_ae,
            FM_grace_period=self.fm_grace,
            AD_grace_period=self.ad_grace,
            learning_rate=0.1, hidden_ratio=0.75,
        )

    def process_vector(self, vec: np.ndarray) -> dict:
        """Traite un vecteur AfterImage (mode optimal)."""
        if self._kitnet is None:
            return self._unavailable_result()
        if len(vec) != self._n and self.packet_count == 0:
            self.configure_n_features(len(vec))
        return self._process_raw(vec)

    def process(self, features: dict) -> dict:
        """Mode dégradé — features UNSW-NB15 par flux."""
        if self._kitnet is None:
            return self._unavailable_result()
        vec = np.array([float(features.get(col, 0)) for col in FALLBACK_FEATURES], dtype=np.float64)
        return self._process_raw(vec)

    def _process_raw(self, vec: np.ndarray) -> dict:
        vec  = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        rmse = float(self._kitnet.process(vec))
        self.rmse_history.append(rmse)
        self.packet_count += 1

        if self.packet_count == self.grace_total:
            window = np.array(self.rmse_history[-min(10_000, len(self.rmse_history)):])
            self.threshold = float(np.mean(window) + 3 * np.std(window))
            self.trained   = True
            print(f"[KitNET] Entraîné — seuil automatique = {self.threshold:.6f}")

        progress = min(self.packet_count / self.grace_total, 1.0)

        if not self.trained:
            return {
                "rmse":           round(rmse, 6),
                "is_anomaly":     False,
                "phase":          "FM" if self.packet_count < self.fm_grace else "AD",
                "progress":       round(progress, 4),
                "threshold":      self.threshold,
                "severity_score": 0.0,
                "trained":        False,
                "mode":           self._mode,
            }

        sev_score  = rmse / max(self.threshold, 1e-9)
        is_anomaly = rmse > self.threshold
        return {
            "rmse":           round(rmse, 6),
            "is_anomaly":     is_anomaly,
            "phase":          "monitoring",
            "progress":       1.0,
            "threshold":      round(self.threshold, 6),
            "severity_score": round(min(sev_score, 10.0), 3),
            "trained":        True,
            "mode":           self._mode,
        }

    def _unavailable_result(self) -> dict:
        return {
            "rmse": 0.0, "is_anomaly": False,
            "phase": "unavailable", "progress": 0.0,
            "threshold": 0.0, "severity_score": 0.0,
            "trained": False, "mode": "unavailable",
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "kitnet":       self._kitnet,
                "threshold":    self.threshold,
                "trained":      self.trained,
                "packet_count": self.packet_count,
                "rmse_history": self.rmse_history[-5_000:],
                "fm_grace":     self.fm_grace,
                "ad_grace":     self.ad_grace,
                "n_features":   self._n,
                "mode":         self._mode,
            }, f)
        print(f"[KitNET] Sauvegardé → {path}")

    @classmethod
    def load(cls, path: str) -> "ZeroDayEngine":
        with open(path, "rb") as f:
            s = pickle.load(f)
        e = cls.__new__(cls)
        e._kitnet       = s["kitnet"]
        e.threshold     = s["threshold"]
        e.trained       = s["trained"]
        e.packet_count  = s["packet_count"]
        e.rmse_history  = s.get("rmse_history", [])
        e.fm_grace      = s.get("fm_grace", 5_000)
        e.ad_grace      = s.get("ad_grace", 50_000)
        e.grace_total   = e.fm_grace + e.ad_grace
        e._n            = s.get("n_features", len(FALLBACK_FEATURES))
        e._mode         = s.get("mode", "afterimage")
        print(f"[KitNET] Chargé depuis {path} — {e.packet_count} paquets")
        return e

    @property
    def n_features(self) -> int:
        return self._n