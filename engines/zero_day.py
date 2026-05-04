"""
engines/zero_day.py — CORRIGÉ v3
==================================
FIXES v3 :
[FIX-1] Import numpy_compat_patch EN PREMIER pour corriger np.Inf → NumPy 2.0
[FIX-2] configure_n_features() protégé contre double-init
[FIX-3] n_features=0 par défaut → auto-config au 1er vecteur AfterImage
[FIX-4] _process_raw() : gestion d'erreur améliorée avec reset si KitNET corrompu
"""

# ── PATCH NUMPY 2.0 — DOIT ÊTRE EN PREMIER ────────────────────────
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'Infinity'):
    np.Infinity = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
# ──────────────────────────────────────────────────────────────────

import sys
import pickle
from pathlib import Path

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

FALLBACK_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'smeansz', 'dmeansz',
    'Sjit', 'Sintpkt', 'tcprtt', 'synack'
]


class ZeroDayEngine:
    """
    Détection d'anomalies via KitNET.
    n_features=0 → auto-config au 1er vecteur AfterImage reçu.
    """

    def __init__(
        self,
        fm_grace:      int   = 5_000,
        ad_grace:      int   = 50_000,
        max_ae_size:   int   = 10,
        learning_rate: float = 0.1,
        n_features:    int   = 0,
    ):
        self.fm_grace      = fm_grace
        self.ad_grace      = ad_grace
        self.grace_total   = fm_grace + ad_grace
        self.max_ae_size   = max_ae_size
        self.learning_rate = learning_rate
        self._n            = n_features if n_features > 0 else len(FALLBACK_FEATURES)
        self._kitnet       = None
        self._configured   = (n_features > 0)

        if KITNET_AVAILABLE and n_features > 0:
            self._kitnet = self._build_kitnet(n_features)
            print(f"[KitNET] Initialisé avec {self._n} features")
        elif KITNET_AVAILABLE:
            print(f"[KitNET] n_features=0 → auto-config au 1er vecteur AfterImage")

        self.rmse_history: list[float] = []
        self.threshold:    float       = 0.1
        self.packet_count: int         = 0
        self.trained:      bool        = False
        self._mode = "afterimage" if KITNET_AVAILABLE else "unavailable"
        self._consecutive_errors: int  = 0

    def _build_kitnet(self, n: int):
        """Construit un KitNET propre. Isolé pour faciliter le reset."""
        try:
            k = kit.KitNET(
                n                    = n,
                max_autoencoder_size = self.max_ae_size,
                FM_grace_period      = self.fm_grace,
                AD_grace_period      = self.ad_grace,
                learning_rate        = self.learning_rate,
                hidden_ratio         = 0.75,
            )
            return k
        except Exception as e:
            print(f"[KitNET] Erreur construction : {e}")
            return None

    def configure_n_features(self, n: int):
        """
        Reconfigure KitNET avec le vrai nombre de features AfterImage.
        Appelé automatiquement au 1er vecteur si n_features=0.
        """
        if n == self._n and self._configured and self._kitnet is not None:
            return
        if not KITNET_AVAILABLE:
            return

        print(f"[KitNET] Reconfiguration : {self._n} → {n} features")
        self._n = n
        self._kitnet = self._build_kitnet(n)
        if self._kitnet is not None:
            self._configured = True
            self._consecutive_errors = 0
            print(f"[KitNET] KitNET recréé avec {n} features — OK")
        else:
            print(f"[KitNET] ERREUR : échec de la construction avec {n} features")

    def process_vector(self, vec: np.ndarray) -> dict:
        """Traite un vecteur AfterImage (mode optimal)."""
        if not KITNET_AVAILABLE:
            return self._unavailable_result()

        # Auto-config au 1er appel si pas encore configuré
        if not self._configured or self._kitnet is None:
            self.configure_n_features(len(vec))

        if self._kitnet is None:
            return self._unavailable_result()

        if len(vec) != self._n:
            self.configure_n_features(len(vec))

        return self._process_raw(vec)

    def process(self, features: dict) -> dict:
        """Mode dégradé — features UNSW-NB15 par flux."""
        if self._kitnet is None:
            if not self._configured and KITNET_AVAILABLE:
                self.configure_n_features(len(FALLBACK_FEATURES))
            else:
                return self._unavailable_result()
        vec = np.array(
            [float(features.get(col, 0)) for col in FALLBACK_FEATURES],
            dtype=np.float64
        )
        return self._process_raw(vec)

    def _process_raw(self, vec: np.ndarray) -> dict:
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            rmse = float(self._kitnet.process(vec))
            self._consecutive_errors = 0  # reset sur succès
        except Exception as e:
            self._consecutive_errors += 1
            # Logguer seulement le 1er et les multiples de 100
            if self._consecutive_errors == 1 or self._consecutive_errors % 100 == 0:
                print(f"[KitNET] process() error #{self._consecutive_errors} : {e}")
            # Si trop d'erreurs consécutives → tenter de reconstruire
            if self._consecutive_errors >= 50:
                print(f"[KitNET] Trop d'erreurs ({self._consecutive_errors}), reconstruction...")
                self._configured = False
                self._kitnet = None
                self._consecutive_errors = 0
            return self._unavailable_result()

        self.rmse_history.append(rmse)
        self.packet_count += 1

        if self.packet_count == self.grace_total:
            window = np.array(self.rmse_history[-min(10_000, len(self.rmse_history)):])
            self.threshold = float(np.mean(window) + 3 * np.std(window))
            self.trained   = True
            print(f"[KitNET] ✓ Entraîné — seuil automatique = {self.threshold:.6f}")

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
        e._kitnet             = s["kitnet"]
        e.threshold           = s["threshold"]
        e.trained             = s["trained"]
        e.packet_count        = s["packet_count"]
        e.rmse_history        = s.get("rmse_history", [])
        e.fm_grace            = s.get("fm_grace", 5_000)
        e.ad_grace            = s.get("ad_grace", 50_000)
        e.grace_total         = e.fm_grace + e.ad_grace
        e._n                  = s.get("n_features", len(FALLBACK_FEATURES))
        e._mode               = s.get("mode", "afterimage")
        e._configured         = True
        e.max_ae_size         = 10
        e.learning_rate       = 0.1
        e._consecutive_errors = 0
        print(f"[KitNET] Chargé depuis {path} — {e.packet_count} paquets")
        return e

    @property
    def n_features(self) -> int:
        return self._n