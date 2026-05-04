"""
engines/zero_day.py — v5 PRETRAINED
=====================================

ROOT CAUSE OF FALSE POSITIVES (was v4):
────────────────────────────────────────
KitNET was trained ONLINE on live Wi-Fi traffic.
After 55 000 packets the threshold was set to 0.08037 from your home
network's normal traffic profile.  But live traffic has high variance
(HTTPS, DNS bursts, TLS handshakes…) so most packets consistently score
RMSE > 0.08  → EVERYTHING flagged as Zero-day.

FIX — Pre-trained model approach:
──────────────────────────────────
Load  models/kitsune_mirai_model.pkl  (trained offline on the Mirai PCAP
dataset).  This model:
  • Is already in execute-mode (no grace period needed)
  • Has a threshold calibrated to Mirai dataset's BENIGN traffic profile
  • Detects Mirai-style botnet traffic reliably

Fallback: if the pkl is not found, fall back to online training (original
v4 behaviour) with a MUCH higher threshold safety factor (×4 instead of
×1.5) to reduce false positives.

PKL format support:
───────────────────
We handle every structure seen in the wild from Kitsune-py:
  1. dict  with keys  'model'/'kitnet'/'KitNET' + 'threshold'/'FPR'/'th'
  2. Raw   KitNET  object saved directly  (pickle.dump(kitnet, f))
  3. list  [kitnet_obj, threshold]
  4. dict  with key 'stats' containing the threshold (Mirsky's original format)
"""

import sys
import pickle
import io
import numpy as np
from pathlib import Path
from collections import deque

# ── numpy compat patch (NumPy ≥ 2.0) ─────────────────────────────
for _attr, _val in [('Inf', np.inf), ('Infinity', np.inf), ('NaN', np.nan),
                    ('bool', bool), ('int', int), ('float', float)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _val)

# ── Locate KitNET-py / Kitsune-py ────────────────────────────────
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
    print("[WARN] KitNET non disponible")


# ── Custom Unpickler — fixes module-path mismatches in saved pkls ─
# The Mirai pkl was saved when KitNET was importable as 'KitNET.KitNET'
# (package layout), but in KitNET-py it's a flat module 'KitNET'.
# We intercept find_class() to remap all known variants.
class _KitNETUnpickler(pickle.Unpickler):
    """
    Redirect old KitNET pickle module paths to current flat-module layout.

    The Mirai pkl was saved when KitNET was a package (KitNET/KitNET.py),
    so pickle stored class references as ('KitNET.KitNET', 'KitNET').
    In KitNET-py the layout is flat (KitNET.py at root), so we must
    intercept find_class and resolve the class directly from sys.modules
    WITHOUT triggering a new import of 'KitNET.KitNET' (which doesn't exist).
    """

    # Classes that live in KitNET.py (the flat module)
    _KITNET_CLASSES = {'KitNET', 'corMicro', 'dA', 'AE', 'AE_row', 'corClust'}

    def find_class(self, module, name):
        # Any reference to KitNET.* submodule → resolve from the flat KitNET module
        if (module == "KitNET.KitNET"
                or module.startswith("KitNET.")
                or module.startswith("kitsune.")
                or module.startswith("Kitsune.")):
            # The actual module is the flat 'KitNET' already in sys.modules
            flat_mod = sys.modules.get("KitNET")
            if flat_mod is not None and hasattr(flat_mod, name):
                return getattr(flat_mod, name)
            # Fallback: try the last segment as module name
            flat_name = module.split(".")[-1]
            flat_mod2 = sys.modules.get(flat_name)
            if flat_mod2 is not None and hasattr(flat_mod2, name):
                return getattr(flat_mod2, name)
            # Last resort: import and get
            try:
                import importlib
                mod = importlib.import_module(flat_name)
                return getattr(mod, name)
            except Exception:
                pass
        return super().find_class(module, name)


def _safe_pickle_load(path: str):
    """Load a pkl using the patched unpickler."""
    with open(path, "rb") as f:
        data = f.read()
    return _KitNETUnpickler(io.BytesIO(data)).load()

# ── Constants ─────────────────────────────────────────────────────
FALLBACK_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'smeansz', 'dmeansz',
    'Sjit', 'Sintpkt', 'tcprtt', 'synack'
]

# Online-training thresholding parameters (fallback mode only)
MIN_REAL_RMSE_FOR_THRESHOLD = 2_000
THRESHOLD_PERCENTILE        = 99
THRESHOLD_SAFETY_FACTOR     = 4.0

# Pre-trained mode: warm-up calibration
# During warm-up, ALL packets are scored but NO alerts fire.
# After PRETRAINED_WARMUP_PKTS packets, the threshold is set at
# PRETRAINED_THRESHOLD_PERCENTILE of the observed live RMSE distribution
# × PRETRAINED_SAFETY_FACTOR.  This adapts the Mirai model's threshold
# to YOUR network's traffic profile automatically.
PRETRAINED_WARMUP_PKTS          = 3_000   # ~1 min at 50 pps
PRETRAINED_THRESHOLD_PERCENTILE = 99      # 99th percentile of live "normal" traffic
PRETRAINED_SAFETY_FACTOR        = 3.0    # ×3 above P99 → very few false positives


# ══════════════════════════════════════════════════════════════════
class ZeroDayEngine:
    """
    KitNET wrapper supporting two modes:

    PRE-TRAINED MODE  (recommended)
    ────────────────
    Load a pkl trained offline on a labelled dataset (e.g. Mirai PCAP).
    No grace period.  Starts detecting immediately.
    Threshold is read from the pkl (calibrated to the training dataset).

    ONLINE MODE  (fallback)
    ─────────────
    Train on live traffic.  Needs fm_grace + ad_grace packets before the
    threshold is computed.  Prone to FPs if live traffic is diverse.
    """

    # ── Constructor ───────────────────────────────────────────────
    def __init__(
        self,
        fm_grace:      int   = 5_000,
        ad_grace:      int   = 50_000,
        max_ae_size:   int   = 10,
        learning_rate: float = 0.1,
        n_features:    int   = 0,
        pretrained_path: str | None = None,   # ← NEW
    ):
        self.fm_grace      = fm_grace
        self.ad_grace      = ad_grace
        self.grace_total   = fm_grace + ad_grace
        self.max_ae_size   = max_ae_size
        self.learning_rate = learning_rate

        self.rmse_history:    list[float] = []
        self._post_grace_rmse: deque      = deque(maxlen=10_000)
        self.threshold:  float = 0.1
        self.packet_count: int = 0
        self.trained:      bool = False
        self._kitnet_executing: bool = False
        self._real_rmse_count:  int  = 0
        self._consecutive_errors: int = 0

        # ── Try to load pre-trained model ─────────────────────────
        self._pretrained = False
        if pretrained_path:
            loaded = self._try_load_pretrained(pretrained_path)
            if loaded:
                self._pretrained = True
                self._mode = "pretrained"
                print(f"[KitNET] ✓ Modèle pré-entraîné chargé : {pretrained_path}")
                print(f"[KitNET]   n_features={self._n}  threshold(original)={self.threshold:.6f}")
                print(f"[KitNET]   Warm-up live: {PRETRAINED_WARMUP_PKTS} paquets avant détection")
                return   # skip normal init

        # ── Normal online init ────────────────────────────────────
        self._n = n_features if n_features > 0 else len(FALLBACK_FEATURES)
        self._kitnet = None
        self._configured = (n_features > 0)
        self._mode = "online"

        if KITNET_AVAILABLE and n_features > 0:
            self._kitnet = self._build_kitnet(n_features)
            print(f"[KitNET] Mode online initialisé avec {self._n} features")
        elif KITNET_AVAILABLE:
            print(f"[KitNET] Mode online — auto-config au 1er vecteur AfterImage")

    # ── Pre-trained loader ────────────────────────────────────────
    def _try_load_pretrained(self, path: str) -> bool:
        """
        Attempt to load a pre-trained KitNET pickle.
        Uses _KitNETUnpickler to handle module-path mismatches
        (e.g. pkl saved with 'KitNET.KitNET' but loaded in flat layout).
        Returns True on success.
        """
        p = Path(path)
        if not p.exists():
            print(f"[KitNET] PKL non trouvé : {path}")
            return False

        try:
            obj = _safe_pickle_load(str(p))
            print(f"[KitNET] PKL chargé — type: {type(obj).__name__}")
        except Exception as e:
            print(f"[KitNET] Erreur fatale PKL : {e}")
            print(f"[KitNET] Conseil: vérifiez que KitNET-py est dans sys.path et importé")
            return False

        # ── Format detection ──────────────────────────────────────
        kitnet_obj = None
        threshold  = None
        n_features = None

        # Format 1: dict with named keys
        if isinstance(obj, dict):
            print(f"[KitNET] PKL dict keys: {list(obj.keys())}")

            # Try to find the KitNET object
            for key in ('model', 'kitnet', 'KitNET', 'kit', 'detector', 'engine'):
                if key in obj:
                    kitnet_obj = obj[key]
                    break

            # Try to find threshold
            for key in ('threshold', 'FPR', 'th', 'anomaly_threshold',
                        'rmse_threshold', 'thr', 'Threshold'):
                if key in obj and isinstance(obj[key], (int, float)):
                    threshold = float(obj[key])
                    break

            # Try to find n_features
            for key in ('n_features', 'n', 'num_features', 'features'):
                if key in obj and isinstance(obj[key], int):
                    n_features = obj[key]
                    break

            # Mirsky original format: dict contains stats array
            if 'stats' in obj and threshold is None:
                stats = obj['stats']
                if hasattr(stats, '__len__') and len(stats) > 0:
                    arr = np.array(stats, dtype=float)
                    threshold = float(np.percentile(arr, 99)) * 1.5
                    print(f"[KitNET] Seuil calculé depuis 'stats' P99: {threshold:.6f}")

            # If threshold still not found but there's an rmse array
            if threshold is None:
                for key in ('rmse', 'rmse_history', 'benign_rmse', 'train_rmse'):
                    if key in obj:
                        arr = np.array(obj[key], dtype=float)
                        arr = arr[arr > 0]   # remove zeros from grace period
                        if len(arr) >= 10:
                            threshold = float(np.percentile(arr, 99)) * 2.0
                            print(f"[KitNET] Seuil recalculé depuis '{key}' P99×2: {threshold:.6f}")
                            break

        # Format 2: raw KitNET object
        elif KITNET_AVAILABLE and isinstance(obj, kit.KitNET):
            kitnet_obj = obj
            print(f"[KitNET] PKL = objet KitNET direct")

        # Format 3: list [kitnet, threshold]
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            kitnet_obj = obj[0]
            if isinstance(obj[1], (int, float)):
                threshold = float(obj[1])
            print(f"[KitNET] PKL = liste [{type(obj[0]).__name__}, {obj[1]}]")

        # Format 4: KitNET with threshold attribute stored on it
        if kitnet_obj is not None and threshold is None:
            for attr in ('threshold', 'FPR', 'anomaly_threshold', '_threshold'):
                if hasattr(kitnet_obj, attr):
                    v = getattr(kitnet_obj, attr)
                    if isinstance(v, (int, float)) and v > 0:
                        threshold = float(v)
                        print(f"[KitNET] Seuil depuis attribut '{attr}': {threshold:.6f}")
                        break

        # Try to get n_features from KitNET object
        if kitnet_obj is not None and n_features is None:
            for attr in ('n', 'num_features', 'n_features', 'FM_n'):
                if hasattr(kitnet_obj, attr):
                    v = getattr(kitnet_obj, attr)
                    if isinstance(v, int) and v > 0:
                        n_features = v
                        break

        # ── Validate ──────────────────────────────────────────────
        if kitnet_obj is None:
            print(f"[KitNET] WARN: impossible d'extraire l'objet KitNET du PKL")
            print(f"[KitNET] Contenu PKL: {type(obj)}")
            # Last resort: maybe the object IS the kitnet
            if hasattr(obj, 'process'):
                kitnet_obj = obj
                print(f"[KitNET] Objet PKL possède .process() → utilisation directe")
            else:
                return False

        if threshold is None or threshold <= 0:
            print(f"[KitNET] Seuil original invalide ({threshold}) → sera recalibré sur trafic live")
            threshold = float('inf')   # block all alerts until calibration done
        else:
            print(f"[KitNET] Seuil original du modèle = {threshold:.6f}")
            print(f"[KitNET] Ce seuil sera REMPLACÉ après warm-up live ({PRETRAINED_WARMUP_PKTS} paquets)")
            threshold = float('inf')   # always recalibrate — Mirai ≠ your traffic

        n_features = n_features or 100   # AfterImage default

        # ── Apply ─────────────────────────────────────────────────
        self._kitnet   = kitnet_obj
        self._n        = n_features
        self._configured = True
        self.threshold = threshold        # inf until warm-up completes
        self.trained   = False            # no alerts until threshold is calibrated
        self._kitnet_executing = True
        self._real_rmse_count  = 1        # bypass online grace detection

        # Warm-up calibration state
        # We collect RMSE scores from your live traffic for PRETRAINED_WARMUP_PKTS
        # packets, then set the threshold at P99 × PRETRAINED_SAFETY_FACTOR.
        # During warm-up: trained=False → no alerts, dashboard shows "warm-up"
        self._warmup_rmse:  list[float] = []
        self._warmup_done:  bool        = False

        return True

    # ── Build online KitNET ───────────────────────────────────────
    def _build_kitnet(self, n: int):
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

    # ── Auto-configure n_features ─────────────────────────────────
    def configure_n_features(self, n: int):
        if self._pretrained:
            return   # never reconfigure a pre-trained model
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
        else:
            print(f"[KitNET] ERREUR : échec construction avec {n} features")

    # ── Public API ────────────────────────────────────────────────
    def process_vector(self, vec: np.ndarray) -> dict:
        """Process a raw AfterImage feature vector."""
        if not KITNET_AVAILABLE:
            return self._unavailable_result()

        if self._pretrained:
            # Pre-trained: never reconfigure, just process
            pass
        else:
            if not self._configured or self._kitnet is None:
                self.configure_n_features(len(vec))
            if self._kitnet is None:
                return self._unavailable_result()
            if len(vec) != self._n:
                self.configure_n_features(len(vec))

        return self._process_raw(vec)

    def process(self, features: dict) -> dict:
        """Process a feature dict (UNSW-NB15 format)."""
        if self._kitnet is None:
            if not self._pretrained and not self._configured and KITNET_AVAILABLE:
                self.configure_n_features(len(FALLBACK_FEATURES))
            else:
                return self._unavailable_result()
        vec = np.array(
            [float(features.get(col, 0)) for col in FALLBACK_FEATURES],
            dtype=np.float64
        )
        return self._process_raw(vec)

    # ── Core processing ───────────────────────────────────────────
    def _process_raw(self, vec: np.ndarray) -> dict:
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            rmse = float(self._kitnet.process(vec))
            self._consecutive_errors = 0
        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors % 50 == 1:
                print(f"[KitNET] process() erreur #{self._consecutive_errors}: {e}")
            if self._consecutive_errors >= 50 and not self._pretrained:
                print(f"[KitNET] Trop d'erreurs, reconstruction…")
                self._configured = False
                self._kitnet = None
                self._consecutive_errors = 0
            return self._unavailable_result()

        self.rmse_history.append(rmse)
        self.packet_count += 1

        # ── Pre-trained mode ──────────────────────────────────────
        if self._pretrained:
            # ── Warm-up phase: collect live RMSE, no alerts ───────
            if not self._warmup_done:
                if rmse > 0:
                    self._warmup_rmse.append(rmse)

                n_collected = len(self._warmup_rmse)
                progress = min(n_collected / PRETRAINED_WARMUP_PKTS, 1.0)

                if n_collected >= PRETRAINED_WARMUP_PKTS:
                    arr = np.array(self._warmup_rmse)
                    p_val = float(np.percentile(arr, PRETRAINED_THRESHOLD_PERCENTILE))
                    self.threshold  = p_val * PRETRAINED_SAFETY_FACTOR
                    self._warmup_done = True
                    self.trained      = True
                    print(
                        f"[KitNET] ✓ Warm-up terminé — seuil live = {self.threshold:.4f} "
                        f"(P{PRETRAINED_THRESHOLD_PERCENTILE}={p_val:.4f} × {PRETRAINED_SAFETY_FACTOR}) "
                        f"sur {n_collected} paquets"
                    )

                return {
                    "rmse":            round(rmse, 6),
                    "is_anomaly":      False,
                    "phase":           "warmup",
                    "progress":        round(progress, 4),
                    "threshold":       0.0,   # unknown until warm-up done
                    "severity_score":  0.0,
                    "trained":         False,
                    "mode":            self._mode,
                    "real_rmse_count": self.packet_count,
                }

            # ── Detection phase: threshold calibrated to live traffic ─
            is_anomaly = rmse > self.threshold
            sev_score  = rmse / max(self.threshold, 1e-9)
            return {
                "rmse":            round(rmse, 6),
                "is_anomaly":      is_anomaly,
                "phase":           "pretrained",
                "progress":        1.0,
                "threshold":       round(self.threshold, 6),
                "severity_score":  round(min(sev_score, 10.0), 3),
                "trained":         True,
                "mode":            self._mode,
                "real_rmse_count": self.packet_count,
            }

        # ── Online mode ───────────────────────────────────────────
        if rmse > 0.0:
            self._real_rmse_count += 1
            self._post_grace_rmse.append(rmse)
            if not self._kitnet_executing and self._real_rmse_count >= 10:
                self._kitnet_executing = True
                print(f"[KitNET] Mode execute détecté après {self.packet_count} paquets")

        if (not self.trained
                and self._kitnet_executing
                and self._real_rmse_count >= MIN_REAL_RMSE_FOR_THRESHOLD):
            arr = np.array(list(self._post_grace_rmse))
            p99 = float(np.percentile(arr, THRESHOLD_PERCENTILE))
            self.threshold = max(p99 * THRESHOLD_SAFETY_FACTOR, 1e-6)
            self.trained   = True
            print(
                f"[KitNET] ✓ Entraîné — seuil = {self.threshold:.6f} "
                f"(P{THRESHOLD_PERCENTILE}={p99:.6f} × {THRESHOLD_SAFETY_FACTOR})"
                f" sur {self._real_rmse_count} RMSE réels"
            )

        progress = min(self.packet_count / self.grace_total, 1.0)

        if not self.trained:
            return {
                "rmse":            round(rmse, 6),
                "is_anomaly":      False,
                "phase":           "FM" if self.packet_count < self.fm_grace else "AD",
                "progress":        round(progress, 4),
                "threshold":       self.threshold,
                "severity_score":  0.0,
                "trained":         False,
                "mode":            self._mode,
                "real_rmse_count": self._real_rmse_count,
            }

        sev_score  = rmse / max(self.threshold, 1e-9)
        is_anomaly = rmse > self.threshold
        return {
            "rmse":            round(rmse, 6),
            "is_anomaly":      is_anomaly,
            "phase":           "monitoring",
            "progress":        1.0,
            "threshold":       round(self.threshold, 6),
            "severity_score":  round(min(sev_score, 10.0), 3),
            "trained":         True,
            "mode":            self._mode,
            "real_rmse_count": self._real_rmse_count,
        }

    # ── Unavailable fallback ──────────────────────────────────────
    def _unavailable_result(self) -> dict:
        return {
            "rmse": 0.0, "is_anomaly": False,
            "phase": "unavailable", "progress": 0.0,
            "threshold": 0.0, "severity_score": 0.0,
            "trained": False, "mode": "unavailable",
            "real_rmse_count": 0,
        }

    # ── Persistence ───────────────────────────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "kitnet":           self._kitnet,
                "threshold":        self.threshold,
                "trained":          self.trained,
                "packet_count":     self.packet_count,
                "rmse_history":     self.rmse_history[-5_000:],
                "post_grace_rmse":  list(self._post_grace_rmse),
                "real_rmse_count":  self._real_rmse_count,
                "kitnet_executing": self._kitnet_executing,
                "fm_grace":         self.fm_grace,
                "ad_grace":         self.ad_grace,
                "n_features":       self._n,
                "mode":             self._mode,
                "pretrained":       self._pretrained,
            }, f)
        print(f"[KitNET] Sauvegardé → {path}")

    @classmethod
    def load(cls, path: str) -> "ZeroDayEngine":
        with open(path, "rb") as f:
            s = pickle.load(f)
        e = cls.__new__(cls)
        e._kitnet               = s["kitnet"]
        e.threshold             = s["threshold"]
        e.trained               = s["trained"]
        e.packet_count          = s["packet_count"]
        e.rmse_history          = s.get("rmse_history", [])
        e._post_grace_rmse      = deque(s.get("post_grace_rmse", []), maxlen=10_000)
        e._real_rmse_count      = s.get("real_rmse_count", 0)
        e._kitnet_executing     = s.get("kitnet_executing", False)
        e.fm_grace              = s.get("fm_grace", 5_000)
        e.ad_grace              = s.get("ad_grace", 50_000)
        e.grace_total           = e.fm_grace + e.ad_grace
        e._n                    = s.get("n_features", len(FALLBACK_FEATURES))
        e._mode                 = s.get("mode", "online")
        e._pretrained           = s.get("pretrained", False)
        e._configured           = True
        e.max_ae_size           = 10
        e.learning_rate         = 0.1
        e._consecutive_errors   = 0
        e._calib_done           = True
        e._calib_rmse           = []
        e._threshold_needs_calibration = False
        print(f"[KitNET] Chargé depuis {path} — {e.packet_count} paquets")
        return e

    @property
    def n_features(self) -> int:
        return self._n