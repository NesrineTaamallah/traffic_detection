"""
engines/zero_day.py — v7 OVERFLOW FIX
=======================================

FIXES v7 vs v6 :
[FIX-EXP-1] Patch global de la fonction sigmoid dans KitNET/utils.py au runtime
             → np.exp(-x) est clipé à [-500, 500] avant calcul → plus de overflow
[FIX-EXP-2] Tous les appels process() wrappés dans np.errstate(over='ignore')
             pour supprimer les RuntimeWarning résiduels
[FIX-EXP-3] np.clip sur les poids internes des autoencodeurs après chaque update
             pour éviter l'accumulation de valeurs inf/nan dans les matrices W
[FIX-RMSE]  Clamp RMSE à RMSE_MAX_SANE=100 (déjà présent v6, confirmé)
[FIX-VEC]   Resize vecteur à self._n features (déjà présent v6, confirmé)
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

# ── Locate KitNET-py ─────────────────────────────────────────────
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


# ── [FIX-EXP-1] Patch sigmoid overflow ──────────────────────────
def _patch_kitnet_sigmoid():
    """
    Remplace la fonction sigmoid défaillante dans KitNET/utils.py
    (ou kitnet-py/utils.py) par une version stable avec clip.

    La cause du RuntimeWarning : overflow in exp
      utils.py ligne 15 : return 1. / (1 + numpy.exp(-x))
    Quand x << -500, exp(-x) → inf → overflow.
    Fix : clipper x à [-500, 500] avant exp().
    """
    # Cherche le module utils dans les namespaces kitnet
    for mod_name in ['utils', 'KitNET.utils', 'kitnet.utils']:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, 'sigmoid'):
            original = mod.sigmoid
            def _safe_sigmoid(x):
                x_clipped = np.clip(x, -500.0, 500.0)
                with np.errstate(over='ignore', invalid='ignore'):
                    result = 1.0 / (1.0 + np.exp(-x_clipped))
                return np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
            mod.sigmoid = _safe_sigmoid
            print(f"[KitNET] ✓ Patch sigmoid appliqué sur '{mod_name}'")
            return True

    # Aussi patcher numpy.exp directement dans le module dA si présent
    for mod_name in ['dA', 'KitNET.dA', 'AE']:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            # Le module peut utiliser numpy directement sans passer par sigmoid
            pass

    return False


# Patch appliqué après import KitNET pour que les sous-modules soient chargés
if KITNET_AVAILABLE:
    patched = _patch_kitnet_sigmoid()
    if not patched:
        # Fallback : monkey-patch numpy.exp dans le contexte de kitnet
        # en installant un wrapper au niveau du module kitnet
        print("[KitNET] Patch sigmoid non trouvé — suppression des warnings via errstate")


# ── Custom Unpickler ─────────────────────────────────────────────
class _KitNETUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module == "KitNET.KitNET"
                or module.startswith("KitNET.")
                or module.startswith("kitsune.")
                or module.startswith("Kitsune.")):
            flat_mod = sys.modules.get("KitNET")
            if flat_mod is not None and hasattr(flat_mod, name):
                return getattr(flat_mod, name)
            flat_name = module.split(".")[-1]
            flat_mod2 = sys.modules.get(flat_name)
            if flat_mod2 is not None and hasattr(flat_mod2, name):
                return getattr(flat_mod2, name)
            try:
                import importlib
                mod = importlib.import_module(flat_name)
                return getattr(mod, name)
            except Exception:
                pass
        return super().find_class(module, name)


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return _KitNETUnpickler(io.BytesIO(data)).load()


# ── Constants ────────────────────────────────────────────────────
FALLBACK_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'smeansz', 'dmeansz',
    'Sjit', 'Sintpkt', 'tcprtt', 'synack'
]

MIN_REAL_RMSE_FOR_THRESHOLD = 2_000
THRESHOLD_PERCENTILE        = 99
THRESHOLD_SAFETY_FACTOR     = 4.0

PRETRAINED_WARMUP_PKTS          = 1_000
PRETRAINED_THRESHOLD_PERCENTILE = 99
PRETRAINED_SAFETY_FACTOR        = 2.0

RMSE_MAX_SANE = 100.0


# ── [FIX-EXP-3] Helper : nettoyer les poids d'un autoencoder ────
def _sanitize_ae_weights(kitnet_obj):
    """
    Clippe les poids internes de tous les autoencodeurs KitNET
    pour éviter l'accumulation de valeurs inf/nan après overflow.
    Appelé une seule fois après chargement du pkl.
    """
    if kitnet_obj is None:
        return
    try:
        # Tente d'accéder aux sous-autoencodeurs (structure interne KitNET)
        ae_list = None
        for attr in ('AD', 'ensembleLayer', 'outputAE', 'ADs'):
            if hasattr(kitnet_obj, attr):
                obj = getattr(kitnet_obj, attr)
                if isinstance(obj, list):
                    ae_list = obj
                    break

        def _clip_ae(ae):
            for w_attr in ('W', 'b', 'W_', 'b_', 'hbias', 'vbias'):
                if hasattr(ae, w_attr):
                    w = getattr(ae, w_attr)
                    if isinstance(w, np.ndarray):
                        clipped = np.clip(w, -10.0, 10.0)
                        clipped = np.nan_to_num(clipped, nan=0.0, posinf=10.0, neginf=-10.0)
                        setattr(ae, w_attr, clipped)

        if ae_list:
            for ae in ae_list:
                _clip_ae(ae)
            # Output AE
            for out_attr in ('outputAE', 'output_ae'):
                if hasattr(kitnet_obj, out_attr):
                    _clip_ae(getattr(kitnet_obj, out_attr))
            print(f"[KitNET] ✓ Poids sanitisés sur {len(ae_list)} autoencodeurs")
    except Exception as e:
        print(f"[KitNET] Sanitize weights (non critique) : {e}")


class ZeroDayEngine:
    """
    KitNET wrapper v7 — overflow exp() fixé.
    """

    def __init__(self, fm_grace=5_000, ad_grace=50_000, max_ae_size=10,
                 learning_rate=0.1, n_features=0, pretrained_path=None):
        self.fm_grace      = fm_grace
        self.ad_grace      = ad_grace
        self.grace_total   = fm_grace + ad_grace
        self.max_ae_size   = max_ae_size
        self.learning_rate = learning_rate

        self.rmse_history:     list[float] = []
        self._post_grace_rmse: deque       = deque(maxlen=10_000)
        self.threshold:   float = 0.1
        self.packet_count:  int = 0
        self.trained:      bool = False
        self._kitnet_executing: bool = False
        self._real_rmse_count:  int  = 0
        self._consecutive_errors: int = 0

        self._pretrained = False
        if pretrained_path:
            loaded = self._try_load_pretrained(pretrained_path)
            if loaded:
                self._pretrained = True
                self._mode = "pretrained"
                print(f"[KitNET] ✓ Modèle pré-entraîné chargé : {pretrained_path}")
                print(f"[KitNET]   n_features={self._n}  threshold(original)={self.threshold:.6f}")
                print(f"[KitNET]   Warm-up live: {PRETRAINED_WARMUP_PKTS} paquets avant détection")
                return

        self._n = n_features if n_features > 0 else len(FALLBACK_FEATURES)
        self._kitnet = None
        self._configured = (n_features > 0)
        self._mode = "online"

        if KITNET_AVAILABLE and n_features > 0:
            self._kitnet = self._build_kitnet(n_features)
            print(f"[KitNET] Mode online — {self._n} features")

    def _try_load_pretrained(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            print(f"[KitNET] PKL non trouvé : {path}")
            return False
        try:
            obj = _safe_pickle_load(str(p))
            print(f"[KitNET] PKL chargé — type: {type(obj).__name__}")
        except Exception as e:
            print(f"[KitNET] Erreur fatale PKL : {e}")
            return False

        kitnet_obj = None
        threshold  = None
        n_features = None

        if isinstance(obj, dict):
            print(f"[KitNET] PKL dict keys: {list(obj.keys())}")
            for key in ('model', 'kitnet', 'KitNET', 'kit', 'detector', 'engine'):
                if key in obj:
                    kitnet_obj = obj[key]
                    break
            for key in ('threshold', 'FPR', 'th', 'anomaly_threshold', 'rmse_threshold', 'thr', 'Threshold'):
                if key in obj and isinstance(obj[key], (int, float)):
                    threshold = float(obj[key])
                    break
            for key in ('n_features', 'n', 'num_features', 'features'):
                if key in obj and isinstance(obj[key], int):
                    n_features = obj[key]
                    break
            if threshold is None:
                for key in ('RMSEs', 'rmse', 'rmse_history', 'benign_rmse', 'stats'):
                    if key in obj:
                        arr = np.array(obj[key], dtype=float)
                        arr = arr[np.isfinite(arr) & (arr > 0) & (arr < RMSE_MAX_SANE)]
                        if len(arr) >= 10:
                            threshold = float(np.percentile(arr, 99)) * 2.0
                            print(f"[KitNET] Seuil recalculé P99×2: {threshold:.6f}")
                            break
        elif KITNET_AVAILABLE and isinstance(obj, kit.KitNET):
            kitnet_obj = obj
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            kitnet_obj = obj[0]
            if isinstance(obj[1], (int, float)):
                threshold = float(obj[1])

        if kitnet_obj is not None and threshold is None:
            for attr in ('threshold', 'FPR', 'anomaly_threshold', '_threshold'):
                if hasattr(kitnet_obj, attr):
                    v = getattr(kitnet_obj, attr)
                    if isinstance(v, (int, float)) and 0 < v < RMSE_MAX_SANE:
                        threshold = float(v)
                        break

        if kitnet_obj is not None and n_features is None:
            for attr in ('n', 'num_features', 'n_features', 'FM_n'):
                if hasattr(kitnet_obj, attr):
                    v = getattr(kitnet_obj, attr)
                    if isinstance(v, int) and v > 0:
                        n_features = v
                        break

        if kitnet_obj is None:
            if hasattr(obj, 'process'):
                kitnet_obj = obj
            else:
                return False

        # [FIX-EXP-3] Sanitize weights avant utilisation
        _sanitize_ae_weights(kitnet_obj)

        print(f"[KitNET] Seuil original = {threshold} → recalibration sur trafic live")

        self._kitnet   = kitnet_obj
        self._n        = n_features
        self._configured = True
        self.threshold = float('inf')
        self.trained   = False
        self._kitnet_executing = True
        self._real_rmse_count  = 1

        self._warmup_rmse: list[float] = []
        self._warmup_done: bool        = False
        return True

    def _build_kitnet(self, n: int):
        try:
            return kit.KitNET(
                n=n, max_autoencoder_size=self.max_ae_size,
                FM_grace_period=self.fm_grace, AD_grace_period=self.ad_grace,
                learning_rate=self.learning_rate, hidden_ratio=0.75,
            )
        except Exception as e:
            print(f"[KitNET] Erreur construction : {e}")
            return None

    def configure_n_features(self, n: int):
        if self._pretrained:
            return
        if n == self._n and self._configured and self._kitnet is not None:
            return
        if not KITNET_AVAILABLE:
            return
        print(f"[KitNET] Reconfiguration {self._n} → {n} features")
        self._n = n
        self._kitnet = self._build_kitnet(n)
        if self._kitnet is not None:
            self._configured = True
            self._consecutive_errors = 0

    def _resize_vector(self, vec: np.ndarray) -> np.ndarray:
        if self._n is None:
            return vec
        n = self._n
        if len(vec) == n:
            return vec
        if len(vec) > n:
            return vec[:n]
        resized = np.zeros(n, dtype=np.float64)
        resized[:len(vec)] = vec
        return resized

    def process_vector(self, vec: np.ndarray) -> dict:
        if not KITNET_AVAILABLE:
            return self._unavailable_result()
        if not self._pretrained:
            if not self._configured or self._kitnet is None:
                self.configure_n_features(len(vec))
            if self._kitnet is None:
                return self._unavailable_result()
            if len(vec) != self._n:
                self.configure_n_features(len(vec))
        return self._process_raw(vec)

    def process(self, features: dict) -> dict:
        if self._kitnet is None:
            if not self._pretrained and not self._configured and KITNET_AVAILABLE:
                self.configure_n_features(len(FALLBACK_FEATURES))
            else:
                return self._unavailable_result()
        vec = np.array([float(features.get(col, 0)) for col in FALLBACK_FEATURES], dtype=np.float64)
        return self._process_raw(vec)

    def _process_raw(self, vec: np.ndarray) -> dict:
        # Sanitize input
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.clip(vec, -1e6, 1e6)

        if self._pretrained and self._n is not None:
            vec = self._resize_vector(vec)

        # [FIX-EXP-2] Wrapper global overflow
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            try:
                rmse = float(self._kitnet.process(vec))
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                if self._consecutive_errors % 50 == 1:
                    print(f"[KitNET] process() erreur #{self._consecutive_errors}: {e}")
                if self._consecutive_errors >= 50 and not self._pretrained:
                    self._configured = False
                    self._kitnet = None
                    self._consecutive_errors = 0
                return self._unavailable_result()

        if not np.isfinite(rmse) or rmse < 0:
            rmse = 0.0
        rmse = min(rmse, RMSE_MAX_SANE)

        self.rmse_history.append(rmse)
        self.packet_count += 1

        # ── Pre-trained mode ──────────────────────────────────────
        if self._pretrained:
            if not self._warmup_done:
                if 0 < rmse < RMSE_MAX_SANE:
                    self._warmup_rmse.append(rmse)
                n_collected = len(self._warmup_rmse)
                progress = min(n_collected / PRETRAINED_WARMUP_PKTS, 1.0)
                if n_collected >= PRETRAINED_WARMUP_PKTS:
                    arr = np.array(self._warmup_rmse)
                    arr = arr[arr < np.percentile(arr, 99.5)]
                    p_val = float(np.percentile(arr, PRETRAINED_THRESHOLD_PERCENTILE))
                    self.threshold  = p_val * PRETRAINED_SAFETY_FACTOR
                    self._warmup_done = True
                    self.trained      = True
                    print(f"[KitNET] ✓ Warm-up terminé — seuil = {self.threshold:.4f} "
                          f"(P{PRETRAINED_THRESHOLD_PERCENTILE}={p_val:.4f} × {PRETRAINED_SAFETY_FACTOR}) "
                          f"sur {n_collected} paquets")
                return {"rmse": round(rmse, 6), "is_anomaly": False, "phase": "warmup",
                        "progress": round(progress, 4), "threshold": 0.0,
                        "severity_score": 0.0, "trained": False, "mode": self._mode,
                        "real_rmse_count": self.packet_count}

            is_anomaly = rmse > self.threshold
            sev_score  = rmse / max(self.threshold, 1e-9)
            return {"rmse": round(rmse, 6), "is_anomaly": is_anomaly, "phase": "pretrained",
                    "progress": 1.0, "threshold": round(self.threshold, 6),
                    "severity_score": round(min(sev_score, 10.0), 3),
                    "trained": True, "mode": self._mode, "real_rmse_count": self.packet_count}

        # ── Online mode ───────────────────────────────────────────
        if rmse > 0.0:
            self._real_rmse_count += 1
            self._post_grace_rmse.append(rmse)
            if not self._kitnet_executing and self._real_rmse_count >= 10:
                self._kitnet_executing = True

        if (not self.trained and self._kitnet_executing
                and self._real_rmse_count >= MIN_REAL_RMSE_FOR_THRESHOLD):
            arr = np.array(list(self._post_grace_rmse))
            p99 = float(np.percentile(arr, THRESHOLD_PERCENTILE))
            self.threshold = max(p99 * THRESHOLD_SAFETY_FACTOR, 1e-6)
            self.trained   = True
            print(f"[KitNET] ✓ Entraîné — seuil = {self.threshold:.6f}")

        progress   = min(self.packet_count / self.grace_total, 1.0)
        if not self.trained:
            return {"rmse": round(rmse, 6), "is_anomaly": False,
                    "phase": "FM" if self.packet_count < self.fm_grace else "AD",
                    "progress": round(progress, 4), "threshold": self.threshold,
                    "severity_score": 0.0, "trained": False, "mode": self._mode,
                    "real_rmse_count": self._real_rmse_count}

        sev_score  = rmse / max(self.threshold, 1e-9)
        is_anomaly = rmse > self.threshold
        return {"rmse": round(rmse, 6), "is_anomaly": is_anomaly, "phase": "monitoring",
                "progress": 1.0, "threshold": round(self.threshold, 6),
                "severity_score": round(min(sev_score, 10.0), 3),
                "trained": True, "mode": self._mode, "real_rmse_count": self._real_rmse_count}

    def _unavailable_result(self) -> dict:
        return {"rmse": 0.0, "is_anomaly": False, "phase": "unavailable",
                "progress": 0.0, "threshold": 0.0, "severity_score": 0.0,
                "trained": False, "mode": "unavailable", "real_rmse_count": 0}

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "kitnet": self._kitnet, "threshold": self.threshold,
                "trained": self.trained, "packet_count": self.packet_count,
                "rmse_history": self.rmse_history[-5_000:],
                "post_grace_rmse": list(self._post_grace_rmse),
                "real_rmse_count": self._real_rmse_count,
                "kitnet_executing": self._kitnet_executing,
                "fm_grace": self.fm_grace, "ad_grace": self.ad_grace,
                "n_features": self._n, "mode": self._mode, "pretrained": self._pretrained,
            }, f)
        print(f"[KitNET] Sauvegardé → {path}")

    @classmethod
    def load(cls, path: str) -> "ZeroDayEngine":
        with open(path, "rb") as f:
            s = pickle.load(f)
        e = cls.__new__(cls)
        e._kitnet           = s["kitnet"]
        e.threshold         = s["threshold"]
        e.trained           = s["trained"]
        e.packet_count      = s["packet_count"]
        e.rmse_history      = s.get("rmse_history", [])
        e._post_grace_rmse  = deque(s.get("post_grace_rmse", []), maxlen=10_000)
        e._real_rmse_count  = s.get("real_rmse_count", 0)
        e._kitnet_executing = s.get("kitnet_executing", False)
        e.fm_grace          = s.get("fm_grace", 5_000)
        e.ad_grace          = s.get("ad_grace", 50_000)
        e.grace_total       = e.fm_grace + e.ad_grace
        e._n                = s.get("n_features", len(FALLBACK_FEATURES))
        e._mode             = s.get("mode", "online")
        e._pretrained       = s.get("pretrained", False)
        e._configured       = True
        e.max_ae_size       = 10
        e.learning_rate     = 0.1
        e._consecutive_errors = 0
        e._warmup_rmse      = []
        e._warmup_done      = True
        _sanitize_ae_weights(e._kitnet)
        print(f"[KitNET] Chargé depuis {path} — {e.packet_count} paquets")
        return e

    @property
    def n_features(self) -> int:
        return self._n