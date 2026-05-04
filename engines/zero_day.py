"""
engines/zero_day.py — CORRIGÉ v4 (FINAL)
==========================================

DIAGNOSTIC DES BUGS (pourquoi tout = Zero-day) :

BUG CRITIQUE #1 — Seuil automatique = 0.000000
────────────────────────────────────────────────
Log : "[KitNET] ✓ Entraîné — seuil automatique = 0.000000"

Cause : Le threshold est calculé comme mean + 3*std des RMSE de la
phase d'entraînement. Pendant la grace period (FM+AD = 55 000 paquets),
KitNET retourne 0.0 à chaque appel (comportement officiel Kitsune-py :
"returns zero during the FM feature-mapping and AD grace periods").
Donc window = [0, 0, 0, ...] → mean=0, std=0 → threshold=0.

Résultat : TOUT paquet avec rmse > 0.0 (même 0.00001) déclenche
is_anomaly=True → tout le trafic normal devient "Zero-day".

FIX : Ne collecter que les RMSE POST-grace (phase monitoring) pour
calculer le seuil. Attendre d'avoir ≥ 500 vrais RMSE > 0.

BUG CRITIQUE #2 — Double comptage grace_total
──────────────────────────────────────────────
packet_count est incrémenté dans _process_raw() à chaque paquet.
Mais KitNET gère en interne ses propres compteurs FM/AD.
On compare packet_count == grace_total pour déclencher le calcul
du seuil, mais rmse_history contient encore tous les 0.0 de la
grace period. Quand grace_total est atteint, le dernier rmse dans
l'historique vient juste de passer en mode "execute" → window
contient 10 000 zéros + quelques vrais RMSE. Résultat : threshold ≈ 0.

FIX : Séparer rmse_history (tous) de post_grace_rmse (seulement
ceux reçus APRÈS que KitNET soit en mode execute, détecté par rmse > 0
de façon stable).

BUG #3 — known_attack.py : 31 features manquantes
──────────────────────────────────────────────────
Le modèle attend des features OHE de 'state' et 'service'
(state_ACC, state_CLO, state_CON..., service_http, service_ftp...)
et une colonne 'proto_freq' (frequency encoding).
En live, ces colonnes ne peuvent pas être calculées sans le dataset
d'entraînement pour les fréquences ni le context de flux réseau.
→ Elles tombent toutes à 0, ce qui biaise fortement la classification.

FIX appliqué dans known_attack.py : valeurs par défaut réalistes +
warn supprimé (elles seront toujours 0 en live, c'est attendu).

BUG #4 — Fusion trop agressive (decision.py)
─────────────────────────────────────────────
Quand threshold=0, severity_score = rmse / max(0, 1e-9) = rmse * 1e9.
Donc severity_score ≥ 2.5 → CRITICAL pour TOUT paquet post-grace.
Et is_anomaly=True partout → toutes les alertes sont déclenchées.

FIX : Corriger le threshold en premier (BUG#1), et ajouter un guard
dans DecisionFusion pour ignorer les alertes si threshold trop bas.
"""

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

import sys
import pickle
from pathlib import Path
from collections import deque

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

# Nombre minimum de RMSE réels (> 0) avant de calculer le seuil
MIN_REAL_RMSE_FOR_THRESHOLD = 500
# Percentile pour le seuil (99e percentile = seulement 1% du trafic normal est anomalie)
THRESHOLD_PERCENTILE = 99
# Facteur multiplicatif de sécurité sur le seuil
THRESHOLD_SAFETY_FACTOR = 1.5


class ZeroDayEngine:
    """
    Détection d'anomalies via KitNET.

    CORRECTION PRINCIPALE : Le seuil est calculé sur les RMSE POST-grace
    uniquement (quand KitNET est en mode execute), évitant le bug
    threshold=0 causé par les zéros de la grace period.
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

        # Historique TOUS les RMSE (incluant les 0 de la grace period)
        self.rmse_history: list[float] = []

        # [FIX BUG#1] Historique UNIQUEMENT des RMSE réels post-grace (> 0)
        # Utilisé pour calculer le seuil. Taille limitée aux 10k derniers.
        self._post_grace_rmse: deque = deque(maxlen=10_000)

        self.threshold:    float = 0.1   # valeur initiale non-nulle
        self.packet_count: int   = 0
        self.trained:      bool  = False

        # [FIX BUG#1] Détection du passage en mode execute de KitNET
        # KitNET retourne 0.0 pendant FM et AD, puis des valeurs > 0 après.
        # On détecte le début du mode execute par une séquence de RMSE > 0.
        self._kitnet_executing: bool = False
        self._real_rmse_count:  int  = 0   # nombre de RMSE > 0 consécutifs/accumulés

        self._mode = "afterimage" if KITNET_AVAILABLE else "unavailable"
        self._consecutive_errors: int  = 0

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

    def configure_n_features(self, n: int):
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
        if not KITNET_AVAILABLE:
            return self._unavailable_result()
        if not self._configured or self._kitnet is None:
            self.configure_n_features(len(vec))
        if self._kitnet is None:
            return self._unavailable_result()
        if len(vec) != self._n:
            self.configure_n_features(len(vec))
        return self._process_raw(vec)

    def process(self, features: dict) -> dict:
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
            self._consecutive_errors = 0
        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors == 1 or self._consecutive_errors % 100 == 0:
                print(f"[KitNET] process() error #{self._consecutive_errors} : {e}")
            if self._consecutive_errors >= 50:
                print(f"[KitNET] Trop d'erreurs ({self._consecutive_errors}), reconstruction...")
                self._configured = False
                self._kitnet = None
                self._consecutive_errors = 0
            return self._unavailable_result()

        self.rmse_history.append(rmse)
        self.packet_count += 1

        # ── [FIX BUG#1] Détection du mode execute de KitNET ──────────
        # KitNET retourne exactement 0.0 pendant FM et AD grace periods.
        # En mode execute, il retourne des RMSE > 0 (même très petits).
        # On détecte la transition par l'apparition de valeurs non-nulles.
        if rmse > 0.0:
            self._real_rmse_count += 1
            self._post_grace_rmse.append(rmse)
            if not self._kitnet_executing and self._real_rmse_count >= 10:
                # KitNET vient de passer en mode execute
                self._kitnet_executing = True
                print(f"[KitNET] Mode execute détecté après {self.packet_count} paquets")

        # ── Calcul du seuil (uniquement sur les RMSE post-grace réels) ─
        # [FIX BUG#1] On attend d'avoir au moins MIN_REAL_RMSE vrais RMSE
        # avant de calculer le seuil — évite threshold=0
        if (not self.trained
                and self._kitnet_executing
                and self._real_rmse_count >= MIN_REAL_RMSE_FOR_THRESHOLD):

            arr = np.array(list(self._post_grace_rmse))
            # Percentile 99 = seuil au-dessus duquel 1% du trafic "normal"
            # (vu pendant l'apprentissage) est considéré anomalie
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
                "rmse":           round(rmse, 6),
                "is_anomaly":     False,
                "phase":          "FM" if self.packet_count < self.fm_grace else "AD",
                "progress":       round(progress, 4),
                "threshold":      self.threshold,
                "severity_score": 0.0,
                "trained":        False,
                "mode":           self._mode,
                "real_rmse_count": self._real_rmse_count,
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
            "real_rmse_count": self._real_rmse_count,
        }

    def _unavailable_result(self) -> dict:
        return {
            "rmse": 0.0, "is_anomaly": False,
            "phase": "unavailable", "progress": 0.0,
            "threshold": 0.0, "severity_score": 0.0,
            "trained": False, "mode": "unavailable",
            "real_rmse_count": 0,
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "kitnet":              self._kitnet,
                "threshold":           self.threshold,
                "trained":             self.trained,
                "packet_count":        self.packet_count,
                "rmse_history":        self.rmse_history[-5_000:],
                "post_grace_rmse":     list(self._post_grace_rmse),
                "real_rmse_count":     self._real_rmse_count,
                "kitnet_executing":    self._kitnet_executing,
                "fm_grace":            self.fm_grace,
                "ad_grace":            self.ad_grace,
                "n_features":          self._n,
                "mode":                self._mode,
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
        e._mode                 = s.get("mode", "afterimage")
        e._configured           = True
        e.max_ae_size           = 10
        e.learning_rate         = 0.1
        e._consecutive_errors   = 0
        print(f"[KitNET] Chargé depuis {path} — {e.packet_count} paquets")
        return e

    @property
    def n_features(self) -> int:
        return self._n