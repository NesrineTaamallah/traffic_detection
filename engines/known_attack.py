"""
engines/known_attack.py  — CORRIGÉ v4
=======================================
PROBLÈME IDENTIFIÉ dans les logs :
    Le modèle attend des features transformées (log_, dsport_freq, ct_ftp_cmd…)
    mais on lui envoyait les features UNSW-NB15 brutes.

SOLUTION :
    [FIX-1] _get_model_features() : lit dynamiquement les features attendues
             depuis les attributs feature_names_in_ / feature_names_ du modèle.
    [FIX-2] _engineer_features() : applique TOUTES les transformations
             connues du dataset UNSW-NB15 étendu (log_, freq_, is_, ct_…).
    [FIX-3] _preprocess() : construit le DataFrame avec exactement les colonnes
             du modèle, remplit les manquantes à 0, supprime les inconnues.
    [FIX-4] Logs détaillés au 1er appel pour diagnostiquer les features.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ── Features brutes UNSW-NB15 générées par capture.py ────────────
RAW_UNSW = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm',
]

HIGH_SKEW = [
    'sbytes', 'dbytes', 'Sload', 'Dload', 'Sjit',
    'smeansz', 'dmeansz', 'dur', 'res_bdy_len',
    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
]


def _get_model_features(model) -> list[str] | None:
    """Extrait la liste de features attendues depuis le modèle sklearn/XGBoost."""
    for attr in ["feature_names_in_", "feature_names_", "feature_name_"]:
        if hasattr(model, attr):
            return list(getattr(model, attr))
    # Pipeline
    if hasattr(model, "steps"):
        for _, step in model.steps:
            r = _get_model_features(step)
            if r:
                return r
    return None


def _engineer_features(raw: dict) -> dict:
    """
    Applique toutes les transformations de feature-engineering
    couramment appliquées sur UNSW-NB15 avant entraînement.
    Génère ~80+ colonnes à partir des 35 colonnes brutes.
    """
    f = {k: float(raw.get(k, 0)) for k in RAW_UNSW}

    # ── Log1p transforms (colonnes haute asymétrie) ───────────────
    for col in HIGH_SKEW:
        val = f.get(col, 0.0)
        f[f"log_{col}"] = float(np.log1p(max(val, 0)))

    # ── Ratios dérivés ────────────────────────────────────────────
    total_bytes = f['sbytes'] + f['dbytes']
    f['total_bytes']      = total_bytes
    f['log_total_bytes']  = float(np.log1p(total_bytes))
    f['bytes_ratio']      = f['sbytes'] / max(f['dbytes'], 1)
    f['pkts_ratio']       = f['Spkts'] / max(f['Dpkts'], 1)
    f['load_ratio']       = f['Sload'] / max(f['Dload'], 1)
    f['byte_per_pkt_s']   = f['sbytes'] / max(f['Spkts'], 1)
    f['byte_per_pkt_d']   = f['dbytes'] / max(f['Dpkts'], 1)
    f['total_pkts']       = f['Spkts'] + f['Dpkts']
    f['log_total_pkts']   = float(np.log1p(f['total_pkts']))
    f['pkt_size_diff']    = abs(f['smeansz'] - f['dmeansz'])
    f['jit_diff']         = abs(f['Sjit'] - f['Djit'])
    f['intpkt_diff']      = abs(f['Sintpkt'] - f['Dintpkt'])

    # ── Fréquences ports (proxy simple) ───────────────────────────
    sport = int(raw.get('_sport', 0))
    dport = int(raw.get('_dport', 0))
    f['sport']            = float(sport)
    f['dsport']           = float(dport)     # alias commun dans UNSW
    f['dport']            = float(dport)
    # Fréquences normalisées (sans vraie table de fréquence on met 0)
    f['sport_freq']       = 0.0
    f['dsport_freq']      = 0.0
    f['dport_freq']       = 0.0
    f['src_port_freq']    = 0.0
    f['dst_port_freq']    = 0.0

    # ── Flags binaires ────────────────────────────────────────────
    f['is_sm_ips_ports']  = 1.0 if sport == dport else 0.0
    f['is_ftp_login']     = 1.0 if dport in (21, 20) else 0.0
    f['ct_ftp_cmd']       = 0.0   # pas d'inspection payload en live
    f['ct_flw_http_mthd'] = f.get('ct_flw_http_mthd', 0.0)

    # ── Proto encoding ────────────────────────────────────────────
    proto_str = str(raw.get('_proto', '')).lower()
    proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
    f['proto_num']  = float(proto_map.get(proto_str, 0))
    f['proto_tcp']  = 1.0 if proto_str == 'tcp'  else 0.0
    f['proto_udp']  = 1.0 if proto_str == 'udp'  else 0.0
    f['proto_icmp'] = 1.0 if proto_str == 'icmp' else 0.0

    # ── TTL features ──────────────────────────────────────────────
    f['ttl_diff']   = abs(f['sttl'] - f['dttl'])
    f['ttl_ratio']  = f['sttl'] / max(f['dttl'], 1)

    # ── TCP handshake features ────────────────────────────────────
    f['synack_ratio'] = f['synack'] / max(f['tcprtt'], 1e-6)
    f['ackdat_ratio'] = f['ackdat'] / max(f['tcprtt'], 1e-6)

    # ── Durée features ────────────────────────────────────────────
    f['log_dur'] = float(np.log1p(max(f['dur'], 0)))
    f['inv_dur'] = 1.0 / max(f['dur'], 1e-6)

    # ── Service port categories ───────────────────────────────────
    web_ports   = {80, 443, 8080, 8443}
    db_ports    = {3306, 5432, 1433, 27017}
    ssh_ports   = {22}
    dns_ports   = {53}
    f['is_web_port']  = 1.0 if dport in web_ports   else 0.0
    f['is_db_port']   = 1.0 if dport in db_ports     else 0.0
    f['is_ssh_port']  = 1.0 if dport in ssh_ports    else 0.0
    f['is_dns_port']  = 1.0 if dport in dns_ports    else 0.0
    f['is_priv_port'] = 1.0 if dport < 1024          else 0.0

    # ── ct_ counters (on n'a pas l'historique complet, proxy = 0) ─
    for col in [
        'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst',
        'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    ]:
        f.setdefault(col, 0.0)

    # Nettoyage inf/nan
    for k, v in f.items():
        if not np.isfinite(v):
            f[k] = 0.0

    return f


class KnownAttackEngine:
    """
    Wrapper du pipeline XGBoost hiérarchique.
    [FIX-3] Adapte dynamiquement les features aux colonnes attendues par le modèle.
    """

    def __init__(self, models_dir: str = "./models"):
        base = Path(models_dir)
        self.binary     = joblib.load(base / "best_binary_model.pkl")
        self.multiclass = joblib.load(base / "xgb_hierarchical_multiclass.pkl")
        self.scaler     = joblib.load(base / "scaler_hierarchical.pkl")
        self.label_enc  = joblib.load(base / "label_encoder_hierarchical.pkl")
        self.pt = None
        pt_path = base / "powertransformer_hierarchical.pkl"
        if pt_path.exists():
            self.pt = joblib.load(pt_path)

        # [FIX-1] Lire les features attendues dynamiquement
        self._model_features: list[str] | None = _get_model_features(self.binary)
        if self._model_features is None:
            self._model_features = _get_model_features(self.scaler)

        self._first_call = True

        if self._model_features:
            print(f"[KnownAttack] {len(self._model_features)} features attendues par le modèle")
            print(f"[KnownAttack] Exemple features : {self._model_features[:8]}...")
        else:
            print("[KnownAttack] WARN : impossible de lire les features du modèle — fallback RAW_UNSW")

        print(f"[KnownAttack] Modèles chargés depuis {models_dir}")

    def _preprocess(self, raw: dict) -> pd.DataFrame:
        """
        [FIX-2/3] Pipeline :
          1. Engineer toutes les features dérivées
          2. Construire le DataFrame avec EXACTEMENT les colonnes du modèle
          3. Colonnes manquantes → 0, colonnes inconnues → ignorées
          4. Scaler + PowerTransformer optionnel
        """
        # Étape 1 : générer toutes les features
        enriched = _engineer_features(raw)

        # Étape 2 : colonnes cibles
        if self._model_features:
            cols = self._model_features
        else:
            cols = [c for c in RAW_UNSW if c in enriched]

        # Étape 3 : DataFrame aligné
        row = {col: enriched.get(col, 0.0) for col in cols}
        X = pd.DataFrame([row], columns=cols)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # Log de diagnostic (1er appel seulement)
        if self._first_call:
            self._first_call = False
            missing = [c for c in cols if c not in enriched]
            if missing:
                print(f"[KnownAttack] Features manquantes (→ 0) : {missing[:10]}{'...' if len(missing)>10 else ''}")
            else:
                print(f"[KnownAttack] ✓ Toutes les {len(cols)} features présentes")

        # Étape 4 : scaling
        try:
            X_sc = pd.DataFrame(self.scaler.transform(X), columns=cols)
        except Exception as e:
            print(f"[KnownAttack] Scaler error : {e} — utilisation sans scaling")
            X_sc = X

        # PowerTransformer optionnel
        if self.pt:
            hsk_cols = [c for c in HIGH_SKEW if c in X_sc.columns]
            log_cols  = [f"log_{c}" for c in HIGH_SKEW if f"log_{c}" in X_sc.columns]
            pt_cols   = [c for c in hsk_cols + log_cols if c in X_sc.columns]
            if pt_cols:
                try:
                    X_sc[pt_cols] = self.pt.transform(X_sc[pt_cols])
                except Exception:
                    pass

        return X_sc

    def predict(self, features: dict) -> dict:
        X = self._preprocess(features)

        try:
            is_attack  = int(self.binary.predict(X)[0])
            bin_proba  = float(self.binary.predict_proba(X)[0][1])
        except Exception as e:
            print(f"[KnownAttack] Binary predict error : {e}")
            return {"is_attack": False, "label": "Normal", "attack_type": None,
                    "confidence": 0.0, "bin_proba": 0.0}

        if not is_attack:
            return {
                "is_attack":   False,
                "label":       "Normal",
                "attack_type": None,
                "confidence":  round(1 - bin_proba, 4),
                "bin_proba":   round(bin_proba, 4),
            }

        try:
            class_idx   = self.multiclass.predict(X)[0]
            class_prob  = self.multiclass.predict_proba(X)[0]
            attack_type = self.label_enc.inverse_transform([class_idx])[0]
        except Exception as e:
            print(f"[KnownAttack] Multiclass predict error : {e}")
            return {
                "is_attack":   True,
                "label":       "Attack",
                "attack_type": "Unknown",
                "confidence":  round(bin_proba, 4),
                "bin_proba":   round(bin_proba, 4),
            }

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