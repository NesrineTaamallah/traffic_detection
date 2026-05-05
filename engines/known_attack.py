"""
engines/known_attack.py — CORRIGÉ v6
=======================================

FIXES v6:
[FIX-1] Ajout du one-hot encoding du champ 'state' UNSW-NB15
        (state_ACC, state_CLO, state_CON, state_ECO, state_ECR,
         state_FIN, state_INT, state_MAS, state_PAR, state_REQ,
         state_RST, state_TST, state_TXD, state_URH, state_URN)
        → résout les 31 features manquantes signalées dans les logs.

[FIX-2] Ajout de proto_freq, src_port_freq, dst_port_freq
        (compteurs de fréquence absents en live → 0.0, mais il
         faut que la colonne existe dans le DataFrame).

[FIX-3] log_ct_src_ ltm → alias correct 'ct_src_ ltm' (espace UNSW)

[FIX-4] Nettoyage : suppression des features redondantes /
        harmonisation avec inspect_models.py output.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ── Features brutes UNSW-NB15 ─────────────────────────────────────
RAW_UNSW = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm',
]

# Colonnes log1p
ALL_LOG_COLS = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'ct_src_ltm', 'sport', 'dport', 'dsport',
    'total_bytes', 'total_pkts', 'bytes_ratio', 'pkts_ratio',
    'load_ratio', 'byte_per_pkt_s', 'byte_per_pkt_d',
    'pkt_size_diff', 'jit_diff', 'intpkt_diff',
]

# [FIX-1] Tous les états UNSW-NB15 connus
UNSW_STATES = [
    'ACC', 'CLO', 'CON', 'ECO', 'ECR', 'FIN', 'INT',
    'MAS', 'PAR', 'REQ', 'RST', 'TST', 'TXD', 'URH', 'URN',
]


def _get_model_features(model) -> list[str] | None:
    for attr in ["feature_names_in_", "feature_names_", "feature_name_"]:
        if hasattr(model, attr):
            return list(getattr(model, attr))
    if hasattr(model, "steps"):
        for _, step in model.steps:
            r = _get_model_features(step)
            if r:
                return r
    return None


def _infer_state_from_flow(raw: dict) -> str:
    """
    Tente de déduire le state UNSW-NB15 depuis les champs TCP disponibles.
    En live, on n'a pas le champ 'state' natif → heuristique basée sur
    les flags TCP observés pendant la capture du flux.
    """
    proto = str(raw.get('_proto', '')).upper()
    if proto == 'ICMP':
        return 'ECO'
    if proto == 'UDP':
        return 'CON'
    # TCP
    fin_seen = raw.get('_fin_seen', False)
    syn_seen = raw.get('_syn_seen', False)
    ack_seen = raw.get('_ack_seen', False)
    rst_seen = raw.get('_rst_seen', False)
    if rst_seen:
        return 'RST'
    if fin_seen:
        return 'FIN'
    if syn_seen and ack_seen:
        return 'CON'
    if syn_seen:
        return 'INT'
    # Fallback
    return 'CON'


def _engineer_features(raw: dict) -> dict:
    """
    Applique toutes les transformations de feature-engineering UNSW-NB15.

    [FIX-1] Ajout des one-hot state_XXX
    [FIX-2] Ajout proto_freq, src_port_freq, dst_port_freq
    [FIX-3] Alias ct_src_ ltm (avec espace) → ct_src_ltm
    """
    f = {}
    for k in RAW_UNSW:
        val = raw.get(k, 0)
        f[k] = float(val) if val is not None else 0.0

    # Alias sans espace (typo UNSW-NB15)
    f['ct_src_ltm'] = f.get('ct_src_ ltm', 0.0)

    # ── [FIX-1] State one-hot encoding ───────────────────────────
    state_val = str(raw.get('state', '')).upper().strip()
    if not state_val:
        state_val = _infer_state_from_flow(raw)
    for s in UNSW_STATES:
        f[f'state_{s}'] = 1.0 if state_val == s else 0.0

    # ── Log1p sur toutes les colonnes numériques ──────────────────
    for col in ALL_LOG_COLS:
        val = f.get(col, 0.0)
        f[f"log_{col}"] = float(np.log1p(max(val, 0)))

    # [FIX-3] log pour ct_src_ ltm (espace) → log_ct_src_ ltm
    f['log_ct_src_ ltm'] = float(np.log1p(max(f.get('ct_src_ ltm', 0.0), 0)))

    # ── Ratios dérivés ────────────────────────────────────────────
    total_bytes = f['sbytes'] + f['dbytes']
    total_pkts  = f['Spkts']  + f['Dpkts']
    f['total_bytes']      = total_bytes
    f['total_pkts']       = total_pkts
    f['log_total_bytes']  = float(np.log1p(total_bytes))
    f['log_total_pkts']   = float(np.log1p(total_pkts))
    f['bytes_ratio']      = f['sbytes'] / max(f['dbytes'], 1)
    f['pkts_ratio']       = f['Spkts']  / max(f['Dpkts'],  1)
    f['load_ratio']       = f['Sload']  / max(f['Dload'],  1)
    f['byte_per_pkt_s']   = f['sbytes'] / max(f['Spkts'],  1)
    f['byte_per_pkt_d']   = f['dbytes'] / max(f['Dpkts'],  1)
    f['pkt_size_diff']    = abs(f['smeansz']  - f['dmeansz'])
    f['jit_diff']         = abs(f['Sjit']     - f['Djit'])
    f['intpkt_diff']      = abs(f['Sintpkt']  - f['Dintpkt'])

    for col in ['bytes_ratio', 'pkts_ratio', 'load_ratio',
                'byte_per_pkt_s', 'byte_per_pkt_d',
                'pkt_size_diff', 'jit_diff', 'intpkt_diff']:
        f[f"log_{col}"] = float(np.log1p(max(f[col], 0)))

    # ── Ports ─────────────────────────────────────────────────────
    sport = int(raw.get('_sport', 0))
    dport = int(raw.get('_dport', 0))
    f['sport']            = float(sport)
    f['dsport']           = float(dport)
    f['dport']            = float(dport)
    f['log_sport']        = float(np.log1p(sport))
    f['log_dsport']       = float(np.log1p(dport))
    f['log_dport']        = float(np.log1p(dport))

    # [FIX-2] Fréquences de ports — inconnus en live, on met 0
    f['sport_freq']       = 0.0
    f['dsport_freq']      = 0.0
    f['dport_freq']       = 0.0
    f['src_port_freq']    = 0.0
    f['dst_port_freq']    = 0.0
    # [FIX-2] proto_freq
    f['proto_freq']       = 0.0

    # ── Flags binaires ────────────────────────────────────────────
    f['is_sm_ips_ports']  = 1.0 if sport == dport else 0.0
    f['is_ftp_login']     = 1.0 if dport in (21, 20) else 0.0
    f['ct_ftp_cmd']       = 0.0

    # ── Proto encoding ────────────────────────────────────────────
    proto_str = str(raw.get('_proto', '')).lower()
    proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
    f['proto_num']  = float(proto_map.get(proto_str, 0))
    f['proto_tcp']  = 1.0 if proto_str == 'tcp'  else 0.0
    f['proto_udp']  = 1.0 if proto_str == 'udp'  else 0.0
    f['proto_icmp'] = 1.0 if proto_str == 'icmp' else 0.0

    # ── TTL features ──────────────────────────────────────────────
    f['ttl_diff']     = abs(f['sttl'] - f['dttl'])
    f['ttl_ratio']    = f['sttl'] / max(f['dttl'], 1)
    f['log_ttl_diff'] = float(np.log1p(f['ttl_diff']))

    # ── TCP handshake ─────────────────────────────────────────────
    f['synack_ratio'] = f['synack'] / max(f['tcprtt'], 1e-6)
    f['ackdat_ratio'] = f['ackdat'] / max(f['tcprtt'], 1e-6)
    f['log_dur']      = float(np.log1p(max(f['dur'], 0)))
    f['inv_dur']      = 1.0 / max(f['dur'], 1e-6)

    # ── Service port categories ───────────────────────────────────
    web_ports = {80, 443, 8080, 8443}
    db_ports  = {3306, 5432, 1433, 27017}
    f['is_web_port']  = 1.0 if dport in web_ports else 0.0
    f['is_db_port']   = 1.0 if dport in db_ports  else 0.0
    f['is_ssh_port']  = 1.0 if dport == 22         else 0.0
    f['is_dns_port']  = 1.0 if dport == 53         else 0.0
    f['is_priv_port'] = 1.0 if dport < 1024        else 0.0

    # ── ct_ counters (live → 0) ───────────────────────────────────
    for col in [
        'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst',
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd',
    ]:
        f.setdefault(col, 0.0)

    # ── Nettoyage inf/nan ─────────────────────────────────────────
    for k, v in f.items():
        if isinstance(v, float) and not np.isfinite(v):
            f[k] = 0.0

    return f


class KnownAttackEngine:
    """
    Wrapper du pipeline XGBoost hiérarchique UNSW-NB15.
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

        self._model_features: list[str] | None = _get_model_features(self.binary)
        if self._model_features is None:
            self._model_features = _get_model_features(self.scaler)

        self._first_call = True

        if self._model_features:
            print(f"[KnownAttack] {len(self._model_features)} features attendues par le modèle")
            print(f"[KnownAttack] Exemple features : {self._model_features[:8]}...")
        else:
            print("[KnownAttack] WARN : impossible de lire les features du modèle")

        print(f"[KnownAttack] Modèles chargés depuis {models_dir}")

    def _preprocess(self, raw: dict) -> pd.DataFrame:
        enriched = _engineer_features(raw)

        cols = self._model_features if self._model_features else \
               [c for c in RAW_UNSW if c in enriched]

        row = {col: enriched.get(col, 0.0) for col in cols}
        X = pd.DataFrame([row], columns=cols)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        if self._first_call:
            self._first_call = False
            missing = [c for c in cols if c not in enriched]
            if missing:
                print(f"[KnownAttack] ⚠ Features manquantes (→ 0) [{len(missing)}] : "
                      f"{missing[:15]}{'...' if len(missing) > 15 else ''}")
            else:
                print(f"[KnownAttack] ✓ Toutes les {len(cols)} features présentes")
            print(f"[KnownAttack] Features générées total : {len(enriched)}")

        try:
            X_sc = pd.DataFrame(self.scaler.transform(X), columns=cols)
        except Exception as e:
            print(f"[KnownAttack] Scaler error : {e}")
            X_sc = X

        if self.pt:
            pt_cols = [c for c in X_sc.columns
                       if c in [f"log_{k}" for k in ALL_LOG_COLS]]
            if pt_cols:
                try:
                    X_sc[pt_cols] = self.pt.transform(X_sc[pt_cols])
                except Exception:
                    pass

        return X_sc

    def predict(self, features: dict) -> dict:
        X = self._preprocess(features)

        try:
            is_attack = int(self.binary.predict(X)[0])
            bin_proba = float(self.binary.predict_proba(X)[0][1])
        except Exception as e:
            print(f"[KnownAttack] Binary predict error : {e}")
            return {
                "is_attack": False, "label": "Normal",
                "attack_type": None, "confidence": 0.0, "bin_proba": 0.0,
            }

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