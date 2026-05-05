"""
engines/known_attack.py — v7 FINAL
====================================

FIXES v7 vs v6 :
[FIX-1] Ajout de state_no (état numérique UNSW-NB15 → ct_state_ttl heuristique)
[FIX-2] Ajout de TOUS les service_* one-hot attendus par le modèle :
        service_Unknown, service_dhcp, service_dns, service_ftp,
        service_ftp-data, service_http, service_irc, service_pop3,
        service_radius, service_smtp, service_snmp, service_ssh, service_ssl
[FIX-3] Suppression du RuntimeWarning overflow exp() dans la sigmoid du scaler
        via np.errstate(over='ignore') + np.clip avant transformation
[FIX-4] Robustesse : si le modèle attend des colonnes inconnues → 0.0 par défaut
        sans crash (était déjà le cas, mais maintenant loggé clairement)
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

# [FIX-1] États UNSW-NB15 → encodage numérique ct_state_ttl heuristique
UNSW_STATES = [
    'ACC', 'CLO', 'CON', 'ECO', 'ECR', 'FIN', 'INT',
    'MAS', 'PAR', 'REQ', 'RST', 'TST', 'TXD', 'URH', 'URN',
]

# Mapping état → valeur numérique ct_state_ttl (approximation UNSW-NB15)
_STATE_TO_CT = {
    'FIN': 1, 'INT': 2, 'CON': 3, 'ECO': 4, 'ECR': 4,
    'RST': 5, 'ACC': 6, 'CLO': 7, 'REQ': 8, 'URN': 9,
    'URH': 10, 'MAS': 11, 'TST': 12, 'TXD': 13, 'PAR': 14,
}

# [FIX-2] Services UNSW-NB15 complets (14 services connus par les modèles)
UNSW_SERVICES = [
    'Unknown', 'dhcp', 'dns', 'ftp', 'ftp-data',
    'http', 'irc', 'pop3', 'radius', 'smtp', 'snmp', 'ssh', 'ssl',
]

# Mapping port → service UNSW-NB15
_PORT_TO_SERVICE = {
    21: 'ftp', 20: 'ftp-data', 22: 'ssh', 25: 'smtp',
    53: 'dns', 67: 'dhcp', 68: 'dhcp', 80: 'http',
    110: 'pop3', 161: 'snmp', 194: 'irc', 443: 'ssl',
    8080: 'http', 8443: 'ssl', 1812: 'radius', 1813: 'radius',
}


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
    proto = str(raw.get('_proto', '')).upper()
    if proto == 'ICMP':
        return 'ECO'
    if proto == 'UDP':
        return 'CON'
    if raw.get('_rst_seen', False):
        return 'RST'
    if raw.get('_fin_seen', False):
        return 'FIN'
    if raw.get('_syn_seen', False) and raw.get('_ack_seen', False):
        return 'CON'
    if raw.get('_syn_seen', False):
        return 'INT'
    return 'CON'


def _infer_service_from_port(dport: int, proto: str) -> str:
    svc = _PORT_TO_SERVICE.get(int(dport), None)
    if svc:
        return svc
    if str(proto).upper() == 'UDP':
        return 'Unknown'
    return 'Unknown'


def _engineer_features(raw: dict) -> dict:
    """
    Applique toutes les transformations de feature-engineering UNSW-NB15.
    [FIX-1] Ajout state_no + one-hot state_XXX
    [FIX-2] Ajout service_XXX one-hot complets
    """
    f = {}
    for k in RAW_UNSW:
        val = raw.get(k, 0)
        f[k] = float(val) if val is not None else 0.0

    # Alias ct_src_ ltm (avec espace)
    f['ct_src_ltm'] = f.get('ct_src_ ltm', 0.0)

    # ── [FIX-1] State encoding ────────────────────────────────────
    state_val = str(raw.get('state', '')).upper().strip()
    if not state_val or state_val in ('', 'NAN', 'NONE'):
        state_val = _infer_state_from_flow(raw)

    # One-hot states
    for s in UNSW_STATES:
        f[f'state_{s}'] = 1.0 if state_val == s else 0.0

    # [FIX-1] state_no : valeur numérique encodée ct_state_ttl
    f['state_no'] = float(_STATE_TO_CT.get(state_val, 0))

    # Mise à jour ct_state_ttl si pas déjà renseigné
    if f['ct_state_ttl'] == 0.0 and f['state_no'] > 0:
        f['ct_state_ttl'] = f['state_no']

    # ── [FIX-2] Service one-hot ───────────────────────────────────
    dport   = int(raw.get('_dport', 0))
    proto_s = str(raw.get('_proto', '')).upper()
    service_raw = str(raw.get('service', '')).strip().lower()

    # Priorité : champ 'service' explicite > inférence port
    if service_raw and service_raw not in ('', 'nan', 'none', '-'):
        detected_svc = service_raw
    else:
        detected_svc = _infer_service_from_port(dport, proto_s).lower()

    for svc in UNSW_SERVICES:
        f[f'service_{svc}'] = 1.0 if detected_svc == svc.lower() else 0.0

    # ── Log1p transformations ─────────────────────────────────────
    for col in ALL_LOG_COLS:
        val = f.get(col, 0.0)
        f[f"log_{col}"] = float(np.log1p(max(val, 0)))

    f['log_ct_src_ ltm'] = float(np.log1p(max(f.get('ct_src_ ltm', 0.0), 0)))

    # ── Ratios dérivés ────────────────────────────────────────────
    total_bytes = f['sbytes'] + f['dbytes']
    total_pkts  = f['Spkts']  + f['Dpkts']
    f['total_bytes']     = total_bytes
    f['total_pkts']      = total_pkts
    f['log_total_bytes'] = float(np.log1p(total_bytes))
    f['log_total_pkts']  = float(np.log1p(total_pkts))
    f['bytes_ratio']     = f['sbytes'] / max(f['dbytes'], 1)
    f['pkts_ratio']      = f['Spkts']  / max(f['Dpkts'],  1)
    f['load_ratio']      = f['Sload']  / max(f['Dload'],  1)
    f['byte_per_pkt_s']  = f['sbytes'] / max(f['Spkts'],  1)
    f['byte_per_pkt_d']  = f['dbytes'] / max(f['Dpkts'],  1)
    f['pkt_size_diff']   = abs(f['smeansz']  - f['dmeansz'])
    f['jit_diff']        = abs(f['Sjit']     - f['Djit'])
    f['intpkt_diff']     = abs(f['Sintpkt']  - f['Dintpkt'])

    for col in ['bytes_ratio', 'pkts_ratio', 'load_ratio',
                'byte_per_pkt_s', 'byte_per_pkt_d',
                'pkt_size_diff', 'jit_diff', 'intpkt_diff']:
        f[f"log_{col}"] = float(np.log1p(max(f[col], 0)))

    # ── Ports ─────────────────────────────────────────────────────
    sport = int(raw.get('_sport', 0))
    f['sport']     = float(sport)
    f['dsport']    = float(dport)
    f['dport']     = float(dport)
    f['log_sport'] = float(np.log1p(sport))
    f['log_dsport']= float(np.log1p(dport))
    f['log_dport'] = float(np.log1p(dport))

    # Fréquences inconnues en live → 0
    for col in ['sport_freq', 'dsport_freq', 'dport_freq', 'src_port_freq', 'dst_port_freq', 'proto_freq']:
        f[col] = 0.0

    # ── Flags binaires ────────────────────────────────────────────
    f['is_sm_ips_ports'] = 1.0 if sport == dport else 0.0
    f['is_ftp_login']    = 1.0 if dport in (21, 20) else 0.0
    f['ct_ftp_cmd']      = 0.0

    # ── Proto encoding ────────────────────────────────────────────
    proto_str = proto_s.lower()
    proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
    f['proto_num']  = float(proto_map.get(proto_str, 0))
    f['proto_tcp']  = 1.0 if proto_str == 'tcp'  else 0.0
    f['proto_udp']  = 1.0 if proto_str == 'udp'  else 0.0
    f['proto_icmp'] = 1.0 if proto_str == 'icmp' else 0.0

    # ── TTL ───────────────────────────────────────────────────────
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
    for col in ['ct_state_ttl', 'ct_srv_src', 'ct_srv_dst',
                'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd']:
        f.setdefault(col, 0.0)

    # ── Nettoyage inf/nan ─────────────────────────────────────────
    for k, v in f.items():
        if isinstance(v, float) and not np.isfinite(v):
            f[k] = 0.0

    return f


class KnownAttackEngine:
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
            print(f"[KnownAttack] Exemple : {self._model_features[:8]}...")
        else:
            print("[KnownAttack] WARN : impossible de lire les features du modèle")

        print(f"[KnownAttack] Modèles chargés depuis {models_dir}")
        print(f"[KnownAttack] Services supportés : {UNSW_SERVICES}")
        print(f"[KnownAttack] États TCP supportés : {UNSW_STATES}")

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
                print(f"[KnownAttack] ⚠ Features encore manquantes [{len(missing)}] : {missing[:10]}")
            else:
                print(f"[KnownAttack] ✓ Toutes les {len(cols)} features présentes")

        # [FIX-3] Overflow exp() : cliper les valeurs avant scaler
        X = X.clip(-1e6, 1e6)
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                X_sc = pd.DataFrame(self.scaler.transform(X), columns=cols)
                X_sc = X_sc.fillna(0).replace([np.inf, -np.inf], 0)
                X_sc = X_sc.clip(-50, 50)  # Empêche overflow dans sigmoid/softmax
            except Exception as e:
                print(f"[KnownAttack] Scaler error : {e}")
                X_sc = X

        if self.pt:
            pt_cols = [c for c in X_sc.columns
                       if c in [f"log_{k}" for k in ALL_LOG_COLS]]
            if pt_cols:
                with np.errstate(over='ignore', invalid='ignore'):
                    try:
                        X_sc[pt_cols] = self.pt.transform(X_sc[pt_cols])
                        X_sc[pt_cols] = X_sc[pt_cols].clip(-50, 50)
                    except Exception:
                        pass

        return X_sc

    def predict(self, features: dict) -> dict:
        X = self._preprocess(features)
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                is_attack = int(self.binary.predict(X)[0])
                bin_proba = float(self.binary.predict_proba(X)[0][1])
                bin_proba = float(np.clip(bin_proba, 0.0, 1.0))
            except Exception as e:
                print(f"[KnownAttack] Binary predict error : {e}")
                return {"is_attack": False, "label": "Normal", "attack_type": None, "confidence": 0.0, "bin_proba": 0.0}

        if not is_attack:
            return {"is_attack": False, "label": "Normal", "attack_type": None,
                    "confidence": round(float(np.clip(1 - bin_proba, 0, 1)), 4),
                    "bin_proba": round(bin_proba, 4)}

        with np.errstate(over='ignore', invalid='ignore'):
            try:
                class_idx   = self.multiclass.predict(X)[0]
                class_prob  = self.multiclass.predict_proba(X)[0]
                class_prob  = np.clip(class_prob, 0.0, 1.0)
                attack_type = self.label_enc.inverse_transform([class_idx])[0]
            except Exception as e:
                print(f"[KnownAttack] Multiclass predict error : {e}")
                return {"is_attack": True, "label": "Attack", "attack_type": "Unknown",
                        "confidence": round(bin_proba, 4), "bin_proba": round(bin_proba, 4)}

        return {
            "is_attack":   True,
            "label":       "Attack",
            "attack_type": attack_type,
            "confidence":  round(float(class_prob.max()), 4),
            "bin_proba":   round(bin_proba, 4),
            "all_probs":   {cls: round(float(p), 4)
                            for cls, p in zip(self.label_enc.classes_, class_prob)},
        }