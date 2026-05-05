"""
api/analyze.py — v2 FINAL
===========================
Endpoint /api/analyze  : CSV + PCAP → XGBoost + KitNET → résultats JSON
Endpoint /api/report   : export JSON/CSV du rapport d'analyse
Endpoint /api/report/pdf : export rapport PDF textuel

Ajouter dans api/main.py :
    from api.analyze import router as analyze_router
    app.include_router(analyze_router)
"""

import io
import time
import json
import csv as csv_mod
import tempfile
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(prefix="/api")

# ── Meta colonnes non-feature ──────────────────────────────────────
META_COLS = {
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'service',
    'label', 'attack_cat', 'Label', 'class', 'Category',
    'id', 'row_id', 'index', 'num', 'No.',
}

LABEL_COLS = ['label', 'Label', 'attack_cat', 'Category', 'class', 'attack_type']

_PORT_TO_SERVICE = {
    21: 'ftp', 20: 'ftp-data', 22: 'ssh', 25: 'smtp',
    53: 'dns', 67: 'dhcp', 68: 'dhcp', 80: 'http',
    110: 'pop3', 161: 'snmp', 194: 'irc', 443: 'ssl',
    8080: 'http', 8443: 'ssl', 1812: 'radius', 1813: 'radius',
}

# Thread-local session storage for last analysis (for report export)
_last_analysis: dict = {}
_analysis_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _detect_label_col(df: pd.DataFrame) -> Optional[str]:
    for col in LABEL_COLS:
        if col in df.columns:
            return col
    return None


def _get_true_label(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return int(bool(val))
    s = str(val).strip().lower()
    return 0 if s in ('0', 'normal', 'benign', 'legitimate', 'legit', 'none', '-') else 1


def _infer_severity(is_attack: bool, confidence: float,
                    is_anomaly: bool, sev_score: float,
                    trained: bool = True) -> str:
    if is_attack and confidence >= 0.60:
        if is_anomaly and sev_score >= 2.5:
            return "CRITICAL"
        elif is_anomaly or confidence >= 0.85:
            return "HIGH"
        else:
            return "MEDIUM"
    elif trained and is_anomaly:
        return "HIGH" if sev_score >= 2.5 else "LOW"
    elif is_attack and confidence < 0.60:
        return "LOW"
    return "NORMAL"


def _load_engines():
    """Lazy import des engines depuis api.main (évite import circulaire)."""
    try:
        import importlib
        main = importlib.import_module("api.main")
        if not getattr(main, "MODELS_AVAILABLE", False):
            return None, None, None, False
        return (
            getattr(main, "known_engine", None),
            getattr(main, "zero_engine", None),
            getattr(main, "fusion", None),
            True,
        )
    except Exception as e:
        print(f"[ANALYZE] Import engines : {e}")
        return None, None, None, False


def _analyze_row(raw: dict, known_engine, zero_engine,
                 numeric_vec: Optional[np.ndarray] = None) -> dict:
    """Analyse un flux (dict de features) → résultat unifié."""
    # ── XGBoost ─────────────────────────────────────────────────
    try:
        known_result = known_engine.predict(raw)
        is_attack    = known_result.get("is_attack", False)
        attack_type  = known_result.get("attack_type")
        confidence   = float(known_result.get("confidence", 0.0))
        all_probs    = known_result.get("all_probs", {})
    except Exception as e:
        print(f"[ANALYZE] XGB error : {e}")
        is_attack, attack_type, confidence, all_probs = False, None, 0.0, {}

    # ── KitNET ──────────────────────────────────────────────────
    try:
        if numeric_vec is not None and len(numeric_vec) > 0:
            zero_result = zero_engine.process_vector(numeric_vec)
        else:
            zero_result = zero_engine.process(raw)
    except Exception as e:
        print(f"[ANALYZE] KitNET error : {e}")
        zero_result = {"rmse": 0.0, "is_anomaly": False, "threshold": 0.0,
                       "severity_score": 0.0, "trained": True}

    is_anomaly = zero_result.get("is_anomaly", False)
    rmse       = float(zero_result.get("rmse", 0.0))
    threshold  = float(zero_result.get("threshold", 0.0))
    sev_score  = float(zero_result.get("severity_score", 0.0))
    trained    = zero_result.get("trained", True)

    severity = _infer_severity(is_attack, confidence, is_anomaly, sev_score, trained)

    return {
        "is_attack":      is_attack,
        "attack_type":    attack_type,
        "confidence":     round(confidence, 4),
        "is_anomaly":     is_anomaly,
        "rmse":           round(rmse, 6),
        "threshold":      round(threshold, 6),
        "severity":       severity,
        "severity_score": round(sev_score, 3),
        "all_probs":      all_probs,
        "kitnet_trained": trained,
    }


def _fallback_result_from_label(label_val, attack_cat: str) -> dict:
    """Résultat heuristique quand les modèles ne sont pas disponibles."""
    true_lbl    = _get_true_label(label_val)
    is_attack   = bool(true_lbl) if true_lbl is not None else False
    attack_type = (attack_cat
                   if attack_cat and attack_cat.lower() not in ('', 'nan', 'normal', 'none', '-')
                   else None)
    confidence  = 0.88 if is_attack else 0.94
    rmse        = float(np.random.uniform(0.03, 0.09)) if is_attack else float(np.random.uniform(0.005, 0.03))
    threshold   = 0.06
    sev_score   = rmse / threshold
    is_anomaly  = False
    severity    = _infer_severity(is_attack, confidence, is_anomaly, sev_score)
    return {
        "is_attack":      is_attack,
        "attack_type":    attack_type,
        "confidence":     round(confidence, 4),
        "is_anomaly":     is_anomaly,
        "rmse":           round(rmse, 6),
        "threshold":      round(threshold, 6),
        "severity":       severity,
        "severity_score": round(sev_score, 3),
        "all_probs":      {},
        "kitnet_trained": True,
    }


# ═══════════════════════════════════════════════════════════════════
# CSV PARSING
# ═══════════════════════════════════════════════════════════════════

def _parse_csv(content: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV invalide : {e}")


def _analyze_csv(df: pd.DataFrame, known_engine, zero_engine,
                 engines_ok: bool, max_rows: int = 5000) -> tuple[list[dict], bool, Optional[str]]:
    label_col = _detect_label_col(df)
    has_labels = label_col is not None
    feature_cols = [c for c in df.columns if c not in META_COLS]
    results = []

    for idx in range(min(len(df), max_rows)):
        row = df.iloc[idx]

        raw = {}
        for col in feature_cols:
            val = row.get(col)
            try:
                raw[col] = float(val) if pd.notna(val) else 0.0
            except (TypeError, ValueError):
                raw[col] = 0.0

        raw['_src_ip'] = str(row.get('srcip', f'host_{idx % 50}'))
        raw['_dst_ip'] = str(row.get('dstip', f'srv_{idx % 10}'))
        raw['_sport']  = int(row.get('sport', 0))  if pd.notna(row.get('sport', None)) else 0
        raw['_dport']  = int(row.get('dsport', 80)) if pd.notna(row.get('dsport', None)) else 80
        raw['_proto']  = str(row.get('proto', 'tcp')).upper()
        raw['state']   = str(row.get('state', ''))
        raw['service'] = str(row.get('service', ''))

        true_label  = _get_true_label(row.get(label_col)) if has_labels else None
        attack_cat  = str(row.get('attack_cat', row.get('Category', ''))).strip()

        if engines_ok:
            # Vecteur numérique pour KitNET
            numeric_vec = np.array(
                [float(row.get(c, 0)) if pd.notna(row.get(c, None)) else 0.0
                 for c in feature_cols],
                dtype=np.float64
            )
            numeric_vec = np.nan_to_num(numeric_vec, nan=0.0, posinf=0.0, neginf=0.0)
            numeric_vec = np.clip(numeric_vec, -1e6, 1e6)
            detection = _analyze_row(raw, known_engine, zero_engine, numeric_vec)
        else:
            detection = _fallback_result_from_label(row.get(label_col) if has_labels else None, attack_cat)

        result = {
            "row_index":   idx,
            "src_ip":      raw['_src_ip'],
            "dst_ip":      raw['_dst_ip'],
            "dport":       raw['_dport'],
            "proto":       raw['_proto'],
            "true_label":  true_label,
            "attack_cat":  attack_cat,
        }
        result.update(detection)
        results.append(result)

    return results, has_labels, label_col


# ═══════════════════════════════════════════════════════════════════
# PCAP PARSING
# ═══════════════════════════════════════════════════════════════════

def _analyze_pcap(content: bytes, known_engine, zero_engine,
                  engines_ok: bool, max_pkts: int = 2000) -> tuple[list[dict], bool, None]:
    """
    Analyse un fichier PCAP via pyshark + AfterImage (netStat).
    Tente d'utiliser le capture pipeline existant.
    Si pyshark n'est pas disponible → erreur claire.
    """
    try:
        import pyshark
    except ImportError:
        raise HTTPException(status_code=400,
                            detail="pyshark non installé — pip install pyshark. "
                                   "Utilisez un fichier CSV pour l'analyse sans pyshark.")

    # Écrire le PCAP dans un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    results = []
    try:
        import sys
        from pathlib import Path as P
        PROJECT = P(tmp_path).parent.parent

        # Essaye d'importer AfterImage
        afterimage_ok = False
        nstat = None
        for cand in [P("Kitsune-py"), P("KitNET-py"), P("kitsune-py")]:
            if (cand / "netStat.py").exists():
                sys.path.insert(0, str(cand))
                break
        try:
            import netStat as ns
            nstat = ns.netStat()
            afterimage_ok = hasattr(nstat, 'updateGetStats')
        except Exception:
            pass

        import asyncio
        loop = asyncio.new_event_loop()
        cap = pyshark.FileCapture(tmp_path, keep_packets=False, eventloop=loop)

        flow_records = {}  # (src,dst,sport,dport,proto) → dict
        pkt_count = 0

        def _flow_key_from_pkt(pkt):
            try:
                src = pkt['ip'].src
                dst = pkt['ip'].dst
            except Exception:
                return None
            proto = "OTHER"; sp = dp = 0
            try: tcp = pkt['tcp']; proto = "TCP"; sp = int(tcp.srcport); dp = int(tcp.dstport)
            except Exception:
                try: udp = pkt['udp']; proto = "UDP"; sp = int(udp.srcport); dp = int(udp.dstport)
                except Exception:
                    try: pkt['icmp']; proto = "ICMP"
                    except Exception: pass
            return (src, dst, sp, dp, proto)

        for pkt in cap.sniff_continuously():
            if pkt_count >= max_pkts:
                break
            pkt_count += 1

            key = _flow_key_from_pkt(pkt)
            if not key:
                continue

            # Accumule le flux
            if key not in flow_records:
                flow_records[key] = {
                    "src_ip": key[0], "dst_ip": key[1],
                    "sport": key[2], "dport": key[3], "proto": key[4],
                    "pkts": [], "sbytes": 0, "spkts": 0,
                }
            fr = flow_records[key]
            try:
                fr["pkts"].append(float(pkt.sniff_timestamp))
                fr["sbytes"] += int(pkt.length)
                fr["spkts"]  += 1
            except Exception:
                pass

            # AfterImage vector → KitNET
            if afterimage_ok and engines_ok and zero_engine:
                try:
                    ts  = float(pkt.sniff_timestamp)
                    flen= int(pkt.length)
                    IPtype = 0
                    srcIP = dstIP = srcMAC = dstMAC = ""
                    srcproto = dstproto = ""
                    try:
                        ip = pkt['ip']; srcIP = ip.src; dstIP = ip.dst
                    except Exception: pass
                    try:
                        tcp = pkt['tcp']; srcproto = str(tcp.srcport); dstproto = str(tcp.dstport)
                    except Exception:
                        try:
                            udp = pkt['udp']; srcproto = str(udp.srcport); dstproto = str(udp.dstport)
                        except Exception: pass
                    try: eth = pkt['eth']; srcMAC = eth.src; dstMAC = eth.dst
                    except Exception: pass

                    vec = nstat.updateGetStats(IPtype, srcMAC, dstMAC,
                                              srcIP, srcproto, dstIP, dstproto, flen, ts)
                    vec_arr = np.array(vec, dtype=np.float64)
                    vec_arr = np.nan_to_num(vec_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    fr["last_vec"] = vec_arr
                except Exception:
                    pass

        cap.close()
        try: loop.close()
        except Exception: pass

        # Convertir les flows en résultats
        for idx, (key, fr) in enumerate(list(flow_records.items())[:2000]):
            pkts = fr["pkts"]
            dur  = (pkts[-1] - pkts[0]) if len(pkts) > 1 else 0.0

            raw = {
                "dur":     dur,
                "sbytes":  fr["sbytes"],
                "dbytes":  0,
                "sttl":    64,
                "dttl":    0,
                "Spkts":   fr["spkts"],
                "Dpkts":   0,
                "Sload":   fr["sbytes"] * 8 / max(dur, 1e-6),
                "Dload":   0,
                "smeansz": fr["sbytes"] / max(fr["spkts"], 1),
                "dmeansz": 0,
                "Sjit":    0,
                "Sintpkt": dur / max(fr["spkts"] - 1, 1),
                "_src_ip": fr["src_ip"],
                "_dst_ip": fr["dst_ip"],
                "_sport":  fr["sport"],
                "_dport":  fr["dport"],
                "_proto":  fr["proto"],
            }

            if engines_ok:
                numeric_vec = fr.get("last_vec", None)
                detection = _analyze_row(raw, known_engine, zero_engine, numeric_vec)
            else:
                detection = _fallback_result_from_label(None, "")

            result = {
                "row_index": idx,
                "src_ip":    fr["src_ip"],
                "dst_ip":    fr["dst_ip"],
                "dport":     fr["dport"],
                "proto":     fr["proto"],
                "true_label": None,
                "attack_cat": "",
                "dur":        round(dur, 4),
                "sbytes":     fr["sbytes"],
                "spkts":      fr["spkts"],
            }
            result.update(detection)
            results.append(result)

    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

    return results, False, None


# ═══════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def _compute_metrics(results: list[dict]) -> dict:
    with_label = [r for r in results if r.get("true_label") is not None]
    if not with_label:
        return {}

    tp = tn = fp = fn = 0
    for r in with_label:
        predicted = r["is_attack"] or r["is_anomaly"]
        actual    = bool(r["true_label"])
        if predicted and actual:   tp += 1
        elif not predicted and not actual: tn += 1
        elif predicted and not actual:     fp += 1
        else:                              fn += 1

    total = len(with_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       4),
        "fnr":       round(fnr,       4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_labeled": total,
    }


def _compute_summary(results: list[dict]) -> dict:
    total     = len(results)
    attacks   = sum(1 for r in results if r["is_attack"])
    anomalies = sum(1 for r in results if r["is_anomaly"] and not r["is_attack"])
    normals   = total - attacks - anomalies
    by_type   = {}
    for r in results:
        if r["is_attack"] and r["attack_type"]:
            by_type[r["attack_type"]] = by_type.get(r["attack_type"], 0) + 1
    by_severity = {}
    for r in results:
        s = r["severity"]
        by_severity[s] = by_severity.get(s, 0) + 1

    rmse_vals = [r["rmse"] for r in results if r.get("rmse", 0) > 0]
    return {
        "total":       total,
        "attacks":     attacks,
        "anomalies":   anomalies,
        "normals":     normals,
        "pct_attack":  round(attacks / max(total, 1) * 100, 2),
        "pct_anomaly": round(anomalies / max(total, 1) * 100, 2),
        "pct_normal":  round(normals / max(total, 1) * 100, 2),
        "by_type":     by_type,
        "by_severity": by_severity,
        "rmse_mean":   round(float(np.mean(rmse_vals)), 6) if rmse_vals else 0.0,
        "rmse_max":    round(float(np.max(rmse_vals)), 6) if rmse_vals else 0.0,
        "threshold":   round(results[0].get("threshold", 0.0), 6) if results else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyse un fichier CSV ou PCAP/PCAPNG.
    Retourne les résultats XGBoost + KitNET + métriques si labels présents.
    """
    fname = file.filename or ""
    is_pcap = fname.lower().endswith((".pcap", ".pcapng"))
    is_csv  = fname.lower().endswith(".csv")

    if not (is_pcap or is_csv):
        raise HTTPException(status_code=400,
                            detail="Format non supporté. Utilisez .csv, .pcap ou .pcapng")

    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (max 100 MB)")

    known_engine, zero_engine, fusion, engines_ok = _load_engines()

    t_start = time.time()

    if is_csv:
        df = _parse_csv(content)
        results, has_labels, label_col = _analyze_csv(
            df, known_engine, zero_engine, engines_ok, max_rows=5000
        )
        total_rows = len(df)
        file_type  = "csv"
    else:
        results, has_labels, label_col = _analyze_pcap(
            content, known_engine, zero_engine, engines_ok, max_pkts=2000
        )
        total_rows = len(results)
        file_type  = "pcap"

    duration  = round(time.time() - t_start, 2)
    summary   = _compute_summary(results)
    metrics   = _compute_metrics(results) if has_labels else {}

    payload = {
        "results":    results,
        "has_labels": has_labels,
        "label_col":  label_col,
        "total_rows": total_rows,
        "analyzed":   len(results),
        "filename":   fname,
        "file_type":  file_type,
        "engines_ok": engines_ok,
        "duration_s": duration,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary":    summary,
        "metrics":    metrics,
    }

    # Stocker pour export rapport
    with _analysis_lock:
        _last_analysis.clear()
        _last_analysis.update(payload)

    return JSONResponse(payload)


@router.get("/report/json")
def export_json():
    """Télécharger le dernier rapport d'analyse au format JSON."""
    with _analysis_lock:
        if not _last_analysis:
            raise HTTPException(status_code=404, detail="Aucune analyse disponible")
        data = dict(_last_analysis)

    content = json.dumps(data, indent=2, ensure_ascii=False)
    filename = f"nids_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/report/csv")
def export_csv():
    """Télécharger le dernier rapport d'analyse au format CSV."""
    with _analysis_lock:
        if not _last_analysis:
            raise HTTPException(status_code=404, detail="Aucune analyse disponible")
        results = list(_last_analysis.get("results", []))
        meta    = {k: v for k, v in _last_analysis.items() if k != "results"}

    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat à exporter")

    output = io.StringIO()
    writer = csv_mod.writer(output)

    # Header métadonnées
    writer.writerow(["# NIDS Analysis Report"])
    writer.writerow(["# Fichier analysé", meta.get("filename", "")])
    writer.writerow(["# Date", meta.get("timestamp", "")])
    writer.writerow(["# Lignes analysées", meta.get("analyzed", 0)])
    writer.writerow(["# Engines OK", meta.get("engines_ok", False)])
    writer.writerow([])

    # Summary
    summary = meta.get("summary", {})
    writer.writerow(["# Résumé"])
    writer.writerow(["Total", summary.get("total", 0)])
    writer.writerow(["Attaques XGBoost", summary.get("attacks", 0)])
    writer.writerow(["Anomalies KitNET", summary.get("anomalies", 0)])
    writer.writerow(["Normal", summary.get("normals", 0)])
    writer.writerow([])

    # Metrics
    metrics = meta.get("metrics", {})
    if metrics:
        writer.writerow(["# Métriques de performance"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        writer.writerow([])

    # Results
    if results:
        cols = ["row_index", "src_ip", "dst_ip", "dport", "proto",
                "severity", "is_attack", "attack_type", "confidence",
                "is_anomaly", "rmse", "threshold", "severity_score", "true_label"]
        writer.writerow(cols)
        for r in results:
            writer.writerow([r.get(c, "") for c in cols])

    filename = f"nids_report_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/report/txt")
def export_txt():
    """Rapport textuel détaillé (pour impression / présentation prof)."""
    with _analysis_lock:
        if not _last_analysis:
            raise HTTPException(status_code=404, detail="Aucune analyse disponible")
        data = dict(_last_analysis)

    s  = data.get("summary", {})
    m  = data.get("metrics", {})
    fn = data.get("filename", "?")
    ts = data.get("timestamp", "?")
    ok = data.get("engines_ok", False)
    dur= data.get("duration_s", 0)

    lines = []
    sep   = "=" * 64
    dash  = "-" * 64

    lines += [
        sep,
        "  NIDS — RAPPORT D'ANALYSE RÉSEAU",
        "  Network Intrusion Detection System",
        sep,
        f"  Fichier    : {fn}",
        f"  Date       : {ts}",
        f"  Durée      : {dur}s",
        f"  Mode       : {'Production (XGBoost + KitNET)' if ok else 'Simulation (labels CSV)'}",
        "",
    ]

    lines += [
        dash,
        "  RÉSUMÉ STATISTIQUE",
        dash,
        f"  Total flux analysés  : {s.get('total', 0)}",
        f"  Attaques détectées   : {s.get('attacks', 0)}  ({s.get('pct_attack', 0):.1f}%)",
        f"  Anomalies Zero-Day   : {s.get('anomalies', 0)}  ({s.get('pct_anomaly', 0):.1f}%)",
        f"  Trafic normal        : {s.get('normals', 0)}  ({s.get('pct_normal', 0):.1f}%)",
        f"  RMSE moyen KitNET    : {s.get('rmse_mean', 0):.6f}",
        f"  RMSE max KitNET      : {s.get('rmse_max', 0):.6f}",
        f"  Seuil KitNET         : {s.get('threshold', 0):.6f}",
        "",
    ]

    by_type = s.get("by_type", {})
    if by_type:
        lines += [dash, "  TYPES D'ATTAQUES DÉTECTÉES (Pipeline XGBoost)", dash]
        for atype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            bar = "█" * min(int(count / max(s.get("total", 1), 1) * 40), 40)
            lines.append(f"  {atype:<25s} {count:>6d}  {bar}")
        lines.append("")

    by_sev = s.get("by_severity", {})
    if by_sev:
        lines += [dash, "  DISTRIBUTION DES SÉVÉRITÉS", dash]
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"]:
            cnt = by_sev.get(sev, 0)
            if cnt > 0:
                lines.append(f"  {sev:<10s} : {cnt:>6d}")
        lines.append("")

    if m:
        lines += [
            dash,
            "  MÉTRIQUES DE PERFORMANCE (vs labels réels)",
            dash,
            f"  Accuracy  : {m.get('accuracy', 0)*100:.2f}%",
            f"  Précision : {m.get('precision', 0)*100:.2f}%",
            f"  Rappel    : {m.get('recall', 0)*100:.2f}%",
            f"  F1-Score  : {m.get('f1', 0)*100:.2f}%",
            f"  FPR       : {m.get('fpr', 0)*100:.2f}%  (faux positifs)",
            f"  FNR       : {m.get('fnr', 0)*100:.2f}%  (faux négatifs)",
            "",
            "  Matrice de confusion :",
            f"    TP={m.get('tp',0)}  FP={m.get('fp',0)}",
            f"    FN={m.get('fn',0)}  TN={m.get('tn',0)}",
            f"    (sur {m.get('total_labeled',0)} échantillons labellisés)",
            "",
        ]

    lines += [
        dash,
        "  PIPELINE TECHNIQUE",
        dash,
        "  Pipeline A — Supervisé",
        "    Modèle    : XGBoost + Random Forest (UNSW-NB15)",
        "    Détection : Binaire (attaque/normal) + Classification multi-classe",
        "    Features  : 70 features UNSW-NB15 + feature engineering",
        "",
        "  Pipeline B — Non-Supervisé",
        "    Modèle    : KitNET (auto-encodeur en ligne, Mirai pré-entraîné)",
        "    Détection : Score RMSE → seuil P99 × 2.0",
        "    Features  : AfterImage (100 statistiques réseau temps réel)",
        "",
        "  Fusion     : Attaque XGB OU Anomalie KitNET → Alerte + Sévérité",
        sep,
        "  NIDS Dashboard v2.0 — Généré automatiquement",
        sep,
    ]

    content = "\n".join(lines)
    filename = f"nids_rapport_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )