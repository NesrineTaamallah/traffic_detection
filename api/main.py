"""
api/main.py
===========
API FastAPI — orchestre les deux pipelines :

  Pipeline A (par paquet) : capture → AfterImage → KitNET  (zero-day)
  Pipeline B (par flux)   : capture → UNSW-35    → XGBoost (attaques connues)
  Fusion                  : DecisionFusion → Alert → WebSocket → Dashboard

  Mode démo : si les modèles ne sont pas trouvés ou la capture impossible,
  l'API génère du trafic synthétique pour permettre de visualiser le dashboard.
"""

import asyncio
import json
import time
import random
import threading
import math
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Imports conditionnels ─────────────────────────────────────────
MODELS_AVAILABLE = False
CAPTURE_AVAILABLE = False

try:
    from capture.capture      import NetworkCapture
    CAPTURE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Capture non disponible : {e}")

try:
    from engines.known_attack import KnownAttackEngine
    from engines.zero_day     import ZeroDayEngine
    from fusion.decision      import DecisionFusion, Severity
    _models_path = Path("./models")
    _required = ["best_binary_model.pkl", "xgb_hierarchical_multiclass.pkl",
                 "scaler_hierarchical.pkl", "label_encoder_hierarchical.pkl"]
    if all((_models_path / f).exists() for f in _required):
        MODELS_AVAILABLE = True
        print("[INFO] Modèles trouvés — mode production activé")
    else:
        missing = [f for f in _required if not (_models_path / f).exists()]
        print(f"[WARN] Modèles manquants : {missing}")
        print("[INFO] Mode démo activé — trafic synthétique")
except ImportError as e:
    print(f"[WARN] Engines non disponibles : {e}")
    print("[INFO] Mode démo activé")

# ── Fallback classes si imports échouent ──────────────────────────
if not MODELS_AVAILABLE:
    class _FusionSev:
        NORMAL="NORMAL"; LOW="LOW"; MEDIUM="MEDIUM"; HIGH="HIGH"; CRITICAL="CRITICAL"
    Severity = _FusionSev()

app = FastAPI(title="NIDS API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── État partagé ───────────────────────────────────────────────────
state = {
    "alerts":        deque(maxlen=500),
    "rmse_series":   deque(maxlen=400),
    "pps_series":    deque(maxlen=400),
    "attack_counts": {},
    "stats": {
        "total_flows":  0,
        "total_pkts":   0,
        "total_alerts": 0,
        "attacks":      0,
        "anomalies":    0,
    },
    "demo_mode": not MODELS_AVAILABLE,
}
_ws_clients: list[WebSocket] = []
_last_flow_result: dict = {}
_last_flow_lock = threading.Lock()

# ── Initialisation engines (production) ───────────────────────────
if MODELS_AVAILABLE:
    known_engine = KnownAttackEngine(models_dir="./models")
    zero_engine  = ZeroDayEngine(fm_grace=5_000, ad_grace=50_000)
    fusion       = DecisionFusion()

    def on_packet_vector(vec):
        zero_result = zero_engine.process_vector(vec)
        state["stats"]["total_pkts"] += 1
        state["rmse_series"].append({"ts": time.time(), "rmse": zero_result["rmse"]})
        if zero_result.get("is_anomaly") and zero_result.get("trained"):
            with _last_flow_lock:
                last = dict(_last_flow_result)
            if last:
                alert = fusion.decide(
                    features     = last.get("features", {}),
                    known_result = last.get("known_result", {"is_attack": False, "confidence": 0.0}),
                    zero_result  = zero_result,
                )
                if alert:
                    _record_alert(alert)

    def on_flow(features: dict):
        state["stats"]["total_flows"] += 1
        known_result = known_engine.predict(features)
        zero_result  = _get_zero_snapshot()
        with _last_flow_lock:
            _last_flow_result.clear()
            _last_flow_result.update({
                "features": features,
                "known_result": known_result,
                "zero_result": zero_result,
            })
        state["pps_series"].append({"ts": time.time(), "pps": capture.stats["pps"]})
        alert = fusion.decide(features, known_result, zero_result)
        if alert:
            _record_alert(alert)

    def _get_zero_snapshot():
        if not zero_engine.rmse_history:
            return {"rmse": 0.0, "is_anomaly": False, "phase": "training",
                    "progress": 0.0, "threshold": 0.0, "severity_score": 0.0, "trained": False}
        r = zero_engine.rmse_history[-1]
        return {
            "rmse":           round(r, 6),
            "is_anomaly":     zero_engine.trained and r > zero_engine.threshold,
            "phase":          "monitoring" if zero_engine.trained else "training",
            "progress":       min(zero_engine.packet_count / zero_engine.grace_total, 1.0),
            "threshold":      round(zero_engine.threshold, 6),
            "severity_score": round(r / max(zero_engine.threshold, 1e-9), 3) if zero_engine.trained else 0.0,
            "trained":        zero_engine.trained,
        }

    def _record_alert(alert):
        state["stats"]["total_alerts"] += 1
        if alert.is_attack:
            state["stats"]["attacks"] += 1
            cat = alert.attack_type or "Unknown"
            state["attack_counts"][cat] = state["attack_counts"].get(cat, 0) + 1
        if alert.is_anomaly:
            state["stats"]["anomalies"] += 1
        state["alerts"].appendleft(alert.to_dict())

    if CAPTURE_AVAILABLE:
        capture = NetworkCapture(
            interface        = "eth0",
            bpf_filter       = "ip",
            on_flow          = on_flow,
            on_packet_vector = on_packet_vector,
        )
        def run_capture():
            try:
                capture.start()
            except Exception as e:
                print(f"[CAPTURE] Erreur : {e}")
        threading.Thread(target=run_capture, daemon=True).start()

# ── Mode démo : génération de trafic synthétique ──────────────────
ATTACK_TYPES = [
    "DoS GoldenEye", "DoS Hulk", "DoS Slowloris",
    "Fuzzers", "Backdoor", "Analysis",
    "Port Scan", "Worms", "Exploits", "Shellcode"
]
SEV_LIST = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DEMO_IPS = ["192.168.1.{i}".format(i=i) for i in range(10, 30)]

_demo_phase = 0      # 0 = normal, counter
_demo_counter = 0
_demo_rmse_base = 0.03
_demo_packet_count = 0
_demo_grace_total = 200   # rapide pour la démo

def _demo_tick():
    """Génère un pas de trafic synthétique réaliste."""
    global _demo_phase, _demo_counter, _demo_rmse_base, _demo_packet_count

    _demo_packet_count += random.randint(80, 300)
    state["stats"]["total_pkts"] = _demo_packet_count
    progress = min(_demo_packet_count / _demo_grace_total, 1.0)
    trained  = progress >= 1.0

    # Oscillation RMSE normale + bursts d'attaque
    _demo_counter += 1
    t = _demo_counter * 0.15
    base_rmse = _demo_rmse_base + 0.008 * math.sin(t) + random.gauss(0, 0.003)
    threshold = _demo_rmse_base * 3.5 if trained else 0.0

    # Toutes les ~30s, simuler une vague d'attaques
    attack_burst = (_demo_counter % 60) < 10 and trained
    if attack_burst:
        rmse = base_rmse + random.uniform(0.05, 0.18)
    else:
        rmse = max(base_rmse, 0.001)

    state["rmse_series"].append({"ts": time.time(), "rmse": round(rmse, 6)})
    pps = random.randint(120, 800) if not attack_burst else random.randint(1200, 4000)
    state["pps_series"].append({"ts": time.time(), "pps": pps})
    state["stats"]["total_flows"] = int(_demo_packet_count * 0.12)

    # Générer des alertes si attaque
    if attack_burst and trained and random.random() < 0.35:
        _generate_demo_alert(rmse, threshold)

    return {
        "packet_count": _demo_packet_count,
        "threshold":    round(threshold, 6),
        "trained":      trained,
        "progress":     round(progress, 4),
        "n_features":   115,
        "mode":         "demo",
        "rmse_last":    round(rmse, 6),
    }

def _generate_demo_alert(rmse, threshold):
    sev_score = rmse / max(threshold, 0.001)
    if sev_score >= 2.5:
        severity = "CRITICAL"
    elif sev_score >= 1.8:
        severity = "HIGH"
    elif sev_score >= 1.2:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    is_attack = random.random() < 0.7
    attack_type = random.choice(ATTACK_TYPES) if is_attack else None
    src_ip = random.choice(DEMO_IPS)
    dst_ip = f"10.0.0.{random.randint(1,10)}"

    alert = {
        "timestamp":      time.time(),
        "ts_human":       time.strftime('%H:%M:%S'),
        "src_ip":         src_ip,
        "dst_ip":         dst_ip,
        "sport":          random.randint(1024, 65535),
        "dport":          random.choice([80, 443, 22, 8080, 3306, 5432]),
        "proto":          random.choice(["TCP", "UDP"]),
        "severity":       severity,
        "is_attack":      is_attack,
        "attack_type":    attack_type or "Zero-day",
        "confidence":     round(random.uniform(0.62, 0.99), 3),
        "rmse":           round(rmse, 6),
        "is_anomaly":     True,
        "severity_score": round(sev_score, 3),
    }

    state["stats"]["total_alerts"] += 1
    if is_attack:
        state["stats"]["attacks"] += 1
        cat = attack_type or "Unknown"
        state["attack_counts"][cat] = state["attack_counts"].get(cat, 0) + 1
    state["stats"]["anomalies"] += 1
    state["alerts"].appendleft(alert)

# ── WebSocket broadcast ────────────────────────────────────────────
async def broadcast(payload: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)

async def push_loop():
    while True:
        if not MODELS_AVAILABLE:
            # Mode démo
            kitnet_info = _demo_tick()
        else:
            kitnet_info = {
                "packet_count": zero_engine.packet_count,
                "threshold":    zero_engine.threshold,
                "trained":      zero_engine.trained,
                "progress":     min(zero_engine.packet_count / zero_engine.grace_total, 1.0),
                "n_features":   zero_engine.n_features,
                "mode":         zero_engine._mode,
            }

        payload = {
            "stats":         state["stats"],
            "alerts":        list(state["alerts"])[:80],
            "rmse_series":   list(state["rmse_series"])[-60:],
            "pps_series":    list(state["pps_series"])[-60:],
            "attack_counts": state["attack_counts"],
            "capture_stats": {
                "total_pkts": state["stats"]["total_pkts"],
                "interface":  "eth0" if CAPTURE_AVAILABLE else "demo",
                "pps":        state["pps_series"][-1]["pps"] if state["pps_series"] else 0,
            },
            "kitnet":        kitnet_info,
            "demo_mode":     not MODELS_AVAILABLE,
        }
        await broadcast(payload)
        await asyncio.sleep(0.8)

@app.on_event("startup")
async def startup():
    asyncio.create_task(push_loop())

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)

# ── REST endpoints ─────────────────────────────────────────────────
@app.get("/alerts")
def get_alerts(limit: int = 100):
    return list(state["alerts"])[:limit]

@app.get("/stats")
def get_stats():
    base = dict(state["stats"])
    base["attack_counts"] = state["attack_counts"]
    base["demo_mode"] = not MODELS_AVAILABLE
    if MODELS_AVAILABLE:
        base["kitnet"] = {
            "packet_count": zero_engine.packet_count,
            "trained":      zero_engine.trained,
            "threshold":    zero_engine.threshold,
        }
    return base

@app.get("/health")
def health():
    return {
        "status":            "ok",
        "models_available":  MODELS_AVAILABLE,
        "capture_available": CAPTURE_AVAILABLE,
        "demo_mode":         not MODELS_AVAILABLE,
    }

if MODELS_AVAILABLE:
    @app.post("/kitnet/save")
    def save_kitnet():
        zero_engine.save("./models/kitnet_state.pkl")
        return {"status": "saved", "packet_count": zero_engine.packet_count}

    @app.post("/kitnet/reset")
    def reset_kitnet():
        global zero_engine
        zero_engine = ZeroDayEngine(fm_grace=5_000, ad_grace=50_000)
        return {"status": "reset"}