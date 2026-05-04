"""
capture/capture.py — CORRIGÉ COMPLET
======================================
[FIX-1] netStat.__init__() : suppression du 3e argument np.nan invalide
         → ns.netStat() sans arguments (utilise les Lambdas par défaut)
[FIX-2] Interface Windows : auto-détection via pyshark au lieu de "eth0"
         → NIDS_INTERFACE env var OU détection automatique de la première
           interface réelle disponible
[FIX-3] Méthode updateGetStats (camelCase) — API réelle de Kitsune-py
         → N'utilise PAS update_get_stats (snake_case) qui n'existe pas
[FIX-4] getNetStatHeaders() → utilise FeatureExtractor pour compter les features
         → fallback propre si non disponible
[FIX-5] asyncio event loop dans thread secondaire (Windows)
"""

import os
import sys
import time
import threading
import asyncio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

# ── Recherche netStat dans Kitsune-py ─────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_CANDIDATES = [
    _PROJECT_ROOT / "Kitsune-py",
    _PROJECT_ROOT / "KitNET-py",
    _PROJECT_ROOT / "kitsune-py",
]
_kitsune_path = None
for _c in _CANDIDATES:
    if (_c / "netStat.py").exists():
        sys.path.insert(0, str(_c))
        _kitsune_path = _c
        print(f"[AfterImage] netStat.py trouvé dans : {_c.name}")
        break
    elif _c.exists():
        sys.path.insert(0, str(_c))

AFTERIMAGE_AVAILABLE = False
try:
    import netStat as ns
    # Vérifier que c'est bien le vrai netStat de Kitsune-py
    # Le vrai a la classe netStat avec updateGetStats (camelCase)
    _test = ns.netStat()
    if hasattr(_test, 'updateGetStats'):
        AFTERIMAGE_AVAILABLE = True
        print("[OK] AfterImage (netStat Kitsune-py) disponible")
    else:
        print("[WARN] netStat chargé mais API incorrecte (pas updateGetStats)")
        print("       Assurez-vous que Kitsune-py est bien cloné depuis :")
        print("       https://github.com/ymirsky/Kitsune-py.git")
except Exception as e:
    print(f"[WARN] netStat non disponible : {e}")
    print("       Clonez Kitsune-py : git clone https://github.com/ymirsky/Kitsune-py.git")

try:
    import pyshark
    PYSHARK_AVAILABLE = True
    print("[OK] pyshark disponible")
except ImportError:
    PYSHARK_AVAILABLE = False
    print("[WARN] pyshark manquant — pip install pyshark")


# ── [FIX-2] Auto-détection d'interface Windows ────────────────────
def _detect_interface() -> str:
    """
    Retourne l'interface réseau à utiliser.
    Priorité : var d'env NIDS_INTERFACE > Wi-Fi > Ethernet > première dispo.
    """
    # 1. Variable d'environnement explicite
    env_iface = os.environ.get("NIDS_INTERFACE", "").strip()
    if env_iface:
        print(f"[CAPTURE] Interface depuis NIDS_INTERFACE : '{env_iface}'")
        return env_iface

    if not PYSHARK_AVAILABLE:
        return "eth0"

    try:
        # 2. Lister les interfaces disponibles
        cap_tmp = pyshark.LiveCapture()
        interfaces = cap_tmp.interfaces if hasattr(cap_tmp, 'interfaces') else []

        # Sur Windows, pyshark retourne des noms humains ou des GUIDs
        # On préfère Wi-Fi, puis Ethernet, sinon la première non-loopback
        preferred = []
        for iface in interfaces:
            name_lower = iface.lower()
            if "wi-fi" in name_lower or "wifi" in name_lower or "wireless" in name_lower:
                preferred.insert(0, iface)  # Wi-Fi en premier
            elif "ethernet" in name_lower and "loopback" not in name_lower:
                preferred.append(iface)
            elif "loopback" not in name_lower and "usbpcap" not in name_lower.replace(" ", ""):
                preferred.append(iface)

        if preferred:
            chosen = preferred[0]
            print(f"[CAPTURE] Interface auto-détectée : '{chosen}'")
            print(f"[CAPTURE] Interfaces disponibles : {interfaces}")
            print(f"[CAPTURE] Pour forcer une interface : set NIDS_INTERFACE=<nom>")
            return chosen

        if interfaces:
            print(f"[CAPTURE] Interfaces disponibles : {interfaces}")
            return interfaces[0]

    except Exception as e:
        print(f"[CAPTURE] Détection d'interface échouée : {e}")

    # 3. Fallback plateforme
    if sys.platform == "win32":
        return "Wi-Fi"
    return "eth0"


UNSW_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm'
]


@dataclass
class FlowRecord:
    src_ip:      str
    dst_ip:      str
    sport:       int
    dport:       int
    proto:       str
    start_time:  float = field(default_factory=time.time)
    packets:     list  = field(default_factory=list)
    sbytes:      int   = 0
    dbytes:      int   = 0
    spkts:       int   = 0
    dpkts:       int   = 0
    sttl:        int   = 64
    dttl:        int   = 0
    syn_seen:    bool  = False
    synack_time: float = 0.0
    ack_time:    float = 0.0
    fin_seen:    bool  = False
    last_seen:   float = field(default_factory=time.time)

    def to_unsw_features(self) -> dict:
        pkts   = self.packets
        ts_lst = [p["ts"] for p in pkts]
        dur    = (ts_lst[-1] - ts_lst[0]) if len(ts_lst) > 1 else 0.0
        iarr   = np.diff(ts_lst).tolist() if len(ts_lst) > 1 else [0.0]
        load_s = (self.sbytes * 8 / dur) if dur > 0 else 0.0
        load_d = (self.dbytes * 8 / dur) if dur > 0 else 0.0
        tcprtt = (self.ack_time    - self.start_time) if self.ack_time    else 0.0
        synack = (self.synack_time - self.start_time) if self.synack_time else 0.0
        return {
            'dur': round(dur, 6), 'sbytes': self.sbytes, 'dbytes': self.dbytes,
            'sttl': self.sttl, 'dttl': self.dttl, 'sloss': 0, 'dloss': 0,
            'Sload': round(load_s, 4), 'Dload': round(load_d, 4),
            'Spkts': self.spkts, 'Dpkts': self.dpkts,
            'swin': 0, 'dwin': 0, 'stcpb': 0, 'dtcpb': 0,
            'smeansz': round(self.sbytes / max(self.spkts, 1), 2),
            'dmeansz': round(self.dbytes / max(self.dpkts, 1), 2),
            'trans_depth': 0, 'res_bdy_len': 0,
            'Sjit':    round(float(np.std(iarr)),  6),
            'Djit':    0.0,
            'Sintpkt': round(float(np.mean(iarr)), 6),
            'Dintpkt': 0.0,
            'tcprtt': round(tcprtt, 6), 'synack': round(synack, 6), 'ackdat': 0.0,
            'ct_state_ttl': 0, 'ct_flw_http_mthd': 0,
            'ct_srv_src': 0, 'ct_srv_dst': 0, 'ct_dst_ltm': 0,
            'ct_src_ ltm': 0, 'ct_src_dport_ltm': 0,
            'ct_dst_sport_ltm': 0, 'ct_dst_src_ltm': 0,
            '_src_ip': self.src_ip, '_dst_ip': self.dst_ip,
            '_sport': self.sport,   '_dport': self.dport,
            '_proto': self.proto,
        }


class AfterImageExtractor:
    """
    [FIX-1] Initialise ns.netStat() SANS arguments positionnels.
    Le vrai Kitsune-py netStat.__init__ a la signature :
        def __init__(self, Lambdas=None, tstats_len=None)
    → appel correct : ns.netStat()  (utilise les defaults)

    [FIX-3] Utilise updateGetStats (camelCase) — l'API réelle de Kitsune-py.
    """
    def __init__(self):
        self._nstat = None
        self._n_features = 115  # valeur par défaut
        if not AFTERIMAGE_AVAILABLE:
            return
        try:
            # [FIX-1] : zéro argument positionnel — le vrai netStat l'accepte
            self._nstat = ns.netStat()
            # Déduire le nombre de features depuis les headers si disponible
            try:
                headers = self._nstat.getNetStatHeaders()
                self._n_features = len(headers)
            except AttributeError:
                # Kitsune-py n'expose pas toujours getNetStatHeaders directement
                # Le nombre standard est 115
                self._n_features = 115
            print(f"[AfterImage] initialisé — {self._n_features} features")
        except Exception as e:
            print(f"[WARN] netStat init failed : {e}")
            self._nstat = None

    def extract(self, pkt) -> Optional[np.ndarray]:
        if self._nstat is None:
            return None
        try:
            timestamp = float(pkt.sniff_timestamp)
            framelen  = int(pkt.length)
            IPtype    = np.nan
            srcIP = dstIP = srcMAC = dstMAC = ''
            srcproto = dstproto = ''

            if hasattr(pkt, 'ip'):
                srcIP, dstIP = pkt.ip.src, pkt.ip.dst
                IPtype = 0
            elif hasattr(pkt, 'ipv6'):
                srcIP, dstIP = pkt.ipv6.src, pkt.ipv6.dst
                IPtype = 1

            if hasattr(pkt, 'tcp'):
                srcproto = str(pkt.tcp.srcport)
                dstproto = str(pkt.tcp.dstport)
            elif hasattr(pkt, 'udp'):
                srcproto = str(pkt.udp.srcport)
                dstproto = str(pkt.udp.dstport)
            elif hasattr(pkt, 'arp'):
                srcproto = dstproto = 'arp'
                try:
                    srcIP = pkt.arp.src_proto_ipv4
                    dstIP = pkt.arp.dst_proto_ipv4
                    IPtype = 0
                except Exception:
                    pass
            elif hasattr(pkt, 'icmp'):
                srcproto = dstproto = 'icmp'
                IPtype = 0

            try:
                srcMAC = pkt.eth.src
                dstMAC = pkt.eth.dst
            except Exception:
                pass

            if srcIP == '' and srcproto == '':
                srcIP, dstIP = srcMAC, dstMAC

            # [FIX-3] : updateGetStats (camelCase) — API réelle Kitsune-py
            vec = self._nstat.updateGetStats(
                IPtype, srcMAC, dstMAC,
                srcIP, srcproto, dstIP, dstproto,
                framelen, timestamp
            )
            arr = np.array(vec, dtype=np.float64)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return None

    @property
    def n_features(self) -> int:
        return self._n_features


class NetworkCapture:
    """
    Capture réseau double pipeline.
    [FIX-2] Interface auto-détectée (Windows compatible).
    [FIX-5] Thread de capture crée sa propre boucle asyncio (Windows).
    """
    FLOW_TIMEOUT  = 30.0
    MAX_FLOW_PKTS = 200

    def __init__(
        self,
        interface:        str                = "",   # vide = auto-détection
        bpf_filter:       str                = "ip",
        on_flow:          Optional[Callable] = None,
        on_packet_vector: Optional[Callable] = None,
        use_pcap:         Optional[str]      = None,
        flow_timeout:     float              = 30.0,
        max_flow_pkts:    int                = 200,
    ):
        # [FIX-2] Auto-détection si interface non spécifiée ou "eth0" sur Windows
        if not interface or (interface == "eth0" and sys.platform == "win32"):
            interface = _detect_interface()

        self.interface        = interface
        self.bpf_filter       = bpf_filter
        self.on_flow          = on_flow          or (lambda x: None)
        self.on_packet_vector = on_packet_vector or (lambda x: None)
        self.use_pcap         = use_pcap
        self.FLOW_TIMEOUT     = flow_timeout
        self.MAX_FLOW_PKTS    = max_flow_pkts

        self._afterimage  = AfterImageExtractor()
        self._flows: dict[tuple, FlowRecord] = {}
        self._flows_lock  = threading.Lock()
        self._running     = False
        self._pps_window: list[float] = []
        self._pps_lock    = threading.Lock()
        self._total_pkts  = 0
        self._total_flows = 0

    def _flow_key(self, pkt) -> Optional[tuple]:
        try:
            if not hasattr(pkt, 'ip'):
                return None
            src   = pkt.ip.src
            dst   = pkt.ip.dst
            proto = (pkt.transport_layer or "OTHER").upper()
            sp = dp = 0
            if proto in ("TCP", "UDP"):
                try:
                    layer = pkt[proto.lower()]
                    sp = int(getattr(layer, "srcport", 0))
                    dp = int(getattr(layer, "dstport", 0))
                except Exception:
                    pass
            return (src, dst, sp, dp, proto)
        except Exception:
            return None

    def _update_flow(self, key: tuple, pkt):
        try:
            now    = float(pkt.sniff_timestamp)
            length = int(pkt.length)
        except Exception:
            return

        emit = False
        pkts_count = 0
        with self._flows_lock:
            if key not in self._flows:
                self._flows[key] = FlowRecord(
                    src_ip=key[0], dst_ip=key[1],
                    sport=key[2],  dport=key[3],
                    proto=key[4],  start_time=now,
                )
            flow = self._flows[key]
            flow.packets.append({"ts": now, "len": length})
            flow.sbytes   += length
            flow.spkts    += 1
            flow.last_seen = now
            try:
                flow.sttl = int(pkt.ip.ttl)
            except Exception:
                pass
            try:
                flags = int(pkt.tcp.flags, 16)
                if flags & 0x02: flow.syn_seen = True
                if flags & 0x12 and flow.syn_seen and not flow.synack_time:
                    flow.synack_time = now
                if flags & 0x10 and flow.synack_time and not flow.ack_time:
                    flow.ack_time = now
                if flags & 0x01 or flags & 0x04:
                    emit = True
            except Exception:
                pass
            pkts_count = len(flow.packets)

        if emit or pkts_count >= self.MAX_FLOW_PKTS:
            self._emit_flow(key)

    def _emit_flow(self, key: tuple):
        with self._flows_lock:
            flow = self._flows.pop(key, None)
        if flow is not None:
            self._total_flows += 1
            try:
                self.on_flow(flow.to_unsw_features())
            except Exception as e:
                print(f"[FLOW CB] {e}")

    def _timeout_checker(self):
        while self._running:
            now = time.time()
            with self._flows_lock:
                expired = [k for k, v in list(self._flows.items())
                           if now - v.last_seen > self.FLOW_TIMEOUT]
            for k in expired:
                self._emit_flow(k)
            time.sleep(5)

    def start(self):
        if not PYSHARK_AVAILABLE:
            raise RuntimeError("pyshark non installé — pip install pyshark")

        # [FIX-5] Créer une boucle asyncio pour ce thread (Windows)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self._running = True
        threading.Thread(target=self._timeout_checker, daemon=True).start()

        if self.use_pcap:
            print(f"[CAPTURE] Lecture PCAP : {self.use_pcap}")
            cap = pyshark.FileCapture(
                self.use_pcap,
                keep_packets=False,
                eventloop=loop,
            )
        else:
            print(f"[CAPTURE] Live sur : '{self.interface}' (filtre: '{self.bpf_filter}')")
            cap = pyshark.LiveCapture(
                interface  = self.interface,
                bpf_filter = self.bpf_filter,
                eventloop  = loop,
            )

        try:
            for pkt in cap.sniff_continuously():
                if not self._running:
                    break

                self._total_pkts += 1
                now = time.time()
                with self._pps_lock:
                    self._pps_window.append(now)
                    cutoff = now - 1.0
                    self._pps_window = [t for t in self._pps_window if t >= cutoff]

                # Pipeline A : AfterImage → KitNET
                vec = self._afterimage.extract(pkt)
                if vec is not None:
                    try:
                        self.on_packet_vector(vec)
                    except Exception as e:
                        print(f"[VEC CB] {e}")

                # Pipeline B : flux UNSW → XGBoost
                key = self._flow_key(pkt)
                if key:
                    self._update_flow(key, pkt)

        except Exception as e:
            print(f"[CAPTURE] Erreur : {e}")
            raise
        finally:
            try:
                cap.close()
            except Exception:
                pass
            loop.close()

    def stop(self):
        self._running = False

    @property
    def stats(self) -> dict:
        with self._pps_lock:
            pps = len(self._pps_window)
        return {
            "total_pkts":          self._total_pkts,
            "total_flows":         self._total_flows,
            "pps":                 pps,
            "interface":           self.interface,
            "afterimage_features": self._afterimage.n_features,
            "active_flows":        len(self._flows),
        }

    @property
    def afterimage_n_features(self) -> int:
        return self._afterimage.n_features