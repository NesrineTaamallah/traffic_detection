"""
capture/capture.py — CORRIGÉ Windows
======================================
[FIX WIN-1] asyncio event loop manquant dans le thread de capture (Windows)
            → asyncio.set_event_loop(asyncio.new_event_loop()) avant pyshark
[FIX WIN-2] Interface réseau : "eth0" invalide sur Windows
            → utilise l'interface détectée par api/main.py via NIDS_INTERFACE
[FIX WIN-3] netStat.py absent de Kitsune-py cloné → fallback propre
[FIX WIN-4] pyshark.LiveCapture sur Windows nécessite Npcap (pas Wireshark seul)
"""

import sys, time, threading, asyncio, numpy as np
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
for _c in _CANDIDATES:
    if (_c / "netStat.py").exists():
        sys.path.insert(0, str(_c))
        print(f"[AfterImage] netStat.py trouvé dans : {_c.name}")
        break
    elif _c.exists():
        sys.path.insert(0, str(_c))

try:
    import netStat as ns
    AFTERIMAGE_AVAILABLE = True
    print("[OK] AfterImage (netStat) disponible")
except ImportError:
    AFTERIMAGE_AVAILABLE = False
    print("[WARN] netStat.py introuvable dans Kitsune-py")
    print("       Le dossier Kitsune-py existe mais netStat.py est absent.")
    print("       Solution : supprimez Kitsune-py et re-clonez :")
    print("       rmdir /s /q Kitsune-py")
    print("       git clone https://github.com/ymirsky/Kitsune-py.git")

try:
    import pyshark
    PYSHARK_AVAILABLE = True
    print("[OK] pyshark disponible")
except ImportError:
    PYSHARK_AVAILABLE = False
    print("[WARN] pyshark manquant — pip install pyshark")

UNSW_FEATURES = [
    'dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',
    'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'ct_state_ttl','ct_flw_http_mthd','ct_srv_src','ct_srv_dst',
    'ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm',
    'ct_dst_sport_ltm','ct_dst_src_ltm'
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
    def __init__(self):
        self._nstat = None
        if not AFTERIMAGE_AVAILABLE:
            return
        try:
            self._nstat = ns.netStat(np.nan, 100_000_000, 100_000_000)
            print(f"[AfterImage] initialisé — {self.n_features} features")
        except Exception as e:
            print(f"[WARN] netStat init failed : {e}")

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
        if self._nstat is None:
            return 0
        try:
            return len(self._nstat.getNetStatHeaders())
        except Exception:
            return 115


class NetworkCapture:
    """
    Capture réseau double pipeline.
    [FIX WIN-1] Le thread de capture crée sa propre boucle asyncio (Windows).
    [FIX WIN-2] Interface passée en paramètre depuis api/main.py (auto-détectée).
    """
    FLOW_TIMEOUT  = 30.0
    MAX_FLOW_PKTS = 200

    def __init__(
        self,
        interface:        str                = "Wi-Fi",
        bpf_filter:       str                = "ip",
        on_flow:          Optional[Callable] = None,
        on_packet_vector: Optional[Callable] = None,
        use_pcap:         Optional[str]      = None,
        flow_timeout:     float              = 30.0,
        max_flow_pkts:    int                = 200,
    ):
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
            raise RuntimeError("pyshark non installé")

        # ── [FIX WIN-1] Créer une boucle asyncio pour CE thread ───
        # pyshark utilise asyncio en interne. Sur Windows, le thread
        # principal d'uvicorn possède déjà une boucle, mais les threads
        # secondaires n'en ont pas → RuntimeError "no current event loop"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self._running = True
        threading.Thread(target=self._timeout_checker, daemon=True).start()

        if self.use_pcap:
            print(f"[CAPTURE] Lecture PCAP : {self.use_pcap}")
            cap = pyshark.FileCapture(
                self.use_pcap,
                keep_packets=False,
                eventloop=loop,          # passer la boucle explicitement
            )
        else:
            print(f"[CAPTURE] Live sur : '{self.interface}' (filtre: '{self.bpf_filter}')")
            print(f"[CAPTURE] Windows : Npcap requis — https://npcap.com")
            print(f"[CAPTURE] Pour lister les interfaces : python -c \"import pyshark; print(pyshark.LiveCapture().interfaces)\"")
            cap = pyshark.LiveCapture(
                interface  = self.interface,
                bpf_filter = self.bpf_filter,
                eventloop  = loop,       # [FIX WIN-1] boucle explicite
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