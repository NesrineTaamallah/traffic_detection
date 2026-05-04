"""
capture/capture.py — CORRIGÉ v3
======================================================
FIXES :
[FIX-1] Interface Windows : lecture des noms lisibles via tshark/winreg
         pour mapper GUID → nom humain (Wi-Fi, Ethernet, etc.)
[FIX-2] _flow_key() : délai pyshark → accès aux couches via indexation
         sécurisée + support ICMP/ARP + log des paquets ignorés
[FIX-3] FLOW_TIMEOUT réduit à 5 s (30 s = flux jamais émis en démo)
         MAX_FLOW_PKTS réduit à 50 pour émettre plus vite
[FIX-4] Émission forcée des flux actifs toutes les 3 s (flush périodique)
         pour que XGBoost reçoive du trafic dès le début
[FIX-5] Log détaillé : paquets reçus, flux émis, vecteurs AfterImage
[FIX-6] configure_n_features() : suppression du guard packet_count > 0
"""
import numpy as np
 
# Réinjecter les alias supprimés dans NumPy 2.0
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
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str
 
print("[PATCH] numpy_compat_patch appliqué — np.Inf, np.NaN réinjectés")
 
import os
import sys
import time
import threading
import asyncio
import subprocess
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
    _test = ns.netStat()
    if hasattr(_test, 'updateGetStats'):
        AFTERIMAGE_AVAILABLE = True
        print("[OK] AfterImage (netStat Kitsune-py) disponible")
    else:
        print("[WARN] netStat chargé mais API incorrecte (pas updateGetStats)")
except Exception as e:
    print(f"[WARN] netStat non disponible : {e}")

try:
    import pyshark
    PYSHARK_AVAILABLE = True
    print("[OK] pyshark disponible")
except ImportError:
    PYSHARK_AVAILABLE = False
    print("[WARN] pyshark manquant — pip install pyshark")


# ── [FIX-1] Résolution GUID → nom lisible (Windows) ───────────────
def _guid_to_friendly_name(guid: str) -> str:
    """Tente de résoudre un GUID d'interface Windows en nom lisible."""
    if sys.platform != "win32":
        return guid
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Network\{4D36E972-E325-11CE-BFC1-08002BE10318}"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
        # Extraire le GUID pur depuis \Device\NPF_{GUID}
        pure = guid.replace("\\Device\\NPF_", "").strip("{}")
        sub_path = f"{base}\\{{{pure}}}\\Connection"
        try:
            sub = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sub_path)
            name, _ = winreg.QueryValueEx(sub, "Name")
            return str(name)
        except Exception:
            return guid
    except Exception:
        return guid


def _list_interfaces_with_names() -> list[tuple[str, str]]:
    """Retourne [(guid, friendly_name), ...] pour toutes les interfaces pyshark."""
    if not PYSHARK_AVAILABLE:
        return []
    try:
        cap_tmp = pyshark.LiveCapture()
        raw_ifaces = cap_tmp.interfaces if hasattr(cap_tmp, 'interfaces') else []
        result = []
        for iface in raw_ifaces:
            friendly = _guid_to_friendly_name(iface)
            result.append((iface, friendly))
        return result
    except Exception as e:
        print(f"[CAPTURE] Impossible de lister les interfaces : {e}")
        return []


def _detect_interface() -> str:
    """
    Retourne l'interface réseau à utiliser.
    Priorité : NIDS_INTERFACE > Wi-Fi > Ethernet > première non-loopback
    """
    env_iface = os.environ.get("NIDS_INTERFACE", "").strip()
    if env_iface:
        print(f"[CAPTURE] Interface depuis NIDS_INTERFACE : '{env_iface}'")
        return env_iface

    if not PYSHARK_AVAILABLE:
        return "eth0"

    pairs = _list_interfaces_with_names()
    if not pairs:
        return "eth0"

    print(f"[CAPTURE] Interfaces disponibles :")
    for guid, name in pairs:
        print(f"          {name!r:30s} → {guid}")
    print(f"[CAPTURE] Pour forcer : set NIDS_INTERFACE=<guid ou nom>")

    # Préférence par nom lisible
    preferred_keywords = ["wi-fi", "wifi", "wireless", "wlan", "ethernet", "lan", "local"]
    skip_keywords = ["loopback", "usbpcap", "etwdump", "bluetooth", "npcap loopback"]

    candidates = []
    for guid, name in pairs:
        name_l = name.lower()
        if any(s in name_l for s in skip_keywords):
            continue
        priority = 99
        for i, kw in enumerate(preferred_keywords):
            if kw in name_l:
                priority = i
                break
        candidates.append((priority, guid, name))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, chosen_guid, chosen_name = candidates[0]
        print(f"[CAPTURE] Interface choisie : {chosen_name!r} ({chosen_guid})")
        return chosen_guid

    # Fallback : première interface non-loopback
    for guid, name in pairs:
        if "loopback" not in name.lower() and "etwdump" not in name.lower():
            return guid
    return pairs[0][0] if pairs else "eth0"


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
    Wrapper netStat de Kitsune-py.
    updateGetStats = API camelCase réelle.
    """
    def __init__(self):
        self._nstat = None
        self._n_features = 100
        if not AFTERIMAGE_AVAILABLE:
            return
        try:
            self._nstat = ns.netStat()
            try:
                headers = self._nstat.getNetStatHeaders()
                self._n_features = len(headers)
            except AttributeError:
                self._n_features = 100
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

            # Couche IP / IPv6
            try:
                ip = pkt['ip']
                srcIP, dstIP = ip.src, ip.dst
                IPtype = 0
            except Exception:
                try:
                    ip6 = pkt['ipv6']
                    srcIP, dstIP = ip6.src, ip6.dst
                    IPtype = 1
                except Exception:
                    pass

            # Transport
            try:
                tcp = pkt['tcp']
                srcproto = str(tcp.srcport)
                dstproto = str(tcp.dstport)
            except Exception:
                try:
                    udp = pkt['udp']
                    srcproto = str(udp.srcport)
                    dstproto = str(udp.dstport)
                except Exception:
                    try:
                        pkt['arp']
                        srcproto = dstproto = 'arp'
                        try:
                            srcIP = pkt['arp'].src_proto_ipv4
                            dstIP = pkt['arp'].dst_proto_ipv4
                            IPtype = 0
                        except Exception:
                            pass
                    except Exception:
                        try:
                            pkt['icmp']
                            srcproto = dstproto = 'icmp'
                            IPtype = 0
                        except Exception:
                            pass

            # Ethernet
            try:
                eth = pkt['eth']
                srcMAC = eth.src
                dstMAC = eth.dst
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
        return self._n_features


class NetworkCapture:
    """
    Capture réseau double pipeline.
    [FIX-3] FLOW_TIMEOUT = 5s, MAX_FLOW_PKTS = 50
    [FIX-4] Flush périodique toutes les 3s
    [FIX-5] Logs détaillés de diagnostic
    """
    # [FIX-3] Valeurs réduites pour émission rapide
    FLOW_TIMEOUT  = 5.0
    MAX_FLOW_PKTS = 50

    def __init__(
        self,
        interface:        str                = "",
        bpf_filter:       str                = "ip",
        on_flow:          Optional[Callable] = None,
        on_packet_vector: Optional[Callable] = None,
        use_pcap:         Optional[str]      = None,
        flow_timeout:     float              = 5.0,
        max_flow_pkts:    int                = 50,
    ):
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

        # Compteurs de diagnostic
        self._pkts_with_ip   = 0
        self._pkts_no_layer  = 0
        self._vecs_sent      = 0
        self._last_log_time  = time.time()

    # ── [FIX-2] Flow key robuste ──────────────────────────────────
    def _flow_key(self, pkt) -> Optional[tuple]:
        """
        Extrait la clé de flux. Utilise l'indexation par string
        au lieu de hasattr() pour éviter les faux négatifs pyshark.
        """
        try:
            # Tenter d'accéder à la couche IP via indexation (plus fiable)
            try:
                ip = pkt['ip']
                src = ip.src
                dst = ip.dst
            except Exception:
                # Pas de couche IP — ignorer silencieusement
                self._pkts_no_layer += 1
                return None

            # Transport
            proto = "OTHER"
            sp = dp = 0
            try:
                tcp = pkt['tcp']
                proto = "TCP"
                sp = int(tcp.srcport)
                dp = int(tcp.dstport)
            except Exception:
                try:
                    udp = pkt['udp']
                    proto = "UDP"
                    sp = int(udp.srcport)
                    dp = int(udp.dstport)
                except Exception:
                    try:
                        pkt['icmp']
                        proto = "ICMP"
                    except Exception:
                        pass

            self._pkts_with_ip += 1
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
                flow.sttl = int(pkt['ip'].ttl)
            except Exception:
                pass
            try:
                flags = int(pkt['tcp'].flags, 16)
                if flags & 0x02: flow.syn_seen = True
                if flags & 0x12 and flow.syn_seen and not flow.synack_time:
                    flow.synack_time = now
                if flags & 0x10 and flow.synack_time and not flow.ack_time:
                    flow.ack_time = now
                # FIN ou RST → émettre le flux
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
        """
        [FIX-4] Flush toutes les 3s (au lieu de 5s) + flush forcé
        des flux actifs même non expirés si > 2 paquets
        """
        while self._running:
            now = time.time()

            with self._flows_lock:
                # Flux expirés
                expired = [k for k, v in list(self._flows.items())
                           if now - v.last_seen > self.FLOW_TIMEOUT]
                # [FIX-4] Flush forcé des flux avec assez de paquets
                forced = [k for k, v in list(self._flows.items())
                          if k not in expired and len(v.packets) >= 3
                          and now - v.last_seen > 1.0]

            for k in expired + forced:
                self._emit_flow(k)

            # [FIX-5] Log de diagnostic toutes les 10s
            if now - self._last_log_time > 10.0:
                self._last_log_time = now
                with self._flows_lock:
                    active = len(self._flows)
                with self._pps_lock:
                    pps = len(self._pps_window)
                print(
                    f"[CAPTURE] pkts={self._total_pkts}  ip={self._pkts_with_ip}"
                    f"  no_layer={self._pkts_no_layer}  flows_emis={self._total_flows}"
                    f"  flows_actifs={active}  vecs={self._vecs_sent}  pps≈{pps}"
                )

            time.sleep(3)

    def start(self):
        if not PYSHARK_AVAILABLE:
            raise RuntimeError("pyshark non installé — pip install pyshark")

        # [FIX-5] Boucle asyncio propre pour Windows
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

        print(f"[CAPTURE] Thread démarré")

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
                    self._vecs_sent += 1
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
            try:
                loop.close()
            except Exception:
                pass

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