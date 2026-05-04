"""
fusion/decision.py
==================
Combine XGBoost (attaques connues) + KitNET (anomalies zero-day)
→ alerte unifiée avec niveau de sévérité.
"""

from dataclasses import dataclass
from enum import Enum
import time


class Severity(str, Enum):
    NORMAL   = "NORMAL"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    timestamp:      float
    src_ip:         str
    dst_ip:         str
    sport:          int
    dport:          int
    proto:          str
    severity:       Severity
    is_attack:      bool
    attack_type:    str | None
    confidence:     float
    rmse:           float
    is_anomaly:     bool
    severity_score: float
    raw_features:   dict | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp":      self.timestamp,
            "ts_human":       time.strftime('%H:%M:%S', time.localtime(self.timestamp)),
            "src_ip":         self.src_ip,
            "dst_ip":         self.dst_ip,
            "sport":          self.sport,
            "dport":          self.dport,
            "proto":          self.proto,
            "severity":       self.severity.value,
            "is_attack":      self.is_attack,
            "attack_type":    self.attack_type,
            "confidence":     self.confidence,
            "rmse":           self.rmse,
            "is_anomaly":     self.is_anomaly,
            "severity_score": self.severity_score,
        }


class DecisionFusion:
    """
    Logique de fusion des deux pipelines de détection.

    Paramètres
    ----------
    confidence_threshold : seuil de confiance minimum pour valider une attaque XGBoost
    anomaly_factor_high  : ratio RMSE/seuil pour qualifier une anomalie "forte"
    anomaly_factor_med   : ratio RMSE/seuil pour qualifier une anomalie "modérée"
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        anomaly_factor_high:  float = 2.5,
        anomaly_factor_med:   float = 1.5,
    ):
        self.conf_thresh  = confidence_threshold
        self.anomaly_high = anomaly_factor_high
        self.anomaly_med  = anomaly_factor_med

    def decide(
        self,
        features:     dict,
        known_result: dict,
        zero_result:  dict,
    ) -> Alert | None:

        is_attack   = known_result.get("is_attack", False)
        attack_type = known_result.get("attack_type")
        confidence  = known_result.get("confidence", 0.0)
        rmse        = zero_result.get("rmse", 0.0)
        is_anomaly  = zero_result.get("is_anomaly", False)
        sev_score   = zero_result.get("severity_score", 0.0)
        trained     = zero_result.get("trained", False)

        if is_attack and confidence >= self.conf_thresh:
            if is_anomaly and sev_score >= self.anomaly_high:
                severity = Severity.CRITICAL
            elif is_anomaly or confidence >= 0.85:
                severity = Severity.HIGH
            else:
                severity = Severity.MEDIUM
        elif trained and is_anomaly:
            severity = Severity.HIGH if sev_score >= self.anomaly_high else Severity.LOW
        elif is_attack and confidence < self.conf_thresh:
            severity = Severity.LOW
        else:
            return None  # trafic normal

        return Alert(
            timestamp      = time.time(),
            src_ip         = features.get("_src_ip", "?"),
            dst_ip         = features.get("_dst_ip", "?"),
            sport          = features.get("_sport", 0),
            dport          = features.get("_dport", 0),
            proto          = features.get("_proto", "?"),
            severity       = severity,
            is_attack      = is_attack,
            attack_type    = attack_type or ("Zero-day" if is_anomaly else None),
            confidence     = confidence,
            rmse           = rmse,
            is_anomaly     = is_anomaly,
            severity_score = sev_score,
        )