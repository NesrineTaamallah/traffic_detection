"""
netStat.py — Statistiques réseau pour Kitsune/KitNET
Fichier recréé car absent du clone Kitsune-py.
Source originale : https://github.com/ymirsky/Kitsune-py
"""

import numpy as np
from math import sqrt


# ─────────────────────────────────────────
# STRUCTURES DE STATISTIQUES INCREMENTALES
# ─────────────────────────────────────────

class incStat:
    """
    Statistiques incrementales pour une seule variable (1D).
    Maintient : count, mean, variance (via Welford),LS (least squares).
    """

    def __init__(self, Lambda, ID, init_time=0, isTypediff=False):
        self.ID = ID
        self.Lambda = Lambda       # facteur de décroissance temporelle
        self.isTypediff = isTypediff

        # Statistiques de base
        self.weight   = 1e-20
        self.mean     = 0
        self.variance = 0

        # Suivi temporel
        self.cur_time = init_time

        # Résidus (pour corrélation cross-stat)
        self.covs   = {}   # ID -> incStat_cov
        self._decay = 0

    # ── Mise à jour ──────────────────────────────

    def insert(self, x, t=None):
        """Insère une valeur x au temps t."""
        if t is not None:
            if self.isTypediff:
                x = t - self.cur_time
                if x < 0:
                    x = 0
            # Décroissance temporelle
            dt = t - self.cur_time
            self._age(dt)
            self.cur_time = t

        # Mise à jour Welford
        self.weight  += 1
        old_mean      = self.mean
        self.mean    += (x - self.mean) / self.weight
        self.variance += (x - old_mean) * (x - self.mean)

        # Mise à jour covariances
        for cov in self.covs.values():
            cov.insert(x, t)

    def _age(self, dt):
        """Applique la décroissance exponentielle."""
        if dt <= 0:
            return
        factor = 2 ** (-self.Lambda * dt)
        self.weight   *= factor
        self.mean     *= factor
        self.variance *= factor

    # ── Accesseurs ───────────────────────────────

    def get_weight(self):
        return self.weight

    def get_mean(self):
        if self.weight < 1e-20:
            return 0
        return self.mean / self.weight

    def get_var(self):
        if self.weight < 2:
            return 0
        return max(0, self.variance / (self.weight - 1))

    def get_std(self):
        return sqrt(self.get_var())

    def get_radius(self):
        """Rayon = sqrt(var_1 + var_2) — utilisé pour corrélation."""
        return sqrt(self.get_var())

    def get_magnitude(self):
        return sqrt(self.get_mean() ** 2 + self.get_var())

    def get_pcc(self, other_id):
        """Coefficient de corrélation de Pearson avec autre stat."""
        if other_id not in self.covs:
            return 0
        cov  = self.covs[other_id].get_cov()
        denom = self.get_std() * self.covs[other_id].stat2.get_std()
        if denom == 0:
            return 0
        return cov / denom

    # ── Covariance ───────────────────────────────

    def init_cov(self, other, t=None):
        """Initialise le suivi de covariance avec une autre stat."""
        self.covs[other.ID] = incStat_cov(self, other, t)

    def get_cov(self, other_id):
        if other_id not in self.covs:
            return 0
        return self.covs[other_id].get_cov()


class incStat_cov:
    """Covariance incrémentale entre deux incStat."""

    def __init__(self, stat1, stat2, init_time=None):
        self.stat1   = stat1
        self.stat2   = stat2
        self.weight  = 1e-20
        self.co_sum  = 0
        self.cur_time = init_time

    def insert(self, x, t=None):
        if t is not None and self.cur_time is not None:
            dt = t - self.cur_time
            if dt > 0:
                factor = 2 ** (-self.stat1.Lambda * dt)
                self.weight  *= factor
                self.co_sum  *= factor
            self.cur_time = t

        self.weight += 1
        d1 = x        - self.stat1.get_mean()
        d2 = self.stat2.get_mean()
        self.co_sum += d1 * d2

    def get_cov(self):
        if self.weight < 2:
            return 0
        return self.co_sum / (self.weight - 1)


# ─────────────────────────────────────────
# EXTRACTEUR DE FEATURES (NetStat principal)
# ─────────────────────────────────────────

class netStat:
    """
    Extracteur de features statistiques réseau pour KitNET.
    Calcule 115 features à partir des flux réseau en temps réel.
    """

    def __init__(self, Lambdas=None, tstats_len=None):
        if Lambdas is None:
            Lambdas = [5, 3, 1, 0.1, 0.01]
        if tstats_len is None:
            tstats_len = len(Lambdas)

        self.Lambdas = Lambdas

        # Dictionnaires de stats : clé = (src_ip, dst_ip, ...) ou IP seule
        self.HT_jitter    = {}   # stats inter-arrivée par hôte
        self.HT_MI        = {}   # stats MAC+IP source
        self.HT_H         = {}   # stats IP source
        self.HT_Hp        = {}   # stats IP+port source
        self.HT_HpHp      = {}   # stats flux bidirectionnel

        self.num_features  = 115
        self.feature_names = self._build_feature_names()

    # ── Extraction principale ─────────────────────

    def update_get_stats(self, IPtype, srcMAC, dstMAC, srcIP, srcproto,
                         dstIP, dstproto, datagramSize, timestamp):
        """
        Met à jour les stats et retourne le vecteur de 115 features.
        Compatible avec l'API Kitsune originale.
        """
        # Clés de lookup
        src_str    = f"{srcIP}"
        dst_str    = f"{dstIP}"
        srcport_str = f"{srcIP}:{srcproto}"
        dstport_str = f"{dstIP}:{dstproto}"
        flow_str   = f"{srcIP}:{srcproto}-{dstIP}:{dstproto}"
        rflow_str  = f"{dstIP}:{dstproto}-{srcIP}:{srcproto}"
        mi_str     = f"{srcMAC}:{srcIP}"

        t = timestamp
        sz = datagramSize

        # ── Initialisation si nouveau flux ──────────
        self._init_stat(self.HT_MI,   mi_str,      t)
        self._init_stat(self.HT_H,    src_str,     t)
        self._init_stat(self.HT_Hp,   srcport_str, t)
        self._init_stat(self.HT_HpHp, flow_str,    t)

        # ── Mise à jour ─────────────────────────────
        for stat in self.HT_MI[mi_str]:
            stat.insert(sz, t)
        for stat in self.HT_H[src_str]:
            stat.insert(sz, t)
        for stat in self.HT_Hp[srcport_str]:
            stat.insert(sz, t)

        # Flux bidirectionnel
        rflow_exists = rflow_str in self.HT_HpHp
        for stat in self.HT_HpHp[flow_str]:
            stat.insert(sz, t)

        # ── Construction vecteur features ───────────
        features = []

        # Groupe 1 : stats MI (MAC+IP) — 4 features × nb_lambdas
        for stat in self.HT_MI[mi_str]:
            features += self._get_stat_features(stat)

        # Groupe 2 : stats H (IP source) — 4 features × nb_lambdas
        for stat in self.HT_H[src_str]:
            features += self._get_stat_features(stat)

        # Groupe 3 : stats HH (IP src → IP dst) — corrélation
        if rflow_exists:
            for s1, s2 in zip(self.HT_Hp[srcport_str],
                               self.HT_Hp.get(dstport_str, self.HT_Hp[srcport_str])):
                features += self._get_corr_features(s1, s2)
        else:
            features += [0] * (len(self.Lambdas) * 3)

        # Groupe 4 : stats HpHp (flux complet)
        for stat in self.HT_HpHp[flow_str]:
            features += self._get_stat_features(stat)

        # Padding/troncature à 115
        features = features[:self.num_features]
        while len(features) < self.num_features:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    # ── Helpers ──────────────────────────────────

    def _init_stat(self, table, key, t):
        """Initialise une liste d'incStat pour chaque Lambda si absente."""
        if key not in table:
            table[key] = [incStat(lam, key, t) for lam in self.Lambdas]

    def _get_stat_features(self, stat: incStat) -> list:
        """Retourne [weight, mean, std, radius] pour une incStat."""
        return [
            stat.get_weight(),
            stat.get_mean(),
            stat.get_std(),
            stat.get_radius(),
        ]

    def _get_corr_features(self, s1: incStat, s2: incStat) -> list:
        """Retourne [cov, pcc, magnitude] entre deux incStat."""
        pcc = s1.get_pcc(s2.ID) if s2.ID in s1.covs else 0
        cov = s1.get_cov(s2.ID) if s2.ID in s1.covs else 0
        mag = sqrt(s1.get_magnitude() ** 2 + s2.get_magnitude() ** 2)
        return [cov, pcc, mag]

    def _build_feature_names(self) -> list:
        """Génère les noms des 115 features."""
        names = []
        groups = ['MI', 'H', 'HH', 'HpHp']
        metrics = ['weight', 'mean', 'std', 'radius']
        for g in groups:
            for lam in self.Lambdas:
                for m in metrics:
                    names.append(f"{g}_L{lam}_{m}")
        # compléter jusqu'à 115
        while len(names) < 115:
            names.append(f"feat_{len(names)}")
        return names[:115]

    def get_num_features(self) -> int:
        return self.num_features