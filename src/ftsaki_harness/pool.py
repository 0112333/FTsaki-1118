"""Deterministic voting and trial lifecycle. Decimal scores are not voting shares."""
from dataclasses import dataclass, asdict
from decimal import Decimal as D

BINARY = {"drift", "no_drift"}

@dataclass
class Model:
    version: int
    parent: int | None = None
    stable: bool = False
    online: bool = True
    score: str = "0.60"
    eligible: bool = True
    reason: str | None = None

class Pool:
    def __init__(self):
        self.models = {0: Model(0, stable=True, score="1.00")}
        self.windows = {}
        self.group = None
        self.positive = self.negative = 0
        self.event_open = False

    def weights(self):
        stable = sorted((m.version for m in self.models.values() if m.online and m.stable), reverse=True)
        result = {v: D("0.8") ** rank for rank, v in enumerate(stable)}
        result.update({m.version: D(m.score) for m in self.models.values() if m.online and not m.stable})
        return result

    def retire(self, version, reason="capacity"):
        model = self.models[version]
        if not model.online:
            return
        if model.stable and sum(m.online and m.stable for m in self.models.values()) <= 1:
            raise ValueError("Cannot retire the last stable model")
        model.online, model.reason = False, reason
        for child in self.models.values():
            if child.online and not child.stable and child.parent == version:
                child.online, child.reason = False, "parent-retired"

    def admit(self, version, parent):
        if version in self.models:
            return "duplicate"
        if version <= max(self.models):
            raise ValueError("Versions must increase by production order")
        if parent not in self.models or not (self.models[parent].online and self.models[parent].stable):
            return "parent-retired"
        online = [m for m in self.models.values() if m.online]
        if len(online) >= 4:
            oldest = min(online, key=lambda m: m.version)
            if oldest.stable and sum(m.stable for m in online) == 1:
                return "waiting"
            if oldest.version == parent:
                return "parent-retired"
            self.retire(oldest.version)
        self.models[version] = Model(version, parent)
        return "admitted"

    def _aggregate(self, weights, votes, stable_only=False):
        total = sum(weights.values(), D(0))
        valid = {v: w for v, w in weights.items() if votes.get(v) in BINARY}
        coverage = sum(valid.values(), D(0)) / total if total else D(0)
        s = sum((w for v, w in valid.items() if self.models[v].stable), D(0))
        t = sum(valid.values(), D(0)) - s
        if coverage < D("0.70") or not s:
            return {"decision": "insufficient", "coverage": str(coverage), "support": None, "weights": {}}
        factor = min(D(1), D("0.45") * s / (D("0.55") * t)) if t and not stable_only else D(1)
        effective = {v: w if self.models[v].stable else w * factor for v, w in valid.items()}
        denominator = sum(effective.values())
        shares = {v: w / denominator for v, w in effective.items()}
        p = sum((w for v, w in shares.items() if votes[v] == "drift"), D(0))
        decision = "drift" if p >= D("0.65") else "no_drift" if p <= D("0.35") else "observe"
        return {"decision": decision, "coverage": str(coverage), "support": str(p),
                "weights": {str(v): str(w) for v, w in shares.items()}}

    def window(self, window_id, group, votes):
        if window_id in self.windows:
            return self.windows[window_id]
        if group != self.group:
            self.group, self.positive, self.negative, self.event_open = group, 0, 0, False
        weights = self.weights()
        aggregate = self._aggregate(weights, votes)
        reference = self._aggregate({v: w for v, w in weights.items() if self.models[v].stable}, votes, True)
        before = {str(v): asdict(self.models[v]) for v in weights}
        changes = []
        for v in weights:
            m = self.models[v]
            if m.stable or votes.get(v) not in BINARY or reference["decision"] not in BINARY:
                continue
            score = D(m.score) + (D("0.05") if votes[v] == reference["decision"] else D("-0.10"))
            m.score = str(score)
            if score > 1:
                m.stable = True
                changes.append({"version": v, "action": "promoted", "score": str(score)})
            elif score < D("0.60"):
                m.online, m.reason = False, "low-score"
                changes.append({"version": v, "action": "evicted", "score": str(score)})
        decision = aggregate["decision"]
        self.positive = self.positive + 1 if decision == "drift" else 0
        self.negative = self.negative + 1 if decision == "no_drift" else 0
        trigger = self.positive >= 2 and not self.event_open
        if trigger:
            self.event_open = True
        if self.negative >= 2:
            self.event_open = False
        result = {"window_id": window_id, "group": group, "before": before,
                  "raw_weights": {str(v): str(w) for v, w in weights.items()},
                  "votes": {str(v): vote for v, vote in votes.items()},
                  "aggregate": aggregate, "reference": reference, "trigger": trigger,
                  "changes": changes, "after": self.dump()["models"]}
        self.windows[window_id] = result
        return result

    def dump(self):
        return {"models": {str(v): asdict(m) for v, m in self.models.items()},
                "windows": self.windows, "group": self.group, "positive": self.positive,
                "negative": self.negative, "event_open": self.event_open}

    @classmethod
    def restore(cls, data):
        pool = cls()
        pool.models = {int(v): Model(**m) for v, m in data["models"].items()}
        for name in ("windows", "group", "positive", "negative", "event_open"):
            setattr(pool, name, data[name])
        return pool
