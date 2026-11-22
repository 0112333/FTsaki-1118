"""Pure-Python, frozen 32-position forecast gate."""
from dataclasses import dataclass, asdict, field
from math import isfinite, sqrt
from statistics import fmean, pstdev
from .policy import digest

def matrix(rows, dimension, count=None):
    if count is not None and len(rows) != count:
        raise ValueError("Wrong horizon")
    result = []
    for row in rows:
        if len(row) != dimension or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) for x in row):
            raise ValueError("Wrong shape or non-finite feature")
        result.append([float(x) for x in row])
    return result

@dataclass(frozen=True)
class Scale:
    mean: tuple
    std: tuple

    @classmethod
    def fit(cls, rows):
        if not rows:
            raise ValueError("Empty industrial training reference")
        rows = matrix(rows, len(rows[0]))
        columns = list(zip(*rows))
        return cls(tuple(fmean(c) for c in columns), tuple(pstdev(c) for c in columns))

    def __post_init__(self):
        if not self.mean or len(self.mean) != len(self.std):
            raise ValueError("Invalid scale shape")
        if any(not isfinite(x) for x in (*self.mean, *self.std)) or any(x < 0 for x in self.std):
            raise ValueError("Invalid frozen scale")

    @property
    def tolerance(self):
        return [1e-6 * max(1, abs(mu)) for mu in self.mean]

def _trend(values):
    mean = fmean(values)
    slope = sum((i - 15.5) * (v - mean) for i, v in enumerate(values)) / 2728
    change = 31 * slope
    return 1 if change > 0.20 else -1 if change < -0.20 else 0

def evaluate(scenario, prediction, actual, history, scale):
    d = len(scale.mean)
    prediction = matrix(prediction, d, 32)
    if len(actual) != 32 or any(row is None for row in actual):
        return {"outcome": "insufficient", "reason": "incomplete-target"}
    actual = matrix(actual, d, 32)
    history = matrix(history, d)
    if not history or (scenario == "A" and len(history) < 32):
        return {"outcome": "insufficient", "reason": "incomplete-history"}
    if scenario not in "ABC" or len(scenario) != 1:
        raise ValueError("Unknown scenario")
    tolerance = scale.tolerance
    variable = [j for j in range(d) if scale.std[j] > tolerance[j]]
    constant_ok = all(abs(prediction[h][j] - actual[h][j]) <= tolerance[j]
                      for j in range(d) if j not in variable for h in range(32))
    baseline = history[-32:] if scenario == "A" else [history[-1]] * 32
    def error(candidate):
        if not variable:
            return 0.0
        if scenario == "A":
            return fmean(abs(a-b)/scale.std[j] for j in variable
                         for a,b in zip(sorted(row[j] for row in candidate), sorted(row[j] for row in actual)))
        return sqrt(fmean(((candidate[h][j]-actual[h][j])/scale.std[j])**2
                          for h in range(32) for j in variable))
    e, eb = error(prediction), error(baseline)
    trend = (fmean(_trend([row[j]/scale.std[j] for row in prediction]) ==
                   _trend([row[j]/scale.std[j] for row in actual]) for j in variable)
             if variable and scenario != "A" else 1.0)
    passed = constant_ok and e <= 0.50 and trend >= 0.80 and e <= 1.05 * eb + 0.02
    return {"outcome": "passed" if passed else "rejected", "error": e,
            "baseline_error": eb, "trend_match": trend, "constant_ok": constant_ok,
            "variable_features": len(variable)}

@dataclass
class Forecast:
    candidate: str
    scenario: str
    group: str
    as_of: int
    prediction: list
    history: list
    scale: Scale
    actual: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    closed: bool = False

    def __post_init__(self):
        if self.scenario not in {"A", "B", "C"}:
            raise ValueError("Unknown scenario")
        self.prediction = matrix(self.prediction, len(self.scale.mean), 32)
        self.history = matrix(self.history, len(self.scale.mean))
        if not self.history or (self.scenario == "A" and len(self.history) < 32):
            raise ValueError("Insufficient past history before forecast")
        self._frozen_hash = self.frozen_hash

    @property
    def frozen_hash(self):
        return digest({"candidate": self.candidate, "scenario": self.scenario,
                       "group": self.group, "as_of": self.as_of, "prediction": self.prediction,
                       "history": self.history, "scale": asdict(self.scale)})

    def append(self, sample_id, group, order, row):
        if self.frozen_hash != self._frozen_hash:
            raise ValueError("Frozen forecast was modified")
        if self.closed:
            raise ValueError("Forecast already terminated")
        if sample_id in self.ids:
            return None
        if group != self.group:
            self.closed = True
            return {"outcome": "insufficient", "reason": "group-ended"}
        if order <= (self.orders[-1] if self.orders else self.as_of):
            raise ValueError("Target not in increasing source order after cutoff")
        self.orders.append(order)
        self.ids.append(sample_id)
        try:
            self.actual.append(matrix([row], len(self.scale.mean))[0])
        except (ValueError, TypeError):
            self.actual.append(None)
        if len(self.actual) == 32:
            self.closed = True
            return evaluate(self.scenario, self.prediction, self.actual, self.history, self.scale)
        return None

    def end(self):
        self.closed = True
        return {"outcome": "insufficient", "reason": "stream-ended", "observed": len(self.actual)}

    def dump(self):
        return {**asdict(self), "frozen_hash": self._frozen_hash}

    @classmethod
    def restore(cls, data):
        data = dict(data)
        expected = data.pop("frozen_hash")
        data["scale"] = Scale(**data["scale"])
        obj = cls(**data)
        if obj.frozen_hash != expected:
            raise ValueError("Frozen record hash mismatch")
        return obj
