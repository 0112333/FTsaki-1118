"""Frozen development profile; external dependencies remain explicit."""
from dataclasses import asdict, dataclass
import hashlib
import json

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

@dataclass(frozen=True)
class Policy:
    version: str = "development.v1"
    mode: str = "development"
    capacity: int = 4
    training_concurrency: int = 1
    horizon: int = 32
    short_batches: int = 2
    long_batches: int = 8
    label_seed: int = 42
    label_fraction: float = 0.25

    def __post_init__(self):
        if (self.capacity, self.training_concurrency, self.horizon,
            self.short_batches, self.long_batches, self.label_seed,
            self.label_fraction) != (4, 1, 32, 2, 8, 42, 0.25):
            raise ValueError("Changing development constants requires a new policy implementation")
        if self.mode != "development" or self.version != "development.v1":
            raise ValueError("Only the frozen development.v1 profile is implemented")

    @property
    def sha256(self):
        return digest(asdict(self))

    def batch_size(self, scenario):
        return {"A": 16, "B": 8, "C": 32}[scenario]

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            return cls(**json.load(handle))

DEPENDENCIES = {
    "Q01": "Runtime decision supervision contract not supplied",
    "Q02": "PyTRiO training method, targets and minimum cases not supplied",
    "Q03": "Development label release implemented; formal source permissions pending",
    "Q04": "Decision gate specified; industrial release thresholds pending",
    "Q05": "Frozen voting and scoring rules implemented",
    "Q06": "Logical pool 4; decision training concurrency 1; hardware pending",
    "Q07": "External frozen model0 artifact/service and capability acceptance pending",
    "Q08": "Inference adapter available; real inference/PyTRiO contract acceptance pending",
    "Q09": "Finite file replay available; formal source event transport pending",
    "Q10": "Real token, training, label and timeout budgets must be provided",
    "Q11": "Official recording, offline packaging and evaluation contract pending",
    "Q12": "Schema, finite numbers and group boundaries checked",
}
