"""Controller-only manifest inspection and causal observation/label adapters."""
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
from .policy import canonical

def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def flatten(inputs):
    """Canonical feature order is sorted object keys, array index order."""
    pairs = []
    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + "/" + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + "/" + str(index))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            pairs.append((path, float(value)))
        else:
            raise ValueError("Non-finite or non-numeric input at " + path)
    visit(inputs, "")
    return [key for key, value in pairs], [value for key, value in pairs]

class Handoff:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if self.manifest["format"] != "industrial-harness-handoff.v1":
            raise ValueError("Unsupported manifest")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def resolve(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Data path escapes handoff root")
        return path

    def descriptor(self, descriptor):
        path = self.resolve(descriptor["path"])
        expected = descriptor["sha256"].removeprefix("sha256:")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("SHA256 mismatch: " + str(path))
        return path

    def scenario(self, scenario):
        return self.manifest["scenarios"][scenario]

    def verify(self, scenario):
        spec = self.scenario(scenario)
        checked = []
        def walk(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    checked.append(str(self.descriptor(value)))
                else:
                    for child in value.values():
                        walk(child)
        walk(spec)
        roles = {}
        with self.resolve(spec["index"]["path"]).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["id"] in roles:
                    raise ValueError("Duplicate sample in split index")
                roles[row["id"]] = row["role"]
        for role, split in spec["splits"].items():
            desc = split.get("dataset", split.get("observations"))
            rows = list(read_jsonl(self.resolve(desc["path"])))
            if len(rows) != split["rows"]:
                raise ValueError("Row count mismatch")
            ids = set()
            for row in rows:
                if row["id"] in ids or row["id"] not in roles:
                    raise ValueError("Duplicate or unindexed sample")
                ids.add(row["id"])
                # The index uses the source role names from the handoff.
                if roles[row["id"]] != role:
                    raise ValueError("Role mismatch: " + role)
                _, values = flatten(row["inputs"])
                if len(values) != spec["input_features"]:
                    raise ValueError("Dimension mismatch")
        return {"scenario": scenario, "verified_files": len(checked),
                "samples": len(roles), "manifest_sha256": self.sha256}

    def training_features(self, scenario):
        spec = self.scenario(scenario)
        path = self.descriptor(spec["splits"]["industrial_train"]["dataset"])
        order = None
        values = []
        for row in read_jsonl(path):
            keys, vector = flatten(row["inputs"])
            if order is not None and keys != order:
                raise ValueError("Feature schema changed")
            order = keys
            values.append(vector)
        return order, values

class Replay:
    """Only this source iterates unseen observations; no consumer can request lookahead."""
    def __init__(self, path, batch_size, cursor=0):
        if batch_size <= 0 or cursor < 0:
            raise ValueError("Invalid replay cursor or batch size")
        self.path, self.batch_size, self.cursor = Path(path), batch_size, cursor

    def batches(self):
        batch, group = [], None
        for position, row in enumerate(read_jsonl(self.path)):
            if position < self.cursor:
                continue
            if batch and (row["group_id"] != group or len(batch) == self.batch_size):
                yield batch
                batch = []
            group = row["group_id"]
            batch.append(row)
        if batch:
            yield batch

    def commit(self, source_count):
        self.cursor += source_count

class Windows:
    def __init__(self, short=2, long=8):
        self.group = None
        self.batches = deque(maxlen=long)
        self.short = short

    def add(self, group, rows):
        if self.group != group:
            self.group = group
            self.batches.clear()
        self.batches.append(rows)

    def context(self):
        return {"short": [r for batch in list(self.batches)[-self.short:] for r in batch],
                "long": [r for batch in self.batches for r in batch],
                "batches": len(self.batches)}

def release_ids(ids, scenario, group, remaining):
    if type(remaining) is not int or remaining < 0 or len(set(ids)) != len(ids):
        raise ValueError("Invalid label quota or duplicate IDs")
    ranked = sorted(ids, key=lambda sid: hashlib.sha256(canonical([42, scenario, group, sid]).encode()).hexdigest())
    return ranked[:min(math.ceil(len(ids)*0.25), remaining)]


class LabelProvider:
    """Private label file; callers must present arrived IDs with saved predictions."""
    def __init__(self, path):
        self._labels = {row["id"]: row["answer"] for row in read_jsonl(path)}

    def release(self, batch, predicted_ids, scenario, remaining, already=None):
        if remaining < 0:
            raise ValueError("Negative label budget")
        ids = [row["id"] for row in batch]
        if not set(ids) <= set(predicted_ids):
            raise ValueError("Save predictions before requesting labels")
        group = batch[0]["group_id"] if batch else ""
        selected = release_ids(ids, scenario, group, remaining)
        result = {sid: self._labels[sid] for sid in selected}
        if already:
            for sid, value in already.items():
                if sid in result and result[sid] != value:
                    raise ValueError("Label changed")
        return result
