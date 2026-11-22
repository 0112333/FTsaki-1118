"""Fixed-epoch CPU continuation and prospective, paired industrial validation."""
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean

from .data import flatten, release_ids
from .model0_smoke import file_hash, require
from .policy import digest


def finite(value):
    return type(value) in (float, int) and isfinite(value)


def training_snapshot(recent, related, representative, *, released, industrial_train_ids, size):
    """Controller inputs: released is a training-authorized label ledger, not model output."""
    require(type(size) is int and size >= 10 and size % 10 == 0, "Snapshot size must permit 50/30/20 quotas")
    quotas = {"recent": size // 2, "related": size * 3 // 10, "representative": size // 5}
    seen, samples, counts = set(), [], {}
    for source, rows in (("recent", recent), ("related", related), ("representative", representative)):
        chosen = 0
        for row in rows:
            sample_id = row["id"]
            if source == "recent":
                require(sample_id not in industrial_train_ids and sample_id in released
                        and released[sample_id] == row["answer"], "Missing authorized released target")
            else:
                require(sample_id in industrial_train_ids, "Not industrial/train")
            if sample_id in seen or chosen >= quotas[source]:
                continue
            require(row["group"] and row["values"] and all(finite(value) for value in row["values"]), "Invalid training row")
            seen.add(sample_id)
            chosen += 1
            samples.append({key: row[key] for key in ("id", "group", "values", "answer")} | {"source": source})
        counts[source] = chosen
    return {"samples": samples, "counts": counts, "quotas": quotas, "groups": sorted({row["group"] for row in samples}),
            "shortfall": {source: quotas[source] - counts[source] for source in quotas},
            "sha256": digest(samples)}


def continue_cpu(model_type, checkpoint, expected_parent_sha256, destination, snapshot, epochs):
    import numpy as np
    require(type(epochs) is int and epochs > 0, "Explicit fixed epochs required")
    require(file_hash(checkpoint) == expected_parent_sha256, "Parent checkpoint changed")
    require(not Path(destination).exists(), "Candidate already exists")
    samples = snapshot["samples"]
    require(digest(samples) == snapshot["sha256"] and any(row["source"] == "recent" for row in samples),
            "Changed or insufficient training snapshot")
    model = model_type.load(checkpoint)
    start = model.meta["epochs"]
    steps = model.meta["steps"]
    answers = [row["answer"] for row in samples]
    if model.meta["task"] == "classification":
        answers = [model.meta["classes"].index(label) for label in answers]
    else:
        require(all(finite(answer) for answer in answers), "Invalid regression target")
    losses = model.fit(np.asarray([row["values"] for row in samples], dtype=np.float64), np.asarray(answers), epochs)
    require(len(losses) == epochs and all(finite(loss) for loss in losses), "Invalid training loss")
    require(model.meta["epochs"] == start + epochs and model.meta["steps"] > steps, "Optimizer did not advance")
    model.save(destination)
    return {"parent_sha256": expected_parent_sha256, "candidate_sha256": file_hash(destination),
            "snapshot_sha256": snapshot["sha256"], "initial_epochs": start, "final_epochs": model.meta["epochs"],
            "initial_steps": steps, "final_steps": model.meta["steps"], "losses": losses}


class CpuPair:
    def __init__(self, model_type, stable_path, candidate_path, gate):
        import json
        require(file_hash(stable_path) == gate.stable and file_hash(candidate_path) == gate.candidate,
                "Weights do not match frozen comparison")
        self.stable = model_type.load(stable_path)
        self.candidate = model_type.load(candidate_path)
        require(self.stable.meta["paths"] == self.candidate.meta["paths"] and
                tuple(self.stable.meta["classes"]) == tuple(self.candidate.meta["classes"]) == gate.classes,
                "Incompatible industrial models")
        self.order = ["/" + "/".join(str(component[1]) for component in json.loads(path))
                      for path in self.stable.meta["paths"]]
        self.gate = gate

    def predict_batch(self, phase, batch):
        import numpy as np
        vectors = []
        for row in batch:
            order, values = flatten(row["inputs"])
            require(order == self.order, "Industrial feature order mismatch")
            vectors.append(values)
        matrix = np.asarray(vectors, dtype=np.float64)
        def predict(model):
            outputs = model.predict(matrix)
            if self.gate.classes:
                return [self.gate.classes[index] for index in outputs.argmax(axis=1)]
            return outputs.tolist()
        return self.gate.predict_batch(phase, batch, predict(self.stable), predict(self.candidate))


@dataclass(frozen=True)
class IndustrialPolicy:
    min_labels: int
    min_groups: int
    min_classes: int
    min_fraction: float
    min_improvement: float
    rollback_degradation: float

    def __post_init__(self):
        require(all(type(count) is int and count > 0 for count in (self.min_labels, self.min_groups)), "Invalid coverage")
        require(type(self.min_classes) is int and 0 <= self.min_classes <= 6, "Invalid class coverage")
        require(finite(self.min_fraction) and 0 < self.min_fraction <= 1, "Invalid label fraction")
        require(all(finite(value) and value > 0 for value in (self.min_improvement, self.rollback_degradation)),
                "Positive improvement and degradation margins required")


def paired_score(scenario, classes, pairs):
    if not pairs:
        return None, None
    def score(column):
        if scenario != "A":
            return sqrt(fmean((row[column] - row["answer"]) ** 2 for row in pairs))
        def class_f1(label):
            correct = sum(row[column] == row["answer"] == label for row in pairs)
            count = sum(int(row[column] == label) + int(row["answer"] == label) for row in pairs)
            return 2 * correct / count if count else 0
        return fmean(class_f1(label) for label in classes)
    return score("stable"), score("candidate")


class IndustrialGate:
    def __init__(self, store, *, scenario, stable, candidate, trained_ids, ready_at, label_budget,
                 classes=(), policy=None, provenance, handoff_sha256=None):
        require(scenario in ("A", "B", "C") and stable and candidate and stable != candidate, "Invalid versions")
        require(provenance in ("controlled", "closed_loop"), "Unknown evidence source")
        require(type(ready_at) is int and ready_at >= 0 and type(label_budget) is int and label_budget >= 0, "Invalid cutoff or budget")
        require((scenario == "A" and len(set(classes)) == len(classes) == 6) or (scenario != "A" and not classes), "Invalid class order")
        if policy is not None:
            require(isinstance(policy, IndustrialPolicy) and (scenario == "A") == bool(policy.min_classes), "Policy does not match task")
        self.store, self.policy, self.trained = store, policy, frozenset(trained_ids)
        self.scenario, self.classes = scenario, tuple(classes)
        self.stable, self.candidate = stable, candidate
        self.ready_at, self.budget = ready_at, label_budget
        identity = {"mode": "industrial-paired", "provenance": provenance, "scenario": scenario, "classes": list(classes),
                    "stable": stable, "candidate": candidate, "trained_ids_sha256": digest(sorted(self.trained)),
                    "ready_at": ready_at, "label_budget": label_budget, "policy": asdict(policy) if policy else None,
                    "handoff_sha256": handoff_sha256}
        require(not store.export() or store.get("identity") == identity, "Changed industrial experiment identity")
        store.commit({"identity": identity})

    @property
    def active(self):
        if self.store.get("verdict/rollback") or not self.store.get("verdict/release"):
            return self.stable
        return self.candidate if self.store.get("verdict/release")["outcome"] == "published" else self.stable

    def predict_batch(self, phase, batch, stable_outputs, candidate_outputs):
        require(phase in ("release", "rollback"), "Unknown industrial phase")
        require(batch and len(batch) == len(stable_outputs) == len(candidate_outputs), "Unpaired predictions")
        entries = {}
        for row, stable, candidate in zip(batch, stable_outputs, candidate_outputs):
            sample_id = row["id"]
            require(row["group_id"] == batch[0]["group_id"] and sample_id not in self.trained, "Nonprospective sample")
            require(not {"answer", "target", "label"}.intersection(row), "Observation contains feedback")
            for value in (stable, candidate):
                require(value in self.classes if self.scenario == "A" else finite(value), "Invalid industrial output")
            key = "pair/" + sample_id
            require(key not in entries, "Duplicate sample ID")
            entries[key] = {"id": sample_id, "group": row["group_id"], "step": row["replay_order"],
                            "stable": stable, "candidate": candidate, "phase": phase}
        ordinal = batch[0]["replay_order"]
        require(type(ordinal) is int and ordinal > self.ready_at, "Sample not after candidate readiness")
        require([row["replay_order"] for row in batch] == list(range(ordinal, ordinal + len(batch))), "Noncontiguous batch")
        batch_key = "batch/" + str(ordinal)
        entries[batch_key] = {"ids": [row["id"] for row in batch], "phase": phase}
        if self.store.get(batch_key) is not None:
            require(all(self.store.get(key) == item for key, item in entries.items()), "Changed saved batch")
            return ordinal
        require(self.store.get("verdict/" + phase) is None and (phase == "release" or self.active == self.candidate),
                "Industrial phase not open")
        last = max((row["step"] for key, row in self.store.export().items() if key.startswith("pair/")), default=self.ready_at)
        require(ordinal == last + 1, "Skipped or reordered prospective observations")
        self.store.commit(entries)
        return ordinal

    def release_feedback(self, batch_start, provider):
        key = "release/" + str(batch_start)
        existing = self.store.get(key)
        if existing is not None:
            return existing
        batch = self.store.get("batch/" + str(batch_start))
        require(batch is not None, "Save paired predictions before requesting feedback")
        pairs = [self.store.get("pair/" + sample_id) for sample_id in batch["ids"]]
        saved = [{"id": pair["id"], "group_id": pair["group"]} for pair in pairs]
        spent = sum(len(item) for key, item in self.store.export().items() if key.startswith("release/"))
        remaining = self.budget - spent
        answers = provider.release(saved, set(batch["ids"]), self.scenario, remaining) if remaining else {}
        allowed = release_ids(batch["ids"], self.scenario, pairs[0]["group"], remaining)
        require(set(answers) <= set(allowed),
                "Feedback outside sample or quota")
        for answer in answers.values():
            require(answer in self.classes if self.scenario == "A" else finite(answer), "Invalid feedback target")
        self.store.commit({key: answers})
        return answers

    def evaluate(self, phase):
        require(phase in ("release", "rollback"), "Unknown industrial phase")
        previous = self.store.get("verdict/" + phase)
        if previous is not None:
            return previous
        require(phase == "release" or self.active == self.candidate, "No published candidate to monitor")
        records = self.store.export()
        pairs = {item["id"]: item for key, item in records.items() if key.startswith("pair/") and item["phase"] == phase}
        labeled = [dict(pairs[sample_id], answer=answer) for key, item in records.items() if key.startswith("release/")
                   for sample_id, answer in item.items() if sample_id in pairs]
        stable, candidate = paired_score(self.scenario, self.classes, labeled)
        groups = len({item["group"] for item in labeled})
        counts = Counter(item["answer"] for item in labeled) if self.classes else {}
        coverage = len(labeled) / len(pairs) if pairs else 0
        policy = self.policy
        report = {"phase": phase, "labels": len(labeled), "groups": groups, "class_counts": dict(counts),
                  "coverage": coverage, "predictions": len(pairs), "stable_score": stable, "candidate_score": candidate,
                  "metric": {"A": "macro_f1", "B": "rmse_minutes", "C": "rmse_cycles"}[self.scenario]}
        if (not policy or len(labeled) < policy.min_labels or groups < policy.min_groups
                or len(counts) < policy.min_classes or coverage < policy.min_fraction):
            return dict(report, outcome="insufficient", reason="coverage" if policy else "policy-missing")
        gain = candidate - stable if self.scenario == "A" else stable - candidate
        report["gain"] = gain
        if phase == "rollback" and gain > -policy.rollback_degradation:
            return dict(report, outcome="retain")
        report["outcome"] = ("published" if gain >= policy.min_improvement else "rejected") if phase == "release" else "rolled-back"
        self.store.commit({"verdict/" + phase: report})
        return report
