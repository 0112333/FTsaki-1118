"""Freeze the prerequisites for one stage-2b decision training submission."""
from datetime import datetime, timezone

from .policy import canonical, digest


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def timestamp(value):
    value = datetime.fromisoformat(value)
    require(value.utcoffset() is not None, "Deadline requires timezone")
    return value


def reviewed_decision_plan(review_store, event):
    """Keep model attribution immutable; bind an explicitly approved experiment separately."""
    plan = review_store.get("plan/" + event)
    require(plan and plan["source"] == "live-http", "Reviewed live event required")
    supplement = review_store.get("decision-authorization/" + event)
    if "decision" not in plan["reviews"]:
        identity = review_store.get("identity")
        require(supplement and supplement.get("approval")
                and supplement.get("purpose") == "decision-connectivity-experiment"
                and supplement.get("event") == event
                and supplement.get("original_plan_sha256") == digest(plan)
                and supplement.get("parent_model") == plan["parent_model"]
                and supplement.get("scenario") == identity["scenario"]
                and supplement.get("training_objects") == ["decision"],
                "Reviewed decision event or explicit controller experiment authorization required")
    return plan, supplement


def freeze(review_store, event, parent, method, snapshot, authorized_cases, budget, *, now=None):
    identity = review_store.get("identity")
    require(identity and identity.get("mode") == "2a-no-training" and identity.get("source") == "live-http",
            "Live stage 2a review required")
    plan, supplement = reviewed_decision_plan(review_store, event)
    window = review_store.get("vote/" + plan["window_id"])
    require(window and window["event_id"] == event and window["pool"]["trigger"] and window["cutoff"] == plan["cutoff"],
            "Event does not match triggering window")
    require(parent.get("authorized") is True and parent.get("eligible") is True and parent.get("checkpoint")
            and parent.get("model") == plan["parent_model"] == identity["model"]
            and parent.get("version") == plan["parent_version"], "Approved stable parent and checkpoint required")
    require(method.get("approval") and method.get("algorithm") and method.get("parameters")
            and method.get("target") in ("decide", "forecast_next")
            and type(method.get("minimum_cases")) is int and method["minimum_cases"] > 0, "Approved method and sample minimum required")
    require(budget.get("approval") and budget.get("workspace") and budget.get("service") and budget.get("deadline")
            and type(budget.get("max_jobs")) is int and budget["max_jobs"] == 1
            and type(budget.get("max_sequence_tokens")) is int
            and budget["max_sequence_tokens"] > 0 and type(budget.get("max_updates")) is int and budget["max_updates"] > 0,
            "Approved provider and bounded budget required")
    require((now or datetime.now(timezone.utc)) < timestamp(budget["deadline"]), "Training deadline elapsed")
    require(snapshot.get("authorization") and type(snapshot.get("cutoff")) is int and snapshot["cutoff"] >= plan["cutoff"],
            "Authorized snapshot cutoff required")
    boundary = review_store.get(f"vote/{snapshot['cutoff']:012d}")
    require(boundary and boundary["cutoff"] == snapshot["cutoff"], "Snapshot cutoff has not arrived in review stream")
    cases = snapshot.get("cases", [])
    require(len(cases) >= method["minimum_cases"], "Insufficient qualified samples")
    require(snapshot.get("sha256") == digest(cases), "Snapshot digest mismatch")
    require(len({case["id"] for case in cases}) == len(cases), "Duplicate training case")
    ceiling = 0
    sources = {"decide": "independent_annotation", "forecast_next": "observed_future_features"}
    for case in cases:
        require(authorized_cases.get(case["id"]) == digest(case), "Training sample not authorized")
        require(case["scenario"] == identity["scenario"] and case["task"] == method["target"]
                and case["supervision_source"] == sources[case["task"]], "Unsupported supervision source")
        require(case["input_sha256"] and case["target_sha256"]
                and type(case["input_cutoff"]) is int and type(case["supervision_cutoff"]) is int
                and 0 < case["input_cutoff"] <= case["supervision_cutoff"] <= snapshot["cutoff"],
                "Input or available supervision cutoff invalid")
        if case["task"] == "forecast_next":
            require(case["supervision_cutoff"] == case["input_cutoff"] + 32, "Incomplete forecast supervision")
        count = case["sequence_tokens"]
        require(type(count) is int and count > 0, "Missing frozen token count")
        ceiling += count
    require(ceiling <= budget["max_sequence_tokens"], "Training snapshot exceeds token budget")
    return {"run": identity["run"], "scenario": identity["scenario"], "event": event,
            "window_sha256": digest(window), "plan_sha256": digest(plan), "parent": parent,
            "controller_authorization_sha256": digest(supplement) if supplement else None,
            "method": method, "snapshot_sha256": snapshot["sha256"], "snapshot_cutoff": snapshot["cutoff"],
            "snapshot_authorization": snapshot["authorization"], "cases": cases, "budget": budget,
            "sequence_tokens": ceiling}


class TrainingIntent:
    def __init__(self, store, review_store):
        self.store, self.review_store = store, review_store

    def reserve(self, event, parent, method, snapshot, authorized_cases, budget, *, now=None):
        frozen = freeze(self.review_store, event, parent, method, snapshot, authorized_cases, budget, now=now)
        key = f"intent/{frozen['run']}/{event}"
        with self.store.db:
            self.store.db.execute("BEGIN IMMEDIATE")
            existing = self.store.get(key)
            if existing is not None:
                require(existing == frozen, "Frozen submission changed")
                return False
            require(not self.store.db.execute("SELECT 1 FROM records WHERE key GLOB 'intent/*' LIMIT 1").fetchone(),
                    "Stage 2b database already reserved a job")
            self.store.db.execute("INSERT INTO records VALUES (?,?)", (key, canonical(frozen)))
        return True

    def intent(self, run, event):
        pending = self.store.get(f"intent/{run}/{event}")
        require(pending is not None, "Submission intent missing")
        return pending

    def record_handle(self, run, event, handle):
        self.intent(run, event)
        require(isinstance(handle, str) and handle, "Provider handle required")
        self.store.commit({f"handle/{run}/{event}": handle})

    def reconcile(self, run, event):
        self.intent(run, event)
        handle = self.store.get(f"handle/{run}/{event}")
        return {"status": "query-existing" if handle else "awaiting-reconciliation", "handle": handle}
