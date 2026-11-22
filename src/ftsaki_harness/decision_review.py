"""Stage 2a: record model0 votes and review plans without a training executor."""
from .inference import validate_content, validate_hypotheses, InferenceError
from .policy import digest
from .pool import Pool


REQUIRED = {
    "decision": ("parent_eligible", "snapshot_authorized", "method_configured", "supervision_qualified",
                 "sample_threshold_met", "service_ready", "budget_reserved", "before_deadline"),
    "industrial": ("checkpoint_compatible", "snapshot_authorized", "released_training_labels", "sample_threshold_met",
                   "service_ready", "budget_reserved", "before_deadline"),
}
SOURCES = {"decision": {"causal_case_history", "authorized_outcomes"},
           "industrial": {"industrial_train", "released_closed_loop"}}


def validate_plan(proposal, visible_refs):
    if not isinstance(proposal, dict):
        raise ValueError("Plan must be a JSON object")
    refs = proposal.get("evidence_refs")
    objects = proposal.get("training_objects")
    if not isinstance(objects, list) or not objects or not all(isinstance(obj, str) and obj in REQUIRED for obj in objects):
        raise ValueError("Unknown training object")
    if len(set(objects)) != len(objects) or not isinstance(proposal.get("sources"), dict):
        raise ValueError("Invalid training objects or sources")
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) and ref in visible_refs for ref in refs):
        raise ValueError("Plan must cite visible evidence")
    for field in ("hypothesis", "uncertainty", "budget_suggestion"):
        if not isinstance(proposal.get(field), str) or not proposal[field].strip():
            raise ValueError("Missing plan field: " + field)
    return objects


class DecisionReview:
    def __init__(self, store, *, run, scenario, source, model, parent_authorized):
        if source not in ("controlled", "live-http") or parent_authorized is not True:
            raise ValueError("Explicit source and experimental stable parent required")
        if not run or not model or scenario not in ("A", "B", "C"):
            raise ValueError("Incomplete review identity")
        identity = {"run": run, "scenario": scenario, "source": source, "model": model,
                    "mode": "2a-no-training", "parent_version": 0}
        existing = store.export()
        if existing and existing.get("identity") != identity:
            raise ValueError("Unrelated review database")
        store.commit({"identity": identity})
        self.store, self.identity = store, identity
        self.pool = Pool()
        self.event = None
        self.cursor = -1
        for key in sorted(key for key in existing if key.startswith("vote/")):
            row = existing[key]
            self.pool.window(row["window_id"], row["group"], {0: row["vote"]})
            self.event = row["event_id"]
            self.cursor = row["cutoff"]

    def accept_window(self, record, evidence, messages, **response):
        cutoff = evidence["as_of"]["replay_order"]
        if record["source"] != "closed_loop" or record["as_of"] != evidence["as_of"] or record["group"] != evidence["group_id"]:
            raise ValueError("Evidence identity mismatch")
        if digest(messages) != record["messages_sha256"] or record["evidence_sha256"] != digest(evidence):
            raise ValueError("Evidence digest mismatch")
        if cutoff > self.cursor and int(record["window_id"]) != max(0, self.cursor):
            raise ValueError("Skipped intervening window")
        return self.window(evidence, **response)

    def window(self, evidence, *, content=None, stop=None, failure=None):
        group = evidence["group_id"]
        cutoff = evidence["as_of"]["replay_order"]
        if evidence["scenario_id"] != self.identity["scenario"] or type(cutoff) is not int or cutoff < 0:
            raise ValueError("Invalid review window")
        window = f"{cutoff:012d}"
        key = "vote/" + window
        signature = digest([evidence, content, stop, failure])
        prior = self.store.get(key)
        if prior is not None:
            if prior["signature"] != signature:
                raise ValueError("Changed replayed window")
            return prior
        if cutoff <= self.cursor:
            raise ValueError("Out-of-order review")
        reply = None
        if failure is None:
            try:
                reply = validate_content("decide", evidence, content, stop)
                validate_hypotheses(reply, evidence)
            except (TypeError, ValueError, InferenceError):
                reply = None
                failure = "invalid-response"
        vote = reply["decision"] if reply is not None else "insufficient"
        if group != self.pool.group:
            self.event = None
        result = self.pool.window(window, group, {0: vote})
        if result["trigger"]:
            self.event = "event-" + digest([self.identity, group, cutoff])[:20]
        if not self.pool.event_open:
            self.event = None
        row = {"window_id": window, "group": group, "cutoff": cutoff, "signature": signature,
               "input_sha256": digest(evidence), "vote": vote, "failure": failure,
               "cited": reply["evidence_refs"] if reply else [], "visible_refs": evidence["evidence_refs"],
               "decision_output": reply, "pool": result, "event_id": self.event,
               "training_submissions": {"decision": 0, "industrial": 0}}
        self.store.commit({key: row})
        self.cursor = cutoff
        return row

    def review_plan(self, window_id, proposal, controller_checks):
        window = self.store.get("vote/" + window_id)
        if not window or not window["pool"]["trigger"] or not window["event_id"]:
            raise ValueError("No triggering window")
        objects = validate_plan(proposal, window["visible_refs"])
        reviews = {}
        for obj in objects:
            checks = controller_checks.get(obj, {})
            missing = [name for name in REQUIRED[obj] if checks.get(name) is not True]
            sources = proposal.get("sources", {}).get(obj)
            if not isinstance(sources, list) or not sources or not all(isinstance(item, str) and item in SOURCES[obj] for item in sources):
                missing.append("unauthorized-data-selection")
            reviews[obj] = {"missing": missing, "status": "blocked" if missing else "eligible-not-submitted"}
        record = {"event_id": window["event_id"], "window_id": window_id, "source": self.identity["source"],
                  "parent_model": self.identity["model"], "parent_version": 0, "cutoff": window["cutoff"],
                  "proposal": proposal, "controller_checks_sha256": digest(controller_checks),
                  "reviews": reviews, "training_submissions": {"decision": 0, "industrial": 0}}
        self.store.commit({"plan/" + window["event_id"]: record})
        return record
