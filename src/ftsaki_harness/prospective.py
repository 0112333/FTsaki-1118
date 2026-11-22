"""Advance a single scenario cursor for voting and already-frozen candidate forecasts."""
from dataclasses import asdict
import json

from .candidates import Candidates
from .forecast import Forecast, matrix
from .inference import InferenceError, validate_content
from .policy import Policy, digest
from .pool import Pool


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class ProspectiveRun:
    def __init__(self, records, *, run, scenario, source, manifest, feature_order, scale, start, parent_artifact, anchor=None):
        require(scenario in ("A", "B", "C") and source in ("controlled", "closed_loop"), "Unknown replay")
        require(run and manifest and parent_artifact and type(start) is int and start >= 0, "Incomplete replay identity")
        require(len(feature_order) == len(set(feature_order)) == len(scale.mean) == {"A":128,"B":34,"C":24}[scenario],
                "Invalid scenario feature order")
        self.records, self.scale, self.candidates = records, scale, Candidates(records)
        initial = {"cursor": 0, "group": None, "history": [], "pool": Pool().dump()}
        if start:
            require(anchor and anchor["cursor"] == start and anchor["scenario"] == scenario and anchor["manifest"] == manifest
                    and anchor["parent_artifact"] == parent_artifact and anchor["source"] == source and anchor["run"] == run,
                    "Missing or mismatched continuation checkpoint")
            pool = Pool.restore(dict(anchor["pool"], windows={}))
            require(len(pool.models) == 1 and 0 in pool.weights() and pool.models[0].eligible and pool.models[0].stable
                    and pool.group == anchor["group"] and 0 < len(anchor["history"]) <= 32, "Invalid 2a handoff")
            initial = {"cursor": start, "group": pool.group, "history": matrix(anchor["history"], len(scale.mean)),
                       "pool": dict(pool.dump(), windows={})}
        else:
            require(anchor is None, "Unexpected handoff at origin")
        self.identity = {"run": run, "scenario": scenario, "source": source, "manifest": manifest, "start": start,
                         "feature_order": list(feature_order), "scale_sha256": digest(asdict(scale)),
                         "parent_artifact": parent_artifact, "anchor_sha256": digest(anchor), "policy_sha256": Policy().sha256}
        existing = records.db.execute("SELECT 1 FROM records LIMIT 1").fetchone()
        require(not existing or self.identity == records.get("identity"), "Changed run identity")
        records.commit({"identity": self.identity})
        last = records.db.execute("SELECT key FROM records WHERE key GLOB 'step/*' ORDER BY key DESC LIMIT 1").fetchone()
        self.state = records.get(last[0]) if last else initial

    @property
    def cursor(self):
        return self.state["cursor"]

    def _versions(self):
        return [int(key.split("/")[1]) for key, in self.records.db.execute("SELECT key FROM records WHERE key GLOB 'ready/*'")]

    def _check_ready(self, receipt):
        version = receipt["version"]
        require(type(version) is int and version > 0, "Invalid candidate version")
        cutoff, scenario = self.cursor, self.identity["scenario"]
        require(not self.records.get("end") and self.state["history"], "No live history at the boundary")
        require(receipt["ready_at"] == cutoff and type(receipt["trained_through"]) is int and 0 <= receipt["trained_through"] <= cutoff and
                receipt["ready_group"] == self.state["group"], "Candidate not ready at this stream boundary")
        require(receipt["scenario"] == scenario and receipt["source"] == self.identity["source"] and
                receipt["run"] == self.identity["run"], "Candidate belongs to another run")
        require(all(receipt[key] for key in ("artifact", "event", "snapshot_sha256", "parent_artifact")) and
                receipt["artifact"] != receipt["parent_artifact"], "Missing or unchanged training artifact")
        parent = receipt["parent"]
        pool = Pool.restore(self.state["pool"])
        model = pool.models.get(parent)
        require(model and model.stable and model.online and model.eligible, "No eligible online stable parent")
        expected_parent = self.identity["parent_artifact"] if parent == 0 else self.records.get(f"ready/{parent}")["artifact"]
        require(receipt["parent_artifact"] == expected_parent, "Wrong parent artifact")
        require(not self.records.get(f"step-intent/{cutoff + 1}") and not self.records.get("end-intent"),
                "Next observations or end-of-stream already recorded")
        previous = self.records.get(f"ready/{version}")
        require((previous == receipt) if previous else version > max([0] + self._versions()), "Changed or reused candidate")
        earlier = [self.records.get(f"ready/{item}") for item in self._versions() if item < version]
        same_event = [item for item in earlier if item["event"] == receipt["event"]]
        if same_event:
            failed = max(same_event, key=lambda item: item["version"])
            intent = self.records.get(f"retry/{failed['version']}")
            require(intent and intent["parent"] == parent and intent["snapshot_sha256"] == receipt["snapshot_sha256"]
                    and receipt["ready_group"] == failed["ready_group"] and receipt["trained_through"] == intent["data_cutoff"]
                    and cutoff >= intent["reserved_at"],
                    "Retry not bound to a new authorized snapshot")
        return version

    def freeze(self, receipt, prediction, order):
        version = self._check_ready(receipt)
        require(list(order) == self.identity["feature_order"], "Wrong feature order")
        require(not self.records.get(f"finish/{version}"), "Candidate already terminated")
        forecast, plan = self._frozen_forecast(version, prediction)
        self.records.commit({f"ready/{version}":receipt, f"forecast/{version}":forecast.dump()})
        return plan

    def _frozen_forecast(self, version, prediction):
        forecast = Forecast(str(version), self.identity["scenario"], self.state["group"], self.cursor, prediction,
                            self.state["history"], self.scale)
        return forecast, {"version":version, "group":forecast.group, "first":self.cursor + 1,
                          "last":self.cursor + 32, "frozen_hash":forecast.frozen_hash}

    def accept_forecast_response(self, receipt, content=None, stop=None, *, error=None):
        version = receipt["version"]
        require(error in (None, "transport-error", "timeout") and (content is None or isinstance(content, str)),
                "Invalid response envelope")
        key = f"forecast-response/{version}"
        signature = digest([receipt, content, stop, error])
        previous = self.records.get(key)
        if previous:
            require(previous["signature"] == signature, "Candidate response already recorded")
            return previous["result"]
        self._check_ready(receipt)
        require(not self.records.get(f"forecast/{version}") and not self.records.get(f"finish/{version}"),
                "Candidate response already recorded")
        expected = self.identity["feature_order"]
        parsed = None
        failure = None
        if error:
            failure = {"outcome":error, "reason":error}
        else:
            try:
                parsed = validate_content("forecast_next", {"feature_order":expected}, content, stop)
            except (ValueError, TypeError, KeyError, ArithmeticError, InferenceError, RecursionError) as exc:
                failure = {"outcome":"invalid-output", "reason":str(exc)[:160]}
        entries = {f"ready/{version}":receipt}
        if failure:
            failure.update(expected_shape=[32,len(expected)], stop=stop,
                           content_sha256=digest(content) if content is not None else None)
            try:
                payload = json.loads(content)
            except (ValueError, TypeError, RecursionError):
                payload = None
            values = payload.get("values") if isinstance(payload, dict) else None
            if isinstance(values, list):
                failure["actual_rows"] = len(values)
                failure["actual_widths"] = sorted({len(row) for row in values if isinstance(row, list)})
            result = {"validation":failure, "admission":"not-admitted", "resolved_at":self.cursor}
            entries[f"finish/{version}"] = result
        else:
            forecast, result = self._frozen_forecast(version, parsed["values"])
            entries[f"forecast/{version}"] = forecast.dump()
        entries[key] = {"signature":signature,"result":result}
        self.records.commit(entries)
        return result

    def advance(self, batch, votes):
        require(bool(batch), "Empty batch")
        first, last, group = batch[0]["order"], batch[-1]["order"], batch[0]["group"]
        require(type(first) is int and type(last) is int and first >= 1 and
                [row["order"] for row in batch] == list(range(first, first + len(batch))), "Nonconsecutive batch")
        require(all(sample["group"] == group for sample in batch) and
                len(batch) <= Policy().batch_size(self.identity["scenario"]), "Mixed group or oversize batch")
        signature = digest([batch, {str(version): value for version, value in votes.items()}])
        step_key = f"step/{last:012d}"
        prior = self.records.get(step_key)
        if prior:
            require(prior["signature"] == signature, "Previously processed batch changed")
            return self.records.get(f"result/{last}")
        require(not self.records.get("end-intent") and first == self.cursor + 1, "Stream ended, skipped or rewound")
        ids = [sample["sample_id"] for sample in batch]
        require(len(set(ids)) == len(ids) and all(not self.records.get("seen/" + sid) for sid in ids),
                "Repeated source position")
        pool = Pool.restore(dict(self.state["pool"], windows={}))
        require(all(type(version) is int and version in pool.weights() for version in votes), "Vote from nonmember")
        require(all(value in ("drift","no_drift","insufficient") for value in votes.values()), "Unknown vote")
        self.records.commit({f"step-intent/{first}": {"signature": signature, "last": last}})
        vote = pool.window(str(last), group, votes)
        changes, commits = {}, {}
        for version in sorted(self._versions()):
            if self.records.get(f"finish/{version}"):
                continue
            outcome = self.candidates.advance(str(version), batch)
            if outcome is None:
                continue
            receipt = self.records.get(f"ready/{version}")
            admission = pool.admit(version, receipt["parent"]) if outcome["outcome"] == "passed" else "not-admitted"
            changes[str(version)] = {"validation": outcome, "admission": admission, "resolved_at": last}
            if admission != "waiting":
                commits[f"finish/{version}"] = changes[str(version)]
        history = list(self.state["history"]) if self.state["group"] == group else []
        for sample in batch:
            try:
                history.extend(matrix([sample["row"]],len(self.scale.mean)))
            except (TypeError, ValueError):
                history.clear()
        pool_state = dict(pool.dump(), windows={})
        state = {"cursor": last, "group": group, "history": history[-32:], "pool": pool_state, "signature": signature}
        result = {"start": first, "end": last, "group": group, "vote": vote, "candidates": changes}
        self.records.commit({step_key: state, f"result/{last}": result, **commits,
                             **{"seen/" + sample["sample_id"]: {"order": sample["order"], "group": group} for sample in batch}})
        self.state = state
        return result

    def reserve_retry(self, failed_version, *, new_ids, authorized_ids, failure_use_authorized,
                      budget_available, before_deadline, eligible_parents):
        ready = self.records.get(f"ready/{failed_version}")
        finish = self.records.get(f"finish/{failed_version}")
        require(ready and finish and finish["validation"]["outcome"] == "rejected", "No failed forward validation")
        require(failure_use_authorized is True and budget_available is True and before_deadline is True,
                "Retry lacks approved failure material, budget or time")
        require(new_ids and len(set(new_ids)) == len(new_ids) and set(new_ids) <= set(authorized_ids),
                "Retry needs distinct authorized new data")
        previous = self.records.get(f"retry/{failed_version}")
        if previous:
            require(previous["new_ids"] == sorted(new_ids) and previous["parent"] in eligible_parents, "Retry intent changed")
            return previous
        require(not self.records.get("end-intent"), "Stream no longer available for a new attempt")
        attempts = [self.records.get(f"ready/{item}") for item in self._versions()]
        attempts = sorted((item for item in attempts if item["event"] == ready["event"]), key=lambda item: item["version"])
        require(attempts[-1]["version"] == failed_version and len(attempts) < 3, "Event retry limit or superseded failure")
        used = {sid for item in attempts for sid in (self.records.get(f"retry/{item['version']}") or {}).get("new_ids", [])}
        require(not used.intersection(new_ids), "New data already used for retry")
        for sample_id in new_ids:
            sample = self.records.get("seen/" + sample_id)
            require(sample and sample["group"] == ready["ready_group"] and
                    ready["trained_through"] < sample["order"] <= self.cursor, "Not same-source newly arrived data")
        parent = ready["parent"]
        models = Pool.restore(self.state["pool"]).models
        while parent is not None:
            model = models[parent]
            if model.stable and model.eligible and parent in eligible_parents:
                break
            parent = model.parent
        require(parent is not None, "No eligible stable ancestor")
        parent_artifact = self.identity["parent_artifact"] if parent == 0 else self.records.get(f"ready/{parent}")["artifact"]
        material = {"event": ready["event"], "parent": parent, "parent_artifact": parent_artifact,
                    "prior_snapshot": ready["snapshot_sha256"], "failed_version": failed_version,
                    "failure_sha256": digest(finish), "new_ids": sorted(new_ids), "data_cutoff": self.cursor}
        intent = dict(material, snapshot_sha256=digest(material), reserved_at=self.cursor, attempt=len(attempts) + 1)
        self.records.commit({f"retry/{failed_version}": intent})
        return intent

    def end(self):
        prior = self.records.get("end")
        if prior:
            return prior
        require(not self.records.get(f"step-intent/{self.cursor + 1}"), "Recover incomplete step before closing")
        self.records.commit({"end-intent": {"cursor": self.cursor}})
        unfinished = {}
        for version in self._versions():
            if not self.records.get(f"finish/{version}"):
                outcome = self.candidates.end(str(version))
                unfinished[str(version)] = {"validation": outcome, "admission": "not-admitted", "resolved_at": self.cursor}
        report = {"cursor": self.cursor, "unfinished": unfinished, "source": self.identity["source"]}
        self.records.commit({"end": report, **{f"finish/{version}": entry for version, entry in unfinished.items()}})
        return report
