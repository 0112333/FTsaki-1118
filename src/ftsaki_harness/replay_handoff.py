"""Bind the live 2a review to the exact causal boundary used by stage 3."""
from collections import deque
from itertools import islice

from .data import flatten
from .forecast import matrix
from .policy import digest
from .pool import Pool
from .prospective import require
from .training_intent import reviewed_decision_plan


def replay_group(scenario, source_group):
    return "g-" + digest(["anonymous.v1", scenario, source_group])[:16]


def prospective_sample(scenario, feature_order, observation):
    order, row = flatten(observation["inputs"])
    require(order == feature_order, "Observation feature order changed")
    return {"group":replay_group(scenario, observation["group_id"]), "order":observation["replay_order"],
            "sample_id":observation["id"], "row":row}


def reviewed_anchor(review, observations, *, manifest, scenario, feature_order, run):
    identity = review.get("identity")
    plan = review.get("replay-plan")
    require(identity and plan and identity["source"] == plan["source"] == "live-http"
            and identity["mode"] == "2a-no-training" and identity["run"] == plan["run"]
            and identity["scenario"] == plan["scenario"] == scenario and plan["manifest"] == manifest
            and plan["parent_authorized"] is True, "Live authorized review and matching manifest required")
    rows = [row for name, row in review.export().items() if name.startswith("vote/")]
    require(rows and len(rows) <= len(plan["windows"]), "No valid review prefix")
    pool = Pool()
    for row, planned in zip(rows, plan["windows"]):
        cutoff = row["cutoff"]
        receipt = review.get(f"call-result/{cutoff}")
        require(cutoff == planned["cutoff"] and receipt and receipt["http_status"] == 200
                and (receipt.get("format_valid") is True or row["failure"] == "invalid-response")
                and receipt["expected_model"] == identity["model"]
                and receipt["provider_model"] in (identity["model"], plan["checkpoint_id"] + "@0"),
                "Review vote lacks its verified request receipt")
        computed = pool.window(row["window_id"], row["group"], {0:row["vote"]})
        require(computed == row["pool"], "Review pool trajectory changed")
        require(not computed["trigger"] or review.get("plan/" + row["event_id"]), "Review paused at an unresolved event")
    latest = rows[-1]
    cutoff = latest["cutoff"]
    tail = deque(maxlen=32)
    previous_group = None
    arrived = 0
    for arriving, observation in enumerate(islice(observations, cutoff), start=1):
        sample = prospective_sample(scenario, feature_order, observation)
        require(sample["order"] == arriving, "Changed arrival order")
        if sample["group"] != previous_group:
            tail.clear()
            previous_group = sample["group"]
        tail.append(sample["row"])
        arrived = arriving
    require(arrived == cutoff and tail and previous_group == latest["group"], "Handoff window is missing or has wrong group")
    history = matrix(list(tail), len(feature_order))
    return {"run":run, "scenario":scenario, "source":"closed_loop", "manifest":manifest, "cursor":cutoff,
            "parent_artifact":identity["model"], "group":latest["group"], "history":history,
            "pool":dict(pool.dump(), windows={}), "review_run":identity["run"], "review_cutoff_sha256":digest(latest)}


def model1_candidate(stream, anchor, review, training, event):
    require(stream.identity["source"] == "closed_loop" and digest(anchor) == stream.identity["anchor_sha256"],
            "Stage 3 requires its original reviewed stream boundary")
    original = anchor["review_run"]
    suffix = f"{original}/{event}"
    intent = training.get("intent/" + suffix)
    handle = training.get("handle/" + suffix)
    job = training.get("submission-result/" + suffix)
    reload = training.get("reload-result/" + suffix)
    require(intent and handle and job and reload and job["status"] == "artifact-ready"
            and reload["status"] == "model1-inference-confirmed", "Completed training and verified child reload required")
    parent = stream.identity["parent_artifact"]
    require(intent["run"] == original and intent["event"] == event and intent["scenario"] == stream.identity["scenario"]
            and intent["parent"]["version"] == 0 and intent["parent"]["model"] == parent == job["parent"] == reload["parent"],
            "Child artifact belongs to another parent, scenario or event")
    artifact = f"trio://{handle}/sampler_weights/stage2b-model1"
    provider_key = job["checkpoint_id"]
    receipt = reload["receipt"]
    require(job["handle"] == reload["handle"] == handle and job["artifact"] == reload["artifact"] == artifact
            and artifact != parent and job["model_id"] == artifact and provider_key == reload["checkpoint_id"]
            and receipt["http_status"] == 200 and receipt["format_valid"] is True and receipt["expected_model"] == artifact
            and receipt["provider_model"] in (artifact, provider_key + "@0") and receipt.get("response_sha256"),
            "Child reload identity or response receipt mismatch")
    plan, supplement = reviewed_decision_plan(review, event)
    window = review.get("vote/" + plan["window_id"])
    require(window and window["event_id"] == event and window["pool"]["trigger"]
            and intent["plan_sha256"] == digest(plan) and intent["window_sha256"] == digest(window)
            and intent.get("controller_authorization_sha256") == (digest(supplement) if supplement else None),
            "Training event differs from the verified review")
    trained_through = intent["snapshot_cutoff"]
    require(type(trained_through) is int and window["cutoff"] <= trained_through <= anchor["cursor"] <= stream.cursor,
            "Verification cannot reuse the training interval")
    return {"version":1, "run":stream.identity["run"], "source":"closed_loop", "scenario":stream.identity["scenario"],
            "event":event, "artifact":artifact, "parent":0, "parent_artifact":parent, "checkpoint_id":provider_key,
            "trained_through":trained_through, "snapshot_sha256":intent["snapshot_sha256"], "ready_at":stream.cursor,
            "ready_group":stream.state["group"], "training_receipt_sha256":digest(job), "reload_receipt_sha256":digest(reload)}
