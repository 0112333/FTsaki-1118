"""Causal closed-loop evidence using the frozen model0 producer and CPU checkpoint."""
import argparse
import importlib.util
import json
from itertools import islice
from pathlib import Path
import sys

from .data import Handoff, Replay, read_jsonl
from .model0_smoke import file_hash, read, require
from .policy import Policy, canonical, digest
from .records import Records


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime(data, training, scenario, contract_digest):
    handoff = Handoff(data / "harness_data_v1/manifest.json")
    contract_path = training / "artifacts/preflight/a_forecast_v2/frozen-contract.json"
    require(file_hash(contract_path) == contract_digest, "Model0 contract changed")
    contract = read(contract_path)
    for name, expected in contract["files"].items():
        require(file_hash(training / name) == expected, "Frozen asset changed: " + name)
    manifest_path = data / "model0_episodes_v1_1/manifest.json"
    require(file_hash(manifest_path) == contract["manifest_sha256"], "Producer manifest changed")
    producer_manifest = read(manifest_path)
    require(producer_manifest["upstream_manifest"]["sha256"] == handoff.sha256, "Handoff changed")
    descriptor, = [row for row in producer_manifest["tooling"] if row["path"] == "model0_data.py"]
    require(file_hash(data / descriptor["path"]) == descriptor["sha256"], "Evidence producer changed")
    producer = module_from_file("model0_evidence_producer", data / descriptor["path"])
    entry = handoff.scenario(scenario)
    checkpoint = handoff.descriptor(entry["initial_checkpoint"])
    model_file = (data / entry["runtime_code_relative_to_source_root"] / "model.py").resolve()
    require(model_file.is_relative_to(data.resolve()), "Industrial source outside handoff project")
    require(file_hash(model_file) == "325026d80747eb36f39c8df1be02b640ae21b3cd082827c66aacbdf273dd61c5",
            "Unverified industrial runtime")
    model = module_from_file("industrial_runtime", model_file).Model.load(checkpoint)
    train = list(read_jsonl(handoff.descriptor(entry["splits"]["industrial_train"]["dataset"])))
    order = [name for name, _ in producer.flatten(train[0]["inputs"])]
    expected_order = ["/" + "/".join(str(part[1]) for part in json.loads(path)) for path in model.meta["paths"]]
    require(order == expected_order and len(order) == entry["input_features"], "Checkpoint feature order mismatch")
    reference = producer.reference_rows(scenario, train)
    context = {"s": scenario, "order": order, "model": model, "checkpoint": checkpoint,
               "reference": producer.population([producer.vector(row) for row in reference]),
               "scale": producer.population([producer.vector(row) for row in train])}
    observations = handoff.descriptor(entry["splits"]["closed_loop"]["observations"])
    sys.path.insert(0, str(training / "scripts"))
    from model0_runtime import runtime_messages
    return producer, context, Replay(observations, Policy().batch_size(scenario)), runtime_messages, handoff.sha256


def window_stream(store, producer, context, replay, message_builder, *, label_provider=None, label_budget=0):
    import numpy as np
    require(type(label_budget) is int and label_budget >= 0, "Invalid label budget")
    require(label_budget == 0 or label_provider is not None, "Missing authorized label provider")
    scenario = context["s"]
    prior_group = None
    seen = set()
    remaining = label_budget
    for batch in replay.batches():
        group = batch[0]["group_id"]
        if group != prior_group:
            rows, values, predictions, released = [], [], [], {}
            prior_group = group
        window = str(replay.cursor)
        vectors = []
        for row in batch:
            require(row["group_id"] == group and row["id"] not in seen, "Duplicate ID or mixed group")
            require(not ({"answer", "target", "label"} & row.keys()), "Observation source contains labels")
            pairs = producer.flatten(row["inputs"])
            require([name for name, _ in pairs] == context["order"], "Feature order changed")
            vectors.append([number for _, number in pairs])
            seen.add(row["id"])
        batch_predictions = context["model"].predict(np.asarray(vectors, dtype=np.float64)).tolist()
        require(len(batch_predictions) == len(batch), "Industrial prediction count mismatch")
        by_id = dict(zip((row["id"] for row in batch), batch_predictions))
        store.commit({"industrial/" + window: {"group": group, "predictions": by_id}})
        release_key = "feedback/" + window
        existing = store.get(release_key)
        if existing is not None:
            newly_released = existing
        elif remaining:
            newly_released = label_provider.release(batch, set(by_id), scenario, remaining)
        else:
            newly_released = {}
        require(set(newly_released) <= by_id.keys() and len(newly_released) <= remaining,
                "Feedback not authorized for this predicted batch")
        store.commit({release_key: newly_released})
        released.update({key: {"answer": answer} for key, answer in newly_released.items()})
        remaining -= len(newly_released)
        rows.extend(batch)
        values.extend(vectors)
        predictions.extend(batch_predictions)
        group_id = "g-" + producer.rank("anonymous.v1", scenario, group)[:16]
        evidence = producer.evidence(context, {"group_id": group_id}, rows, values, predictions, released,
                                     len(rows), remaining, "decide")
        messages = message_builder(evidence, "decide")
        require([message["role"] for message in messages] == ["system", "user"], "Unexpected model role")
        require(json.loads(messages[1]["content"]) == evidence, "Noncanonical decision input")
        record = {"group": group_id, "as_of": evidence["as_of"], "windows": evidence["windows"],
                  "messages_sha256": digest(messages), "label_feedback": evidence["label_feedback"],
                  "window_id": window, "source": "closed_loop", "scenario": scenario}
        evidence_hash = digest(evidence)
        store.commit({"input/" + window: record, "evidence-digest/" + window: evidence_hash})
        record = dict(record, evidence_sha256=evidence_hash)
        replay.commit(len(batch))
        yield record, evidence, messages


def main():
    parser = argparse.ArgumentParser(description="Prepare closed-loop decision windows offline; no model0 or training calls")
    for name in ("data", "training", "config", "database"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scenario", required=True, choices=("A", "B", "C"))
    parser.add_argument("--max-batches", required=True, type=int)
    args = parser.parse_args()
    require(args.max_batches > 0, "Positive window count required")
    contract = read(args.config)["contract_sha256"]
    producer, context, replay, builder, manifest = load_runtime(args.data.resolve(), args.training.resolve(),
                                                                args.scenario, contract)
    store = Records(args.database)
    try:
        identity = {"mode": "2a-evidence-offline", "scenario": args.scenario, "label_budget": 0,
                    "manifest": manifest, "contract": contract, "policy_sha256": Policy().sha256}
        require(not store.export() or store.get("identity") == identity, "Unrelated experiment database")
        store.commit({"identity": identity})
        for record, _, _ in islice(window_stream(store, producer, context, replay, builder), args.max_batches):
            print(canonical(record))
    finally:
        store.close()


if __name__ == "__main__":
    main()
