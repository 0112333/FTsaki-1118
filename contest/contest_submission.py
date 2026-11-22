"""FTsaki participant entry points for contest_harness 1.0.2."""
from functools import lru_cache
from importlib import import_module
import json
from math import nextafter
import os
from pathlib import Path
import shutil
import tempfile


TASKS = ("classification", "regression")
BASELINES = {
    "A": ("baseline_a", "gas_class"),
    "B": ("baseline_b", "minutes"),
    "C": ("baseline_c", "cycles"),
}
CHECKPOINT = "checkpoint.npz"
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]


@lru_cache(maxsize=len(BASELINES))
def profile(dataset_key):
    package, _ = BASELINES[dataset_key]
    config_path = Path(__file__).parent / package / "configs" / f"{dataset_key}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return package, config_path, config


def check_model(model, dataset_key, task):
    _, _, config = profile(dataset_key)
    meta = model.meta
    if meta.get("profile") != dataset_key or meta["task"] != task or task != config["task"]:
        raise ValueError(f"checkpoint and task do not match the {dataset_key} profile")
    if meta["classes"] != config["classes"]:
        raise ValueError(f"checkpoint classes do not match the {dataset_key} profile")
    _, unit = BASELINES[dataset_key]
    # B checkpoints declare minutes. Legacy A/C checkpoints imply the unit by profile.
    checkpoint_unit = meta.get("target_unit", None if dataset_key == "B" else unit)
    if checkpoint_unit != unit:
        raise ValueError(f"checkpoint for {dataset_key} must use target unit {unit!r}")


def parent_checkpoint(parent):
    expected = parent.checkpoint_sha256.removeprefix("sha256:")
    files = {entry["path"]: entry["sha256"].removeprefix("sha256:") for entry in parent.receipt["files"]}
    matches = [
        path for path in parent.descriptor["checkpoint_paths"]
        if files.get(path) == expected
    ]
    if len(matches) != 1:
        raise ValueError("round parent digest must identify exactly one declared checkpoint")
    return parent.path(matches[0])


def choose_artifact(rnd, submitted):
    candidate_id = submitted["artifact_id"]
    parent_id = rnd.parent.artifact_id
    if candidate_id == parent_id:
        raise ValueError("candidate receipt must identify a new artifact")

    history = rnd.history()[-3:]
    history_fields = ("id", "round_index", "selected_artifact_id", "hypothesis", "change_summary")
    evidence = {
        "metric": rnd.task["metric"],
        "metric_direction": rnd.task["metric_direction"],
        "parent": {"artifact_id": parent_id, "public_validation": rnd.frame.get("current_validation") or None},
        "candidate": {"artifact_id": candidate_id, "public_validation": submitted.get("summary") or None},
        "recent_history": [
            {key: record[key] for key in history_fields if key in record} for record in history
        ],
    }
    receipt = rnd.model([
        {
            "role": "system",
            "content": (
                "Use the public validation evidence to select exactly one of the two artifact IDs. "
                "When the evidence does not establish a preference, retain the parent. "
                "Return only a JSON object with string fields selected_artifact_id and reason."
            ),
        },
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, allow_nan=False)},
    ], temperature=0)

    try:
        selection = json.loads(receipt.text)
    except json.JSONDecodeError as exc:
        raise ValueError("decision model must return a JSON artifact selection") from exc
    if not isinstance(selection, dict) or selection.get("selected_artifact_id") not in (parent_id, candidate_id):
        raise ValueError("decision model selected an unknown artifact")
    reason = selection.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason.encode("utf-8")) > 4000:
        raise ValueError("decision model must provide a nonblank reason within 4000 bytes")
    return selection["selected_artifact_id"], reason, receipt.call_id, [record["id"] for record in history]


def train(rnd):
    dataset_key = rnd.dataset_key
    package, config_path, config = profile(dataset_key)
    _, unit = BASELINES[dataset_key]
    task = rnd.task["task"]
    if rnd.task.get("dataset_key", dataset_key) != dataset_key or task != config["task"]:
        raise ValueError(f"round task contract does not match the {dataset_key} profile")
    if task == "regression" and rnd.task.get("target_unit") not in (None, unit):
        raise ValueError(f"round target unit for {dataset_key} must be {unit!r}")

    baseline = import_module(f"{package}.train")
    parent = parent_checkpoint(rnd.parent)
    check_model(baseline.Model.load(parent), dataset_key, task)
    # train_run also checks both public projections against the parent feature paths before fitting.
    work = Path(tempfile.mkdtemp(prefix=f"ftsaki-{dataset_key}-{rnd.round_index}-", dir="/workspace"))
    training = work / "training"
    report = baseline.train_run(
        config_path,
        rnd.data.train_path,
        rnd.data.validation_path,
        output=training,
        resume=parent,
        stage_id=rnd.stage_id or "contest",
    )

    checkpoint = training / CHECKPOINT
    upload_limit = rnd.limits.get("candidate_upload_bytes_per_round")
    if upload_limit is not None and checkpoint.stat().st_size > upload_limit:
        raise ValueError("candidate checkpoint exceeds the host upload budget")
    candidate = work / "candidate"
    candidate.mkdir()
    shutil.copyfile(checkpoint, candidate / CHECKPOINT)
    epochs = report["final_epochs"] - report["initial_epochs"]
    experiment = {
        "hypothesis": "Continuing the parent on this round's public training split may improve public validation.",
        "change_summary": f"Continued the parent CPU baseline for {epochs} epochs on the public training split.",
        "configuration": {**report["configuration"], "dataset_key": dataset_key, "task": task, "target_unit": unit},
        "seed": report["seed"],
    }
    submitted = rnd.submit_candidate(
        candidate, checkpoint_paths=[CHECKPOINT], predict_argv=PREDICT_ARGV, **experiment,
    )
    selected, reason, call_id, history_ids = choose_artifact(rnd, submitted)
    rnd.finish(
        selected_artifact_id=selected, reason=reason, history_ids=history_ids,
        model_call_ids=[call_id], **experiment,
    )


@lru_cache(maxsize=1)
def load_predictor(path, dataset_key, task):
    # All three profiles use the same checkpoint format and prediction implementation.
    baseline = import_module("baseline_a.predict")
    model = baseline.Model.load(path)
    if dataset_key is None:
        dataset_key = model.meta.get("profile")
    if dataset_key not in BASELINES:
        raise ValueError("checkpoint must identify a supported dataset")
    check_model(model, dataset_key, task)
    return baseline.predict, model


def predict(sample, model_dir):
    artifact_path = os.environ.get("CONTEST_ARTIFACT_PATH")
    if not artifact_path:
        raise ValueError("CONTEST_ARTIFACT_PATH must name the selected checkpoint")
    path = Path(artifact_path)
    if not path.resolve().is_relative_to(Path(model_dir).resolve()):
        raise ValueError("selected checkpoint is outside the model mount")
    predict_one, model = load_predictor(path, sample.dataset_key, sample.task)
    result = predict_one(model, sample.inputs)
    if sample.task == "classification":
        scores = result["outputs"]
        winner = max(scores, key=lambda item: item["score"])
        # Match the baseline's first-maximum tie break while satisfying SDK validation.
        for item in scores:
            if item is not winner and item["score"] == winner["score"]:
                item["score"] = nextafter(item["score"], 0.0)
    return result
