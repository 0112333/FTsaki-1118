"""Shared strict IO, data projection and frozen protocols for model0 dataset v1."""
from pathlib import Path
import collections
import functools
import hashlib
import importlib.util
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parent
if sys.platform == "win32":
    sys.path.insert(0, str(ROOT / ".training-deps"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

VERSION = "model0-generator.v1.1"
BATCH = {"A": 16, "B": 8, "C": 32}
REVIEWER = "model0-deterministic-review.v1"
LABEL_BUDGET = 4096

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(path):
    path = Path(path)
    info = path.stat()
    return _digest_cached(str(path.resolve()), info.st_size, info.st_mtime_ns)

@functools.lru_cache(maxsize=20000)
def _digest_cached(path, size, modified_ns):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def rank(*parts):
    return hashlib.sha256(canonical(list(parts)).encode()).hexdigest()

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result

def parse(text):
    def reject(value):
        raise ValueError("invalid JSON constant " + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)

def read_json(path):
    return parse(Path(path).read_text(encoding="utf-8"))

def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("blank JSONL row")
            yield parse(line)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(canonical(record) + "\n")

def record(path, **extra):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), **extra}

def resolve(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("path outside data workspace")
    return p

def flatten(value, path=""):
    if isinstance(value, dict):
        return [p for key in sorted(value) for p in flatten(value[key], path + "/" + key)]
    if isinstance(value, list):
        return [p for i, child in enumerate(value) for p in flatten(child, path + "/" + str(i))]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("non-finite/non-numeric feature at " + path)
    return [(path, value)]

def vector(row):
    return [v for _, v in flatten(row["inputs"])]

def population(x):
    a = np.asarray(x, dtype=np.float64)
    return {"sample_count": len(a), "feature_mean": a.mean(axis=0).tolist(),
            "feature_std": a.std(axis=0, ddof=0).tolist()}

def source_step(s, row):
    return row["meta"][{"A": "source_index", "B": "snapshot", "C": "cycle"}[s]]

def position(s, row, group_step):
    return {"id": row["id"], "replay_order": row["replay_order"],
            "group_step": group_step, "source_step": source_step(s, row)}

def check_descriptor(base, desc):
    p = base / desc["path"]
    if digest(p) != desc["sha256"].removeprefix("sha256:"):
        raise ValueError("upstream digest mismatch: " + str(p))
    return p

def reference_rows(s, data):
    # Fixed reference bank, independent of full industrial_train feature scale.
    if s == "A":
        return data
    strata = collections.defaultdict(list)
    for row in data:
        key = row["meta"]["bearing"] if s == "B" else row["meta"]["source_subset"]
        strata[key].append(row)
    selected = {r["id"] for key, group in strata.items()
                for r in sorted(group, key=lambda r: rank("reference.v1", 42, s, key, r["id"]))[:128]}
    return [r for r in data if r["id"] in selected]

def contexts():
    base = ROOT / "harness_data_v1"
    manifest = read_json(base / "manifest.json")
    result = {}
    for s, entry in manifest["scenarios"].items():
        trainpath = check_descriptor(base, entry["splits"]["industrial_train"]["dataset"])
        obspath = check_descriptor(base, entry["splits"]["model0_source"]["observations"])
        labelpath = check_descriptor(base, entry["splits"]["model0_source"]["labels"])
        checkpoint = check_descriptor(base, entry["initial_checkpoint"])
        train, observations = list(rows(trainpath)), list(rows(obspath))
        ref = reference_rows(s, train)
        order = [k for k, _ in flatten(ref[0]["inputs"])]
        for row in ref + observations:
            if [k for k, _ in flatten(row["inputs"])] != order:
                raise ValueError("feature schema mismatch")
        modelpath = ROOT / entry["source_package"] / "baselines/model.py"
        spec = importlib.util.spec_from_file_location("industrial_model_" + s, modelpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model.load(checkpoint)
        converted_paths = ["/" + "/".join(str(part[1]) for part in json.loads(p)) for p in model.meta["paths"]]
        if converted_paths != order:
            raise ValueError("checkpoint feature order mismatch")
        groups = collections.defaultdict(list)
        for row in observations:
            groups[row["group_id"]].append(row)
        result[s] = {"s": s, "entry": entry, "ref_rows": ref, "order": order,
                     "reference": population([vector(r) for r in ref]),
                     "scale": population([vector(r) for r in train]),
                     "observations": observations, "groups": dict(groups),
                     "trainpath": trainpath, "obspath": obspath, "labelpath": labelpath,
                     "checkpoint": checkpoint, "modelpath": modelpath, "model": model}
    return result

def assignments(ctx):
    result = []
    for s, c in ctx.items():
        dev = set()
        if s == "C":
            strata = collections.defaultdict(list)
            for key in c["groups"]:
                strata[key.split("/")[0]].append(key)
            for fd, keys in sorted(strata.items()):
                n = max(1, int(len(keys) * .2 + .5))
                dev.update(sorted(keys, key=lambda key: rank("split.v1", 42, s, fd, key))[:n])
        if s == "B":
            dev = {"Bearing2_2"}  # Fixed by identifier, not length or performance.
        for group, data in sorted(c["groups"].items()):
            blocks = [(group, data, "dev" if group in dev else "train")]
            if s == "A":
                blocks = [(group + "/block001", data[:64], "train"),
                          (group + "/block002", data[64:], "dev")]
            for key, group_rows, split in blocks:
                result.append({"scenario_id": s, "group_key": s + ":" + key,
                               "source_group": group, "group_id": "g-" + rank("anonymous.v1", s, key)[:16],
                               "source_family_id": "family-" + rank("family.v1", s, key)[:20],
                               "split": split, "sample_ids": [r["id"] for r in group_rows]})
    return result

def protocols():
    return {
        "split": {"id": "model0-split.v1", "seed": 42,
                  "A": "First 64 source-order observations train; last 26 dev. Reset all windows at block boundary; no cross-block history/targets. One collection domain, not independent acquisition domains.",
                  "B": "Bearing2_2 dev; Bearing1_2 and Bearing3_2 train. Identifier-only choice.",
                  "C": "Per FD, round(n*0.2) whole engines dev; ascending SHA256 of canonical ['split.v1',42,'C',FD,group].",
                  "family_rule": "Every task, overlapping window, original and synthetic variant of one group/block stays in the same split.",
                  "industrial_reference_sharing": True},
        "evidence": {"id": "model0-evidence-calculation.v1.1", "batch_sizes": BATCH,
                     "short_batches": 2, "long_batches": 8, "horizon": 32,
                     "reference_selection": "A: all 267 industrial_train rows. B: up to128 per industrial_train bearing. C:128 per FD. Rank by SHA256 canonical ['reference.v1',42,scenario,stratum,id], preserve source order after selection.",
                     "feature_scale": "Population mean/std from ALL industrial_train rows, ddof=0. Original zeros retained. Separate from sampled training_reference.",
                     "standardized_mean_delta_denominator": "Full feature_scale.std; zero std yields null. Mean reference remains training_reference or recent_stable_reference as named.",
                     "reference_sampling_limitation": "Evidence statistics use a finite reference bank, not all industrial_train rows. No performance-driven selection.",
                     "flatten": "Sorted dictionary keys, numeric array index order, slash paths",
                     "current_and_prediction_scope": "Most recent at most2 batches of the same group/block",
                     "feedback_scope": "Released labels among current short-window predictions; fixed six-class macro F1 with zero denominator=0",
                     "long_scope": "Most recent at most8 batches, including real tail",
                     "quality": "All current sources finite. Missing/nonfinite extensions are unsupported and rejected; no silent deletion.",
                     "unknown_state": None},
        "labels": {"id": "model0-label-release.v1", "seed": 42, "fraction": .25,
                   "algorithm": "Sort batch IDs by SHA256(UTF8 canonical([42,scenario,original_source_group,id])); take min(ceil(n*.25),remaining). Same canonical serialization as harness.",
                   "budget_per_group_or_A_block": LABEL_BUDGET,
                   "order": "Append and flush industrial batch predictions before requesting labels from the separate label worker; then append and flush release log; only then build input.",
                   "serialized_label_order": "Within each batch log selected IDs lexicographically; selection itself uses SHA256 ranks.",
                   "default_variant": "natural",
                   "synthetic_variant": "Zero-feedback controlled experiment; fixed label budget0, no label requests, predictions recomputed on transformed inputs.",
                   "train_authorized": False},
        "forecast": {"id": "model0-observed-next32.v1", "source_type": "observed_future_features",
                     "cutoffs": "Every completed batch, including real tail, if 32 same-block future rows exist; A additionally needs32 arrived rows.",
                     "target": "Exactly next32 actual unmodified model0_source feature vectors in original units/order.",
                     "history": "Most recent min(32,arrived) rows; B/C short history allowed and reported",
                     "augmentation": "None; no synthetic forecast targets"},
        "coverage": {"id": "model0-coverage-annotation.v1", "source_type": "independent_annotation",
                     "annotator": "model0-coverage-rule.v1", "reviewer": REVIEWER,
                     "rule": "If fewer than2*batch_size observations have arrived in this group/block, annotate insufficient for this explicitly minimum-coverage decision protocol. No claim of absent input change or task-performance stability.",
                     "independence": "Deterministic annotation and separately executed validator; no LLM labels, no human causal review.",
                     "otherwise": "Natural drift/no_drift unknown: target=null, quarantine. Missing review never maps to insufficient."},
        "controlled": {"id": "model0-paired-empirical-control.v1.1", "source_type": "controlled_intervention",
                       "cutoffs": "Every fourth full batch; two disjoint full short windows are required.",
                       "anchor": "The2 batches immediately preceding the current2 batches, already observable before the current window.",
                       "stable_control": "Replay anchor feature rows exactly, in the same order, into the current observation slots. Equal empirical feature distributions by construction.",
                       "shift": "Same replay as control, then add +/-3 full-industrial_train standard deviations to one deterministic nonconstant feature. Feature/sign chosen by SHA256 of event, independent of split/labels/effects.",
                       "scope": "Simulated current input distribution relative to the known pre-current anchor, not natural device health, global training-domain equality, or real closed-loop gains.",
                       "label_variant": "zero_feedback",
                       "audit": "Record current slot IDs and origin anchor IDs separately. Sources remain original model0_source; transformed vectors live in snapshots. Both versions remain in same family/split.",
                       "insufficient": "Not generated by this experiment"},
    }

def prediction_summary(s, predictions, classes):
    a = np.asarray(predictions, dtype=np.float64)
    if s == "A":
        return {"task": "classification", "sample_count": len(a), "class_order": classes,
                "mean_probabilities": a.mean(axis=0).tolist(),
                "predicted_counts": np.bincount(a.argmax(axis=1), minlength=len(classes)).tolist()}
    return {"task": "regression", "sample_count": len(a), "mean": float(a.mean()),
            "std": float(a.std()), "unit": "minutes" if s == "B" else "cycles"}

def feedback(s, current_rows, predictions, released, classes):
    eligible = len(current_rows)
    pairs = [(r, p, released[r["id"]]["answer"]) for r, p in zip(current_rows, predictions) if r["id"] in released]
    if not pairs:
        metrics = None
    elif s == "A":
        truth = [answer for _, _, answer in pairs]
        pred = [classes[int(np.argmax(p))] for _, p, _ in pairs]
        f1 = []
        for cls in classes:
            tp = sum(t == cls and p == cls for t,p in zip(truth,pred))
            fp = sum(t != cls and p == cls for t,p in zip(truth,pred))
            fn = sum(t == cls and p != cls for t,p in zip(truth,pred))
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        metrics = {"accuracy": sum(t==p for t,p in zip(truth,pred))/len(pairs),
                   "macro_f1": sum(f1)/len(classes),
                   "class_order": classes, "per_class_count": [truth.count(c) for c in classes]}
    else:
        errors = [float(p)-float(answer) for _,p,answer in pairs]
        metrics = {"rmse": math.sqrt(sum(e*e for e in errors)/len(errors)),
                   "mae": sum(abs(e) for e in errors)/len(errors),
                   "unit": "minutes" if s=="B" else "cycles"}
    return {"visible_count": len(pairs), "eligible_prediction_count": eligible,
            "coverage": len(pairs)/eligible if eligible else 0, "metrics": metrics}

def evidence(c, assignment, group_rows, values, predictions, released, end, remaining,
             task, stable=None):
    s, b = c["s"], BATCH[c["s"]]
    # end is a batch end; include actual tail, not an arbitrary last 2*b slice.
    starts = list(range(0, end, b))
    short_start = starts[max(0,len(starts)-2)]
    long_start = starts[max(0,len(starts)-8)]
    short = values[short_start:end]
    stat = population(short)
    ref = c["reference"]
    scale = c["scale"]["feature_std"]
    delta = [(v-m)/sd if sd else None for v,m,sd in zip(stat["feature_mean"],ref["feature_mean"],scale)]
    items = [
        {"id": "feature-location", "kind": "feature_change", "summary": "当前短窗口相对固定工业历史参考的逐特征均值变化。",
         "standardized_mean_delta": delta},
        {"id": "quality", "kind": "data_quality", "summary": "当前窗口输入均为有限数值。",
         "finite_counts": [len(short)]*len(c["order"])},
        {"id": "window-coverage", "kind": "coverage", "summary": "当前可见窗口覆盖数量。",
         "required_short_samples": 2*b, "available_short_samples": len(short)},
        {"id": "feedback-coverage", "kind": "label_coverage", "summary": "性能指标仅覆盖当前窗口中已释放且先有预测的标签。"},
    ]
    if stable is not None:
        items.append({"id": "recent-reference-change", "kind": "feature_change",
                      "summary": "当前短窗口相对截止点之前已建立参考的逐特征均值变化。",
                      "standardized_mean_delta": [(v-m)/sd if sd else None for v,m,sd in
                                                  zip(stat["feature_mean"],stable["feature_mean"],scale)]})
    if len(starts)>=8:
        items.append({"id":"long-window-summary","kind":"long_window_summary",
                      "summary":"最近8批的总体统计，独立于短窗口。",
                      "group_step_start":long_start+1,"group_step_end":end,
                      "finite_counts":[end-long_start]*len(c["order"]), **population(values[long_start:end])})
    last = group_rows[end-1]
    version = "industrial-v1-"+s+"-"+digest(c["checkpoint"])[:12]
    result = {
        "schema_version": "model0.evidence.v1", "scenario_id":s, "group_id":assignment["group_id"],
        "as_of":{"clock":"logical_replay","replay_order":last["replay_order"],"group_step":end,
                 "source_step_name":{"A":"source_index","B":"snapshot","C":"cycle"}[s],
                 "source_step":source_step(s,last)},
        "feature_order":c["order"],
        "windows":{"batch_size":b,
                   "short":{"requested_batches":2,"available_batches":min(2,len(starts)),"sample_count":len(short)},
                   "long":{"requested_batches":8,"available_batches":min(8,len(starts)),"sample_count":end-long_start}},
        "industrial_model":{"baseline_version":version,"current_version":version},
        "feature_scale":{"fit_role":"industrial_train","count":c["scale"]["sample_count"],
                         "mean":c["scale"]["feature_mean"],"std":c["scale"]["feature_std"]},
        "training_reference":ref,"recent_stable_reference":stable,
        "current":{**stat,"missing_fraction":0,"nonfinite_count":0},
        "prediction_summary":prediction_summary(s,predictions[short_start:end],c["model"].meta["classes"]),
        "label_feedback":feedback(s,group_rows[short_start:end],predictions[short_start:end],released,c["model"].meta["classes"]),
        "job_state":{"industrial_training_running":False,"decision_training_running":False},
        "budget_state":{"labels_remaining":remaining,"training_credits_remaining":None,"inference_tokens_remaining":None},
        "evidence_refs":[i["id"] for i in items],"evidence_items":items}
    if task=="forecast_next":
        begin=max(0,end-32)
        result["recent_history"]={"group_steps":list(range(begin+1,end+1)),"values":values[begin:end]}
    return result

def counts(records):
    totals = collections.Counter()
    families = collections.defaultdict(set)
    devices = collections.defaultdict(set)
    for r in records:
        decision = r["target"].get("decision", "future_features") if r["target"] else "unreviewed"
        source = r["audit"]["supervision"]["source_type"] or "unreviewed"
        key = (r["split"],r["scenario_id"],r["task"],decision,source)
        totals[key]+=1
        families[key].add(r["audit"]["split_assignment"]["source_family_id"])
        devices[key].add(r["audit"]["split_assignment"]["group_key"])
    return [{"split":k[0],"scenario_id":k[1],"task":k[2],"target":k[3],"supervision_source":k[4],
             "episodes":n,"source_families":len(families[k]),"groups":len(devices[k])}
            for k,n in sorted(totals.items())]
