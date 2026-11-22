"""Shared opt-in compact evidence projection for BOTH training and runtime.
Canonical episodes and targets remain exact. Adoption requires both consumers
to import project_input and freeze this module's hash; never silently activate.
"""
import copy
import math

VERSION = "model0.evidence.compact.v1"

def rounded(value, digits=3):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite compact input")
        return float(format(value, "."+str(digits)+"g"))
    if isinstance(value, list):
        return [rounded(v,digits) for v in value]
    if isinstance(value, dict):
        return {k:rounded(v,digits) for k,v in value.items()}
    raise ValueError("Unsupported input type")

def project_input(evidence):
    """Statistics in raw units3 significant digits; history in full-scale z units2 decimals.
    Exact full-training feature_scale vectors retained. Targets never accepted.
    This is a different explicitly versioned model input, not evidence v1.
    """
    if evidence.get("schema_version") != "model0.evidence.v1":
        raise ValueError("Expected canonical evidence v1")
    compact=rounded(evidence)
    compact["schema_version"]=VERSION
    compact["feature_scale"]=copy.deepcopy(evidence["feature_scale"])
    if "recent_history" in evidence:
        scale=evidence["feature_scale"]
        compact["recent_history"]={
            "group_steps":copy.deepcopy(evidence["recent_history"]["group_steps"]),
            "encoding":"integer_centi_z_score",
            "decode":"raw = mean[j] + (values[h][j] / 100) * (std[j] if std[j] > 0 else 1); predict target in raw units",
            "values":[[round(100*(x-m)/(sd if sd>0 else 1)) for x,m,sd in zip(row,scale["mean"],scale["std"])] for row in evidence["recent_history"]["values"]]}
    if evidence["training_reference"]["sample_count"]==evidence["feature_scale"]["count"] and evidence["training_reference"]["feature_mean"]==evidence["feature_scale"]["mean"] and evidence["training_reference"]["feature_std"]==evidence["feature_scale"]["std"]:
        compact["training_reference"]={"same_statistics_as":"feature_scale","sample_count":evidence["feature_scale"]["count"]}
    if evidence["scenario_id"]=="A" and "recent_history" in evidence and evidence["current"]["sample_count"]<=len(evidence["recent_history"]["values"]):
        compact["current"]={"sample_count":evidence["current"]["sample_count"],
            "missing_fraction":evidence["current"]["missing_fraction"],"nonfinite_count":evidence["current"]["nonfinite_count"],
            "statistics_from":"Decode last sample_count recent_history rows, then compute population mean/std; these describe quantized input."}
        for item in compact["evidence_items"]:
            if item["id"]=="feature-location":
                item.pop("standardized_mean_delta",None)
                item["statistics_from"]="(decoded_current_mean - training_reference_mean) / feature_scale.std; zero std gives null"
    compact["input_precision"]={
        "statistics_units":"original",
        "statistics_float_significant_digits":3,
        "history_encoding":"full_industrial_scale_integer_centi_z_score",
        "history_max_abs_error_in_scale_units":0.005,
        "feature_scale":"exact_full_industrial_train",
        "integer_values":"exact",
        "target_values":"unchanged_outside_this_projection"}
    return compact
