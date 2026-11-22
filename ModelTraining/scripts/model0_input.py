"""Input-only projection shared by SFT and runtime. Source episodes stay canonical."""
import copy
import hashlib
import math
from pathlib import Path

VERSION = "model0.evidence.a_forecast.v1"
# Floating point reconstruction tolerance, in units of max(1, |raw|, |mean|).
FLOAT_TOLERANCE = 2e-12

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def project_input(evidence, task):
    if task not in {"decide", "forecast_next"}:
        raise ValueError("Unknown task")
    if evidence.get("schema_version") != "model0.evidence.v1":
        raise ValueError("Expected canonical evidence v1")
    result = copy.deepcopy(evidence)
    if task != "forecast_next" or evidence["scenario_id"] != "A":
        return result
    scale = evidence["feature_scale"]
    mu, std = scale["mean"], scale["std"]
    d = len(evidence["feature_order"])
    if d != 128 or len(mu) != d or len(std) != d or not all(finite(v) for v in mu+std) or any(v < 0 for v in std):
        raise ValueError("Invalid full feature scale")
    rows = evidence["recent_history"]["values"]
    steps = evidence["recent_history"]["group_steps"]
    n = evidence["current"]["sample_count"]
    end = evidence["as_of"]["group_step"]
    if not isinstance(n, int) or isinstance(n, bool) or not 0 < n <= len(rows) or len(rows) != len(steps):
        raise ValueError("Current window is not covered by history")
    if steps != list(range(end-len(rows)+1, end+1)) or evidence["windows"]["short"]["sample_count"] != n:
        raise ValueError("History order/current window mismatch")
    if any(len(row) != d or not all(finite(x) for x in row) for row in rows):
        raise ValueError("History contains missing/nonfinite or wrong dimensions")
    current = evidence["current"]
    if current["missing_fraction"] != 0 or current["nonfinite_count"] != 0:
        raise ValueError("Derived current requires fully finite observations")
    # Recompute the ORIGINAL current before dropping its vectors: this proves that
    # the covered history describes the supplied current, rather than just fitting n.
    for j in range(d):
        column = [r[j] for r in rows[-n:]]
        mean = math.fsum(column)/n
        sd = math.sqrt(math.fsum((x-mean)**2 for x in column)/n)
        if not math.isclose(mean, current["feature_mean"][j], rel_tol=1e-10, abs_tol=1e-10) or not math.isclose(sd, current["feature_std"][j], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Current statistics disagree with covered history")
    encoded = []
    for row in rows:
        qrow = []
        for x, m, sd in zip(row, mu, std):
            unit = sd if sd > 0 else 1
            q = round(100*(x-m)/unit)  # Python round: ties-to-even.
            reconstructed = m + q*unit/100
            tolerance = FLOAT_TOLERANCE*max(1, abs(x), abs(m))
            if abs(reconstructed-x) > .005*unit+tolerance:
                raise ValueError("History quantization error exceeds bound")
            qrow.append(q)
        encoded.append(qrow)
    result["schema_version"] = VERSION
    result["recent_history"] = {
        "group_steps": copy.deepcopy(steps), "encoding": "integer_centi_z_score",
        "decode": "raw = mean[j] + values[h][j] * (std[j] if std[j] > 0 else 1) / 100; feature_scale exact; predict raw units",
        "values": encoded}
    ref = result["training_reference"]
    if ref["sample_count"] == scale["count"] and ref["feature_mean"] == mu and ref["feature_std"] == std:
        result["training_reference"] = {k:v for k,v in ref.items() if k not in {"feature_mean","feature_std"}}
        result["training_reference"]["same_statistics_as"] = "feature_scale"
    result["current"] = {k:v for k,v in current.items() if k not in {"feature_mean","feature_std"}}
    result["current"]["statistics_from"] = "Decode last sample_count recent_history rows, then compute population mean/std (ddof=0); describes quantized input."
    for item in result["evidence_items"]:
        if item["id"] == "feature-location":
            item["statistics_scope"] = "Original exact observations before history encoding"
    result["input_precision"] = {
        "history_encoding":"integer_centi_z_score", "history_max_abs_error_in_scale_units":.005,
        "zero_std_encoding_unit":1, "rounding":"ties_to_even",
        "current_statistics":"derived_from_quantized_history",
        "other_statistics":"exact_original_observations", "target":"exact_original_units"}
    return result

def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
