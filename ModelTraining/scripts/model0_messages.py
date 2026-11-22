"""Frozen model0 episode projection. No cloud calls; audit never enters messages."""
import ast
import hashlib
import json
import math
from pathlib import Path

VERSION = "model0.messages.v3"
DIMENSIONS = {"A": 128, "B": 34, "C": 24}
ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "configs/model0_messages.v1.json"

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def manifest_hash(dataset):
    return hashlib.sha256((Path(dataset) / "manifest.json").read_bytes()).hexdigest()

def require_accepted(dataset, acceptance):
    report = json.loads(Path(acceptance).read_text())
    if report.get("accepted_for_training") is not True:
        raise ValueError("Dataset not accepted for training")
    if report.get("dataset_manifest_sha256") != manifest_hash(dataset):
        raise ValueError("Acceptance manifest digest mismatch")
    return report

def system_instruction(task, scenario, *, output_policy=True):
    instruction = json.loads(TEMPLATES.read_text())["instructions"][task]
    if output_policy and task == "forecast_next" and scenario == "A":
        instruction += (" Return exactly one minified JSON object: no Markdown fences, prose, indentation or line breaks."
            " Use exactly 32 rows and 128 numbers per row. In values, round predictions to at most 6 decimal places"
            " in original raw units; omit unnecessary trailing zeros. Do not print decoded history, calculations"
            " or extra digits. Complete the whole JSON object and stop.")
    return instruction

def project(episode, *, input_projection="a_forecast_v1"):
    if episode.get("schema_version") != "model0.episode.v1" or episode.get("record_kind") != "episode":
        raise ValueError("Not a supported real episode")
    if episode.get("split") not in {"train", "dev"}:
        raise ValueError("Quarantined/example records cannot be projected")
    if episode.get("audit", {}).get("eligibility", {}).get("sft_eligible") is not True:
        raise ValueError("Ineligible episode")
    mode = episode["task"]
    evidence, target = episode["input"], episode["target"]
    if mode not in {"decide", "forecast_next"} or not isinstance(target, dict):
        raise ValueError("Unsupported task or missing target")
    scenario = episode["scenario_id"]
    order = evidence["feature_order"]
    if scenario not in DIMENSIONS or len(order) != DIMENSIONS[scenario] or len(set(order)) != len(order):
        raise ValueError("Feature dimension/order invalid")
    if evidence["scenario_id"] != scenario:
        raise ValueError("Scenario mismatch")
    if mode == "decide":
        if set(target) != {"decision", "evidence_refs", "hypotheses"}:
            raise ValueError("Decision fields invalid")
        refs = target["evidence_refs"]
        if target["decision"] not in {"drift", "no_drift", "insufficient"}:
            raise ValueError("Invalid decision")
        if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
            raise ValueError("Invalid references")
        if not set(refs) <= set(evidence["evidence_refs"]):
            raise ValueError("Unknown evidence reference")
        if target["decision"] != "insufficient" and not refs:
            raise ValueError("Binary decision requires evidence")
        if not isinstance(target["hypotheses"], list):
            raise ValueError("Invalid hypotheses")
        for h in target["hypotheses"]:
            if set(h) != {"claim", "status", "evidence_refs"} or not isinstance(h["claim"], str) or h["status"] != "hypothesis":
                raise ValueError("Invalid hypothesis")
            if not isinstance(h["evidence_refs"], list) or not all(isinstance(x, str) for x in h["evidence_refs"]) or not set(h["evidence_refs"]) <= set(evidence["evidence_refs"]):
                raise ValueError("Unknown hypothesis reference")
    else:
        if set(target) != {"horizon", "feature_order", "values"} or target["horizon"] != 32 or target["feature_order"] != order:
            raise ValueError("Forecast schema/order invalid")
        matrix = target["values"]
        if not isinstance(matrix, list) or len(matrix) != 32:
            raise ValueError("Forecast must have 32 rows")
        for row in matrix:
            if not isinstance(row, list) or len(row) != DIMENSIONS[scenario]:
                raise ValueError("Forecast dimension invalid")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in row):
                raise ValueError("Forecast contains invalid numbers")
    if input_projection == "a_forecast_v1":
        from model0_input import project_input
        evidence = project_input(evidence, mode)
    elif input_projection != "canonical":
        raise ValueError("Unknown input projection")
    return [
        {"role": "system", "content": system_instruction(mode, scenario, output_policy=input_projection != "canonical")},
        {"role": "user", "content": canonical(evidence)},
        {"role": "assistant", "content": canonical(target)},
    ]

def token_sample(episode, tokenizer):
    """Explicit next-token shift; use SDK forward_backward(auto_shift=False).
    weights[j] applies to target_tokens[j], never to the input at that position.
    Include the assistant end-of-turn token in the supervised suffix.
    """
    messages = project(episode)
    def ids(result):
        # Transformers 5 can return BatchEncoding rather than a bare ID list.
        if hasattr(result, "keys"):
            result = result["input_ids"]
        if not isinstance(result, list) or not all(isinstance(x, int) for x in result):
            raise ValueError("Expected one unbatched token ID sequence")
        return result
    prefix = ids(tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False))
    full = ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False))
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Chat template assistant boundary is not a token prefix")
    input_tokens, target_tokens = full[:-1], full[1:]
    weights = [float(position >= len(prefix)) for position in range(1, len(full))]
    if not any(weights):
        raise ValueError("No supervised assistant tokens")
    return {"input_tokens": input_tokens, "target_tokens": target_tokens, "weights": weights,
            "prompt_tokens": len(prefix), "sequence_tokens": len(full), "assistant_tokens": len(full)-len(prefix)}

def freeze_harness_templates(source):
    tree = ast.parse(Path(source).read_text())
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == "instructions" for t in n.targets)]
    if len(found) != 1:
        raise ValueError("Cannot identify unique harness instruction mapping")
    instructions = ast.literal_eval(found[0].value)
    if set(instructions) != {"decide", "forecast_next"}:
        raise ValueError("Unknown harness tasks")
    return {"version": VERSION, "harness_source": str(source),
            "harness_source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "instructions": instructions}
