"""OpenAI-compatible transport, with explicit request limits and strict local parsing."""
import json
import os
from urllib.request import Request, urlopen
from .policy import canonical

class InferenceError(RuntimeError):
    pass

def validate_hypotheses(result, evidence):
    hypotheses = result.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("Missing hypotheses list")
    for hypothesis in hypotheses:
        if not (isinstance(hypothesis, dict) and hypothesis.get("status") == "hypothesis"
                and isinstance(hypothesis.get("claim"), str)
                and isinstance(hypothesis.get("evidence_refs"), list)
                and all(isinstance(ref, str) and ref in evidence["evidence_refs"] for ref in hypothesis["evidence_refs"])):
            raise ValueError("Invalid hypothesis")

def validate_content(mode, evidence, content, finish_reason):
    if finish_reason != "stop":
        raise InferenceError("Incomplete response: " + str(finish_reason))
    result = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    if mode == "decide":
        if result.get("decision") not in {"drift", "no_drift", "insufficient"}:
            raise ValueError("Invalid decision")
        refs = result.get("evidence_refs")
        if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
            raise ValueError("Missing evidence refs")
        if not set(refs) <= set(evidence.get("evidence_refs", [])):
            raise ValueError("Unknown evidence citation")
        if result["decision"] != "insufficient" and not refs:
            raise ValueError("Binary decision needs evidence")
    elif mode == "forecast_next":
        from .forecast import matrix
        if result.get("horizon") != 32 or result.get("feature_order") != evidence["feature_order"]:
            raise ValueError("Forecast schema mismatch")
        matrix(result["values"], len(evidence["feature_order"]), 32)
    else:
        raise ValueError("Unknown response mode")
    return result

class OpenAICompatible:
    def __init__(self, base_url, model, api_key_env, timeout, max_tokens):
        if timeout <= 0 or max_tokens <= 0:
            raise ValueError("Explicit positive request limits required")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("HTTP(S) API URL required")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self.key = os.environ[api_key_env] if api_key_env else None

    def request(self, mode, evidence):
        instructions = {
            "decide": 'Return JSON: {"decision":"drift|no_drift|insufficient","evidence_refs":["..."],"hypotheses":[]}. Cite only supplied evidence_refs.',
            "forecast_next": 'Return JSON: {"horizon":32,"feature_order":[...],"values":[32 rows of finite feature values]}. Use precisely the supplied feature order. Predict unseen future inputs from supplied past only.',
        }
        if mode not in instructions:
            raise ValueError("Unknown response mode")
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": instructions[mode]},
                                {"role": "user", "content": canonical(evidence)}]}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = Request(self.url, data=canonical(payload).encode(), headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                outer = json.load(response)
            choice = outer["choices"][0]
            result = validate_content(mode, evidence, choice["message"]["content"], choice.get("finish_reason"))
            return {"output": result, "usage": outer.get("usage"), "provider_model": outer.get("model")}
        except InferenceError:
            raise
        except Exception as exc:
            # Do not include request headers or credentials in the error.
            raise InferenceError(type(exc).__name__ + ": inference failed") from None
