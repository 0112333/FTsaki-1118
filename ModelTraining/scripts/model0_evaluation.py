"""Strict response validation and ORIGINAL harness forecast metrics."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from model0_messages import ROOT, canonical, project

HARNESS=ROOT.parent/"FTsaki_harness/src"
def score(episode,text):
    # Reuse canonical target validation, without requiring a fake audit envelope.
    output=json.loads(text)
    candidate=dict(episode,target=output)
    project(candidate,input_projection="canonical")
    if episode["task"]=="decide":
        truth=episode["target"]["decision"];predicted=output["decision"]
        return {"valid":True,"truth":truth,"prediction":predicted,"correct":truth==predicted,
                "citations_valid":True}
    if str(HARNESS) not in sys.path:sys.path.insert(0,str(HARNESS))
    module=importlib.import_module("ftsaki_harness.forecast")
    scale=episode["input"]["feature_scale"]
    result=module.evaluate(episode["scenario_id"],output["values"],episode["target"]["values"],
        episode["input"]["recent_history"]["values"],module.Scale(tuple(scale["mean"]),tuple(scale["std"])))
    return {"valid":True,**result,"harness_forecast_sha256":hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}

def validate_output_policy(episode, text):
    """Validate the response itself; never normalize or repair generated text."""
    from decimal import Decimal
    if episode["scenario_id"] != "A" or episode["task"] != "forecast_next": return
    obj=json.loads(text,parse_float=Decimal)
    in_string=False; escaped=False
    for ch in text:
        if in_string:
            if escaped: escaped=False
            elif ch == chr(92): escaped=True
            elif ch == chr(34): in_string=False
        elif ch == chr(34): in_string=True
        elif ch.isspace(): raise ValueError("A response is not minified JSON")
    for row in obj["values"]:
        for value in row:
            if isinstance(value,Decimal) and value != value.quantize(Decimal("0.000001")):
                raise ValueError("A prediction exceeds six decimal places")

def summarize_decisions(records):
    labels=("drift","no_drift","insufficient")
    matrix={a:{b:0 for b in labels} for a in labels}
    invalid=0
    for r in records:
        if not r.get("valid"):invalid+=1;continue
        matrix[r["truth"]][r["prediction"]]+=1
    f1s=[]
    for label in labels:
        tp=matrix[label][label];fp=sum(matrix[a][label] for a in labels if a!=label)
        fn=sum(matrix[label][b] for b in labels if b!=label)
        # Invalid responses count as failures, but lack a usable prediction class;
        # report them separately and never omit them from total success rates.
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return {"confusion_matrix_valid_outputs":matrix,"invalid_outputs":invalid,
       "macro_f1_valid_outputs":sum(x for x in f1s if x is not None)/sum(x is not None for x in f1s) if any(x is not None for x in f1s) else None,
       "total":len(records)}
