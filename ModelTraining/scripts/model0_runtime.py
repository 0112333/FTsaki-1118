"""Runtime message adapter. Import from this directory; input must be canonical."""
import json
from model0_input import VERSION, fingerprint, project_input
from model0_messages import TEMPLATES, canonical, system_instruction

def runtime_messages(evidence, task):
    instructions = json.loads(TEMPLATES.read_text())["instructions"]
    return [
        {"role":"system", "content":system_instruction(task, evidence["scenario_id"])},
        {"role":"user", "content":canonical(project_input(evidence, task))},
    ]

def contract():
    return {"input_version":VERSION, "input_code_sha256":fingerprint(),
            "harness_integrated":False}
