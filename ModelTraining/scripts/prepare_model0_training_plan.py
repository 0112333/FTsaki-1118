"""Deterministic candidate schedules only; does not create a training client."""
import collections, hashlib, json
from model0_messages import ROOT, manifest_hash
OUT=ROOT/"artifacts/preflight/a_forecast_v2"
lengths=json.loads((OUT/"lengths.json").read_text())
train=[r for r in lengths if r["split"]=="train"]
assert len(train)==1210 and len({r["episode_id"] for r in train})==1210
buckets=collections.defaultdict(list)
for r in sorted(train,key=lambda r:r["episode_id"]):
    buckets[(r["scenario"],r["task"])].append(r)
ordered=[]
while any(buckets.values()):
    for key in sorted(buckets):
        if buckets[key]:ordered.append(buckets[key].pop(0))
assert len(ordered)==1210
trial=ordered[:16]
report={"state":"candidate_not_executable","manifest_sha256":manifest_hash(ROOT.parent/"data/model0_episodes_v1_1"),
    "input_contract_sha256":hashlib.sha256((OUT/"frozen-contract.json").read_bytes()).hexdigest(),
    "base_model":"Qwen/Qwen3.8-27B","seed":42,
    "trial":{"rank":16,"learning_rate":5e-5,"batch_size":1,"max_epochs":1,"max_optim_steps":16,
        "sampling":"scene/task round robin; episode_id order within bucket","episodes":trial,
        "max_sequence_token_sum":sum(r["sequence_tokens"] for r in trial)},
    "formal":{"fresh_independent_lora":True,"epochs":1,"records":1210,"candidate_batch_size":1,
        "ordered_episode_ids":[r["episode_id"] for r in ordered],
        "sum_sequence_tokens":sum(r["sequence_tokens"] for r in train),
        "sum_assistant_tokens":sum(r["assistant_tokens"] for r in train)},
    "dev":{"records":244,"role":"evaluation_only"},"quarantine":{"records":1012,"role":"excluded"},
    "remaining":["Real complete-output probe","Frozen-dev baseline","Server LoRA verification",
        "Loss normalization and executable limits","Trial checkpoint reload and paired dev comparison before formal"],
    "training_started":False}
(OUT/"training-plan-candidate.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"trial_records":len(trial),"trial_sequence_token_sum":report["trial"]["max_sequence_token_sum"],
    "formal_records":1210,"formal_sequence_token_sum":report["formal"]["sum_sequence_tokens"],"state":report["state"]}))
