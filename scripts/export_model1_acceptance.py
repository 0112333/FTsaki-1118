"""Audit and export the completed frozen 2026-11-19 experiment without remote calls."""
import hashlib
import json
from pathlib import Path

from pytrio.config.config import auth_state
from ftsaki_harness.model0_smoke import read, require
from ftsaki_harness.policy import canonical, digest
from ftsaki_harness.records import Records

root=Path.cwd()
runs=root/"runs"
directory=runs/"stage23-C-model1-2026-11-19"
proposal=read(runs/"remaining-tests-proposal-2026-11-19.json")
grant=read(runs/"remaining-tests-grant-2026-11-19.json")
require(digest(proposal)==grant["proposal_sha256"],"Approved proposal changed")
def export(path):
    records=Records(path)
    try:
        return records.export()
    finally:
        records.close()
training=export(directory/"training.sqlite")
stream=export(directory/"stream.sqlite")
inference=export(directory/"inference.sqlite")
evidence=export(directory/"evidence.sqlite")
review=export(runs/"2a-C-first-device-2026-11-19/review.sqlite")
event=proposal["event"]
run=review["identity"]["run"]
suffix=f"{run}/{event}"
intent=training["intent/"+suffix]
job=training["submission-result/"+suffix]
reload=training["reload-result/"+suffix]
report=stream["report"]
require(job["status"]=="artifact-ready" and job["updates"]==2 and job["sequence_tokens_submitted"]==28796,"Incomplete approved training")
require(reload["status"]=="model1-inference-confirmed" and reload["artifact"]==job["artifact"],"Unverified child reload")
require(sum(key.startswith("submission/") for key in training)==1 and sum(key.startswith("step-done/") for key in training)==2,"Unexpected training submission count")
require(digest(review["plan/"+event])==proposal["original_review_sha256"],"Original model attribution changed")
require(review["plan/"+event]["proposal"]["training_objects"]==["industrial"]
        and review["plan/"+event]["proposal"]["budget_suggestion"]=="hold_training","Original attribution not preserved")
require(intent["controller_authorization_sha256"]==digest(review["decision-authorization/"+event]),"Controller authorization mismatch")
require(report["cursor"]==240 and report["stop_reason"]=="authorized-range-complete","Incomplete continuous replay")
windows=[stream[f"result/{cutoff}"] for cutoff in proposal["replay"]["window_cutoffs"]]
require([row["start"] for row in windows]==[97,129,161,193,225]
        and [row["end"] for row in windows]==[128,160,192,224,240],"Noncontiguous replay")
require(len([key for key in stream if key.startswith("seen/")])==144,"Missing or duplicate source samples")
require(report["training_submissions"]=={"decision":1,"industrial":0} and report["label_releases"]==0,"Unauthorized side effects")
require(report["budget"]["reserved_calls"]<=11 and report["budget"]["reserved_tokens"]<=60014
        and report["budget"]["reported_total_tokens"]<=60014,"Inference budget exceeded")
require(all(not value for key,value in evidence.items() if key.startswith("feedback/")),"Unexpected labels")
require(report["pool"]["models"]["0"]["stable"] and report["pool"]["models"]["0"]["online"],"Stable model0 unavailable")
if "forecast/1" in stream:
    forecast=stream["forecast/1"]
    require(forecast["as_of"]==96 and len(forecast["prediction"])==32,"Wrong frozen forecast boundary")
    positions=[value["order"] for key,value in stream.items() if key.startswith("target/1/")]
    require(positions==list(range(97,129)),"Training interval reused or validation not contiguous")

finish=report["candidate"]
outcome=finish["validation"]["outcome"]
admitted=finish["admission"]=="admitted"
changes=[change["action"] for row in windows for change in row["vote"]["changes"]]
coverage=[
 {"branch":"real decision SFT and child reload","real":True,"evidence":"training.sqlite intent/submission-result/reload-result","controlled":"test_training_intent.py"},
 {"branch":"single frozen 32-position candidate prediction","real":"forecast/1" in stream,"evidence":"stream.sqlite forecast-response/1 and forecast/1","controlled":"test_prospective.py::test_bad_responses_end_the_candidate_and_leave_later_windows_live"},
 {"branch":"same-group future-feature validation","real":"forecast-result/1" in stream,"evidence":"stream.sqlite target/1/* and finish/1","controlled":"test_rules.py::test_perfect_bc_passes"},
 {"branch":"candidate gate rejection","real":outcome=="rejected","evidence":"stream.sqlite finish/1","controlled":"test_rules.py::test_wrong_trend_rejected"},
 {"branch":"invalid candidate response isolation","real":outcome in ("invalid-output","transport-error","timeout"),"evidence":"stream.sqlite finish/1","controlled":"test_prospective.py::test_bad_responses_end_the_candidate_and_leave_later_windows_live"},
 {"branch":"trial admission and scoring","real":admitted,"evidence":"stream.sqlite result/* vote.before/after","controlled":"test_prospective.py::test_wait_32_freeze_once_then_promote_and_resume"},
 {"branch":"stable promotion","real":"promoted" in changes,"evidence":"stream.sqlite result/* vote.changes","controlled":"test_rules.py::test_ninth_agreement_promotes"},
 {"branch":"trial disagreement eviction","real":"evicted" in changes,"evidence":"stream.sqlite result/* vote.changes","controlled":"test_rules.py::test_both_disagreement_directions_evict"},
 {"branch":"failed candidate retraining","real":False,"reason":"Approved retry training jobs = 0","controlled":"test_prospective.py::test_retry_new_material_and_fresh_validation_until_limit"},
 {"branch":"insufficient future or group switch","real":False,"reason":"Authorized same-group range contains all 32 targets","controlled":"test_prospective.py::test_stream_end_does_not_consume_retry and test_disagreement_evicts_and_new_group_never_fills_old_horizon"},
 {"branch":"industrial CPU continuation","real":False,"reason":"Approved industrial training jobs = 0","controlled":"test_industrial.py::test_real_cpu_optimizer_with_only_fixture_data"},
 {"branch":"industrial label feedback, publish, reject and rollback","real":False,"reason":"Approved labels = 0; real industrial release policy remains unconfigured","controlled":"test_industrial.py::test_publish_then_rollback_on_disjoint_prospective_feedback and test_rejection_and_prospective_integrity and test_no_industrial_policy_never_publishes"},
 {"branch":"aggregate budget and unresolved-call isolation","real":False,"reason":"Actual run completed within budget without an unresolved call","controlled":"test_live_acceptance.py"},
]
summary={"date":"2026-11-19","tasks":{"4.3":"passed","5.4":"executed-with-coverage-limitations"},
         "scope":"user-approved decision connectivity experiment on a real event; original model selected industrial only",
         "lineage":report["lineage"],"training":{"jobs":1,"updates":2,"sequence_tokens_submitted":28796,
         "supervised_target_tokens":sum(value["metrics"]["token_count"] for key,value in training.items() if key.startswith("step-done/")),
         "snapshot_sha256":intent["snapshot_sha256"],"currency_cost":None},
         "inference":report["budget"],"replay":{"start":97,"last":240,"new_samples":144,"windows":windows,
         "candidate":finish,"final_pool":report["pool"],"stop_reason":report["stop_reason"]},
         "real_coverage":coverage,"limitations":["32-feature prediction is a proxy gate, not independent drift truth",
         "Agreement-based scoring is not independent drift accuracy","No real industrial training, publication, rollback, or label release",
         "Currency cost not provided by service; recorded token counts are not a monetary invoice"],"test_count":85}
exports={"training-records.json":training,"inference-records.json":inference,"stream-records.json":stream,
         "evidence-records.json":evidence,"review-records.json":review,"acceptance-summary.json":summary}
for name,value in exports.items():
    text=json.dumps(value,ensure_ascii=False,indent=2)+"\n"
    require(not auth_state.api_key or auth_state.api_key not in text,"Credential found; export withheld")
    target=directory/name
    if target.exists():
        require(target.read_text()==text,"Immutable export differs: "+name)
    else:
        with target.open("x") as handle:handle.write(text)
files={}
for path in sorted(directory.glob("*.json")):
    if path.name != "export-manifest.json":
        files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in [*root.glob("scripts/*model1*.py"),root/"src/ftsaki_harness/live_acceptance.py",
             root/"src/ftsaki_harness/training_intent.py",root/"src/ftsaki_harness/replay_handoff.py"]:
    files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in [runs/"remaining-tests-proposal-2026-11-19.json",runs/"remaining-tests-grant-2026-11-19.json",
             runs/"remaining-tests-token-cases-2026-11-19.json"]:
    files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest={"files":files,"logical_snapshot_sha256":{name:digest(value) for name,value in exports.items()},
          "credential_scan":"passed","audit":"passed","remote_calls_during_export":0}
text=json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"
path=directory/"export-manifest.json"
if not path.exists():
    with path.open("x") as handle:handle.write(text)
print(canonical({"audit":"passed","summary":str(directory/"acceptance-summary.json"),"budget":summary["inference"],
                 "candidate":finish,"stable_versions":[key for key,value in report["pool"]["models"].items() if value["online"] and value["stable"]]}))
