"""Execute approved child reload and causal stage-3 replay; resume without resending."""
import argparse
from itertools import islice
import json
from pathlib import Path

import pytrio
from pytrio.config.config import auth_state, configuration
from transformers import AutoTokenizer

from ftsaki_harness.data import Handoff, read_jsonl
from ftsaki_harness.decision_replay import render
from ftsaki_harness.forecast import Scale
from ftsaki_harness.inference import validate_content, validate_hypotheses
from ftsaki_harness.live_acceptance import BoundedInference
from ftsaki_harness.model0_smoke import read, require
from ftsaki_harness.policy import canonical, digest
from ftsaki_harness.pool import Pool
from ftsaki_harness.prospective import ProspectiveRun
from ftsaki_harness.pytrio_training import PyTrioTraining
from ftsaki_harness.records import Records
from ftsaki_harness.replay_handoff import model1_candidate, prospective_sample, reviewed_anchor
from ftsaki_harness.training_intent import TrainingIntent
from ftsaki_harness.window_evidence import load_runtime, window_stream


class PrefixReplay:
    def __init__(self, rows):
        self.rows, self.cursor = rows, 0

    def batches(self):
        for begin in range(0,len(self.rows),32):
            yield self.rows[begin:begin+32]

    def commit(self,count):
        self.cursor += count


def choice(raw):
    return json.loads(raw)["choices"][0]


def vote(receipt, raw, evidence):
    if receipt.get("http_status") != 200 or receipt.get("format_valid") is not True:
        return "insufficient"
    output = choice(raw)
    validated = validate_content("decide",evidence,output["text"],output["finish_reason"])
    validate_hypotheses(validated,evidence)
    return validated["decision"]


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase",choices=("reload","replay"))
args=parser.parse_args()
root=Path.cwd()
runs=root/"runs"
directory=runs/"stage23-C-model1-2026-11-19"
proposal=read(runs/"remaining-tests-proposal-2026-11-19.json")
grant=read(runs/"remaining-tests-grant-2026-11-19.json")
require(grant["approval"] and grant["proposal_sha256"]==digest(proposal),"Approved proposal changed")
require(grant["max_inference_calls"]==11 and grant["max_inference_tokens"]==60014,"Approved inference budget changed")
event=proposal["event"]
base=read(root/"config/model0-smoke.json")
review=Records(runs/"2a-C-first-device-2026-11-19/review.sqlite")
training=Records(directory/"training.sqlite")
aggregate=Records(directory/"inference.sqlite")
stream_store=Records(directory/"stream.sqlite")
inputs=Records(directory/"evidence.sqlite")
producer,context,_,builder,manifest=load_runtime((root.parent/"data").resolve(),(root.parent/"ModelTraining").resolve(),"C",base["contract_sha256"])
handoff=Handoff(root.parent/"data/harness_data_v1/manifest.json")
observation_path=handoff.descriptor(handoff.scenario("C")["splits"]["closed_loop"]["observations"])
# Bound all pre-forecast reads to the already committed 96-position prefix.
prefix=list(islice(read_jsonl(observation_path),96))
require(len(prefix)==96 and len({row["group_id"] for row in prefix})==1,"Changed source prefix")
anchor=reviewed_anchor(review,iter(prefix),manifest=manifest,scenario="C",feature_order=context["order"],run=proposal["replay"]["run"])
require(anchor["cursor"]==96 and anchor["group"]==proposal["replay"]["group"],"Changed reviewed boundary")
order,training_rows=handoff.training_features("C")
stream=ProspectiveRun(stream_store,run=anchor["run"],scenario="C",source="closed_loop",manifest=manifest,
                      feature_order=order,scale=Scale.fit(training_rows),start=96,parent_artifact=anchor["parent_artifact"],anchor=anchor)
stream_store.commit({"anchor":anchor,"grant_sha256":digest(grant)})
tokenizer=AutoTokenizer.from_pretrained(root.parent/"ModelTraining/artifacts/preflight/tokenizer",local_files_only=True)
network=BoundedInference(aggregate,directory,grant,base,tokenizer,auth_state.api_key)
run=review.get("identity")["run"]
job=training.get(f"submission-result/{run}/{event}")
require(job and job["status"]=="artifact-ready","Verified training artifact required")
windows=window_stream(inputs,producer,context,PrefixReplay(prefix),builder)
for record,evidence,messages in windows:
    pass
require(record["as_of"]["replay_order"]==96,"Missing reload boundary")
try:
    if args.phase=="reload":
        _,preflight=render(tokenizer,messages)
        require(preflight==proposal["reload_prompt"],"Reload prompt changed")
        reload_grant={"approval":grant["approval"],"artifact":job["artifact"],"event":event,
                      "prompt_sha256":preflight["prompt_sha256"],"max_output_tokens":256,
                      "max_tokens":preflight["prompt_tokens"]+256,"deadline_seconds":120,"deadline":grant["inference_deadline"]}
        spec={"artifact":job["artifact"],"checkpoint_id":job["checkpoint_id"],"task":"decide",
              "max_output_tokens":256,"deadline_seconds":120,"messages_sha256":digest(messages)}
        network.reserve("model1-reload",spec,preflight)
        probe=Records(directory/"model1-reload.sqlite")
        try:
            result=PyTrioTraining(TrainingIntent(training,review),pytrio,configuration).probe_model1(
                run,event,probe,base,messages,tokenizer,reload_grant,auth_state.api_key)
            receipt=probe.get("online/result")
            if receipt:
                aggregate.commit({"receipt/model1-reload":receipt})
            print(canonical({"phase":"reload","result":result,"budget":network.summary()}),flush=True)
        finally:
            probe.close()
    else:
        ready=stream_store.get("ready/1") or model1_candidate(stream,anchor,review,training,event)
        if not stream_store.get("forecast-response/1"):
            require(stream.cursor==96,"Candidate forecast must precede unseen input")
            forecast_evidence=dict(evidence,recent_history={"group_steps":list(range(65,97)),"values":anchor["history"]})
            forecast_messages=builder(forecast_evidence,"forecast_next")
            _,preflight=render(tokenizer,forecast_messages)
            require(preflight==proposal["candidate_forecast_prompt"],"Frozen candidate prompt changed")
            receipt,raw=network.request("model1-forecast-96",forecast_messages,task="forecast_next",
                                       artifact=ready["artifact"],checkpoint_id=ready["checkpoint_id"],output_tokens=8192,seconds=600)
            if receipt.get("http_status")==200:
                output=choice(raw)
                result=stream.accept_forecast_response(ready,output["text"],output["finish_reason"])
            else:
                result=stream.accept_forecast_response(ready,error="transport-error")
            print(canonical({"phase":"forecast","receipt":receipt,"result":result,"budget":network.summary()}),flush=True)
        # This read happens only after the candidate response and its terminal/frozen state are durable.
        require(stream_store.get("forecast-response/1"),"Persist forecast response before advancing source")
        stop=proposal["replay"]["stop_at"]
        rows=list(islice(read_jsonl(observation_path),stop))
        require(len(rows)==stop and len({row["group_id"] for row in rows})==1,"Authorized group ended early")
        windows=window_stream(inputs,producer,context,PrefixReplay(rows),builder)
        allowed={item["cutoff"]:item for item in review.get("replay-plan")["windows"] if item["cutoff"]>96}
        for record,evidence,messages in windows:
            cutoff=evidence["as_of"]["replay_order"]
            if cutoff<=stream.cursor:
                continue
            require(cutoff in proposal["replay"]["window_cutoffs"] and cutoff in allowed,"Unregistered window")
            require(record["messages_sha256"]==allowed[cutoff]["messages_sha256"],"Changed registered window evidence")
            require(render(tokenizer,messages)[1]["prompt_tokens"]==allowed[cutoff]["prompt_tokens"],"Changed registered prompt size")
            pool=Pool.restore(stream.state["pool"])
            votes={}
            for version in pool.weights():
                model=pool.models[version]
                require(model.online,"Offline pool model")
                artifact=anchor["parent_artifact"] if version==0 else stream_store.get(f"ready/{version}")["artifact"]
                checkpoint=review.get("replay-plan")["checkpoint_id"] if version==0 else ready["checkpoint_id"]
                receipt,raw=network.request(f"model{version}-decide-{cutoff}",messages,task="decide",artifact=artifact,
                                           checkpoint_id=checkpoint,output_tokens=256,seconds=120)
                votes[version]=vote(receipt,raw,evidence)
            start=stream.cursor
            batch=[prospective_sample("C",order,row) for row in rows[start:cutoff]]
            result=stream.advance(batch,votes)
            stream_store.commit({f"live-votes/{cutoff}":{str(k):v for k,v in votes.items()}})
            print(canonical({"phase":"window","cutoff":cutoff,"votes":{str(k):v for k,v in votes.items()},
                             "result":result,"budget":network.summary()}),flush=True)
        ending=stream.end()
        state=Pool.restore(stream.state["pool"])
        report={"run":anchor["run"],"source":"live-http","cursor":stream.cursor,"stop_reason":"authorized-range-complete",
                "candidate":stream_store.get("finish/1"),"pool":state.dump(),"end":ending,"budget":network.summary(),
                "training_submissions":{"decision":1,"industrial":0},"label_releases":0,
                "lineage":{"parent":job["parent"],"model1":job["artifact"],"task":job["handle"],"checkpoint_id":job["checkpoint_id"]}}
        stream_store.commit({"report":report})
        (directory/"live-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print(canonical({"phase":"complete","report":report}),flush=True)
finally:
    for database in (review,training,aggregate,stream_store,inputs):
        database.close()
