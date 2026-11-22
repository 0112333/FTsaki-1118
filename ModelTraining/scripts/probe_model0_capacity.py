"""One-shot bounded PyTRiO capacity probe; refuses to replay a prior invocation."""
import datetime, hashlib, json, math, time
from pathlib import Path
from transformers import AutoTokenizer
import pytrio
from pytrio.config.config import configuration
from model0_messages import ROOT, require_accepted, manifest_hash
from model0_runtime import runtime_messages
from model0_response import response_text
from model0_evaluation import validate_output_policy

OUT=ROOT/"artifacts/preflight/a_forecast_v2"
DATA=ROOT.parent/"data/model0_episodes_v1_1"
def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def main():
    plan=json.loads((OUT/"probe-plan.json").read_text())
    frozen_path=OUT/"frozen-contract.json"
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()==plan["contract_sha256"]
    frozen=json.loads(frozen_path.read_text())
    require_accepted(DATA,ROOT/"artifacts/acceptance/model0_episodes_v1_1/acceptance.json")
    assert manifest_hash(DATA)==frozen["manifest_sha256"]
    for name,sha in frozen["files"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for name,sha in plan.get("execution_files",{}).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    assert plan["max_calls"]<=12
    assert plan["max_billable_text_tokens"]==sum(c["prompt_tokens"]+c["max_output_tokens"] for c in plan["cases"])
    marker=OUT/"probe-started.json"
    with marker.open("x") as f:
        json.dump({"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"plan":plan},f,ensure_ascii=False,indent=2)
    started=time.monotonic();deadline=started+plan["deadline_seconds"]
    configuration.retry_max_attempts=1
    configuration.sampling_retry_timeout_seconds=0
    configuration.control_retry_timeout_seconds=0
    configuration.actor_capacity_wait_timeout_seconds=120
    configuration.sampling_timeout_seconds=plan["deadline_seconds"]
    report={"status":"starting","calls_submitted":0,"input_tokens":0,"output_tokens":0,"results":[],
       "training_started":False,"real_capacity_passed":False,"actual_currency_cost":None,
       "pricing_verified":False,"team_upload_pending":True}
    def save():dump(OUT/"probe-result.json",report)
    save()
    run=None
    try:
        import swanlab
        run=swanlab.init(workspace="ft_sakiko",project="model0-training",public=False,
            name="capacity-probe-a-forecast-v2",job_type="capacity-probe",mode="online",on_online_error="raise",
            log_dir=str(ROOT/"artifacts/swanlog"),settings=swanlab.Settings(interactive=False,
                probe={"hardware":False,"runtime":False,"requirements":False,"git":False,"monitor":False}),
            config={"message_version":frozen["message_version"],"contract_sha256":plan["contract_sha256"],
                "max_calls":plan["max_calls"],"max_text_tokens":plan["max_billable_text_tokens"],"training_started":False})
        report["swanlab_run_id"]=run.id;save()
        tokenizer=AutoTokenizer.from_pretrained(ROOT/"artifacts/preflight/tokenizer",local_files_only=True)
        client=pytrio.ServiceClient()
        assert client.workspace_cuid==plan["workspace"]
        sampler=client.create_sampling_client(base_model=plan["model"])
        report["sampling_task_id"]=sampler.sampling_run_cuid;save()
        lookup={}
        for split in ("train","dev"):
            with (DATA/(split+".jsonl")).open() as f:
                lookup.update((e["episode_id"],e) for e in map(json.loads,f))
        for index,case in enumerate(plan["cases"]):
            if time.monotonic()>=deadline:raise TimeoutError("Stage deadline reached")
            e=lookup[case["episode_id"]]
            ids=tokenizer.apply_chat_template(runtime_messages(e["input"],e["task"]),
                tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if hasattr(ids,"keys"):ids=ids["input_ids"]
            assert len(ids)==case["prompt_tokens"]
            assert len(ids)+case["max_output_tokens"]<=65536
            report["status"]="sampling";report["calls_submitted"]+=1
            report["current_episode_id"]=case["episode_id"];save()
            response=sampler.sample(prompt=pytrio.ModelInput.from_ints(ids),num_samples=1,
                sampling_params=pytrio.SamplingParams(max_tokens=case["max_output_tokens"],temperature=0,seed=42)
                ).result(timeout=max(1,deadline-time.monotonic()))
            dump(OUT/("probe-response-%02d.json"%index),response.model_dump(mode="json"))
            report["input_tokens"]+=response.input_tokens;report["output_tokens"]+=response.output_tokens
            sequence=response.sequences[0];text=response_text(sequence.model_dump(mode="json"),tokenizer)
            valid=False;reason=None
            try:
                answer=json.loads(text)
                if e["task"]=="forecast_next":
                    valid=set(answer)=={"horizon","feature_order","values"} and answer["horizon"]==32 and answer["feature_order"]==e["input"]["feature_order"] and len(answer["values"])==32 and all(
                        isinstance(row,list) and len(row)==len(e["input"]["feature_order"]) and all(
                            isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in row) for row in answer["values"])
                else:
                    valid=set(answer)=={"decision","evidence_refs","hypotheses"} and answer["decision"] in {"drift","no_drift","insufficient"} and isinstance(answer["evidence_refs"],list) and set(answer["evidence_refs"])<=set(e["input"]["evidence_refs"]) and isinstance(answer["hypotheses"],list)
                if valid: validate_output_policy(e,text)
                if not valid:reason="Output contract mismatch"
            except (ValueError,TypeError,KeyError) as ex:
                valid=False;reason=type(ex).__name__
            r={"episode_id":case["episode_id"],"split":case["split"],"scenario":case["scenario"],"task":case["task"],
               "independent_dev_metric":case["independent_dev_metric"],"valid_complete_output":valid,
               "stop_reason":sequence.stop_reason,"input_tokens":response.input_tokens,"output_tokens":response.output_tokens,"failure":reason}
            report["results"].append(r);save()
            swanlab.log({"probe/complete_output":int(valid),"probe/input_tokens":response.input_tokens,
                "probe/output_tokens":response.output_tokens,"probe/calls":report["calls_submitted"]})
            print(json.dumps(r),flush=True)
            if not valid:
                report["status"]="blocked_incomplete_output";save();return
        report["status"]="complete";report["real_capacity_passed"]=True
    except Exception as ex:
        report["status"]="blocked_service_error";report["error_type"]=type(ex).__name__
        report["error_code"]=getattr(ex,"code",None)
        import traceback
        report["error_location"]=[{"file":Path(f.filename).name,"line":f.lineno,"function":f.name} for f in traceback.extract_tb(ex.__traceback__)]
        # No arbitrary error string: server exceptions may carry sensitive request details.
    finally:
        report["elapsed_seconds"]=time.monotonic()-started;save()
        if run is not None:
            try:
                swanlab.log({"probe/real_capacity_passed":int(report["real_capacity_passed"]),
                    "probe/elapsed_seconds":report["elapsed_seconds"],"training/started":0})
                swanlab.finish()
                remote=swanlab.Api().run("ft_sakiko/model0-training/"+run.id)
                report["swanlab_upload_verified"]=remote.name=="capacity-probe-a-forecast-v2" and remote.state=="FINISHED"
            except Exception as log_error:
                report["swanlab_upload_verified"]=False;report["swanlab_error_type"]=type(log_error).__name__
            save()
        print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)
if __name__=="__main__":main()
