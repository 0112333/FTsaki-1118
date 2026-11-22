"""Bounded full-dev baseline, gated on the completed same-protocol capacity probe."""
import collections,concurrent.futures,datetime,hashlib,json,time
from pathlib import Path
from transformers import AutoTokenizer
import pytrio,swanlab
from pytrio.config.config import configuration
from model0_messages import ROOT,require_accepted,manifest_hash
from model0_runtime import runtime_messages
from model0_response import response_text
from model0_evaluation import score
OUT=ROOT/"artifacts/preflight/a_forecast_v2"
DATA=ROOT.parent/"data/model0_episodes_v1_1"
RESULTS=OUT/"baseline"
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def aggregate(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r["scenario"],r["task"])].append(r)
    summaries=[]
    for (scenario,task),items in sorted(groups.items()):
        valid=[r for r in items if r["valid"]]
        result={"scenario":scenario,"task":task,"count":len(items),"valid":len(valid),"format_rate":len(valid)/len(items)}
        if task=="forecast_next":
            result.update(gate_passed=sum(r.get("outcome")=="passed" for r in items),
                mean_error_valid=sum(r["error"] for r in valid)/len(valid) if valid else None,
                mean_trend_valid=sum(r["trend_match"] for r in valid)/len(valid) if valid else None)
        else:
            labels=("drift","no_drift","insufficient")
            matrix={a:{b:0 for b in (*labels,"invalid")} for a in labels}
            for r in items:matrix[r["truth"]][r["prediction"] if r["valid"] else "invalid"]+=1
            per_class={}
            for label in labels:
                tp=matrix[label][label];fp=sum(matrix[a][label] for a in labels if a!=label)
                fn=sum(v for k,v in matrix[label].items() if k!=label)
                per_class[label]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
            negatives=sum(sum(matrix[a].values()) for a in labels if a!="drift")
            positives=sum(matrix["drift"].values())
            result.update(confusion_matrix=matrix,per_class_f1=per_class,
                macro_f1=sum(x for x in per_class.values() if x is not None)/sum(x is not None for x in per_class.values()) if any(x is not None for x in per_class.values()) else None,
                drift_false_positive_rate=sum(matrix[a]["drift"] for a in labels if a!="drift")/negatives if negatives else None,
                drift_false_negative_rate=(positives-matrix["drift"]["drift"])/positives if positives else None,
                insufficient_rate=sum(matrix[a]["insufficient"] for a in labels)/len(items))
        summaries.append(result)
    return summaries
def main():
    plan=json.loads((OUT/"baseline-plan.json").read_text());probe=json.loads((OUT/"probe-result.json").read_text())
    entry=json.loads((OUT/"trial-entry-acceptance.json").read_text())
    if not (entry.get("user_approved") and entry.get("service_call_passed") and entry.get("sequence_lengths_passed")):raise ValueError("Interface capacity gate not passed")
    assert entry["contract_sha256"]==plan["contract_sha256"]
    assert hashlib.sha256((ROOT/"scripts/model0_response.py").read_bytes()).hexdigest()==entry["decoder_sha256"]
    for name,sha in plan.get("execution_files",{}).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    frozen_path=OUT/"frozen-contract.json";frozen=json.loads(frozen_path.read_text())
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()==plan["contract_sha256"]
    require_accepted(DATA,ROOT/"artifacts/acceptance/model0_episodes_v1_1/acceptance.json")
    assert manifest_hash(DATA)==frozen["manifest_sha256"]
    for name,digest in frozen["files"].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/"started.json").open("x") as f:json.dump(plan,f,indent=2)
    deadline=time.monotonic()+plan["deadline_seconds"];rows=[];report={"status":"starting","rows":rows,"calls_submitted":0,"training_started":False}
    def save():dump(RESULTS/"result.json",report)
    def consume(e,response,reused=False):
        s=response["sequences"][0];text=response_text(s,tokenizer)
        result={"episode_id":e["episode_id"],"scenario":e["scenario_id"],"task":e["task"],
            "truth":e["target"].get("decision"),"reused_probe":reused,
            "stop_reason":s["stop_reason"],"input_tokens":response["input_tokens"],"output_tokens":response["output_tokens"]}
        try:result.update(score(e,text))
        except (ValueError,KeyError,TypeError):result.update(valid=False,prediction="invalid")
        rows.append(result)
    run=None
    try:
        tokenizer=AutoTokenizer.from_pretrained(ROOT/"artifacts/preflight/tokenizer",local_files_only=True)
        with (DATA/"dev.jsonl").open() as f:episodes={e["episode_id"]:e for e in map(json.loads,f)}
        for index,r in enumerate(probe["results"]):
            if r["split"]=="dev":
                consume(episodes[r["episode_id"]],json.loads((OUT/("probe-response-%02d.json"%index)).read_text()),True)
        run=swanlab.init(workspace="ft_sakiko",project="model0-training",public=False,name="baseline-messages-v3",
            job_type="baseline",mode="online",on_online_error="raise",log_dir=str(ROOT/"artifacts/swanlog"),
            settings=swanlab.Settings(interactive=False,probe={"hardware":False,"runtime":False,"requirements":False,"git":False,"monitor":False}),
            config={"contract_sha256":plan["contract_sha256"],"max_calls":plan["max_calls"],"max_text_tokens":plan["max_billable_text_tokens"],
                "deadline_seconds":plan["deadline_seconds"],"training_started":False})
        report["swanlab_run_id"]=run.id;save()
        configuration.retry_max_attempts=1;configuration.control_retry_timeout_seconds=0
        configuration.sampling_retry_timeout_seconds=0;configuration.sampling_timeout_seconds=plan["deadline_seconds"]
        configuration.actor_capacity_wait_timeout_seconds=120
        client=pytrio.ServiceClient();assert client.workspace_cuid==plan["workspace"]
        sampler=client.create_sampling_client(base_model=plan["model"])
        report["sampling_run_id"]=sampler.sampling_run_cuid
        done={r["episode_id"] for r in rows}
        pending=[c for c in plan["cases"] if c["episode_id"] not in done]
        def sample(case):
            e=episodes[case["episode_id"]]
            ids=tokenizer.apply_chat_template(runtime_messages(e["input"],e["task"]),tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if hasattr(ids,"keys"):ids=ids["input_ids"]
            assert len(ids)==case["prompt_tokens"] and len(ids)+case["max_output_tokens"]<=65536
            response=sampler.sample(pytrio.ModelInput.from_ints(ids),num_samples=1,
                sampling_params=pytrio.SamplingParams(max_tokens=case["max_output_tokens"],temperature=0,seed=42)).result(timeout=max(1,deadline-time.monotonic()))
            return response.model_dump(mode="json")
        with concurrent.futures.ThreadPoolExecutor(max_workers=plan["concurrency"]) as pool:
            for start in range(0,len(pending),plan["concurrency"]):
                if time.monotonic()>=deadline:raise TimeoutError("Baseline deadline")
                batch=pending[start:start+plan["concurrency"]]
                report["calls_submitted"]+=len(batch);report["status"]="sampling";save()
                futures={pool.submit(sample,c):c for c in batch}
                error=None
                for future in concurrent.futures.as_completed(futures):
                    case=futures[future]
                    try:
                        response=future.result();dump(RESULTS/(case["episode_id"]+".json"),response)
                        consume(episodes[case["episode_id"]],response);save()
                        swanlab.log({"baseline/completed":len(rows),"baseline/format_valid":int(rows[-1]["valid"])})
                        print(json.dumps({"completed":len(rows),"episode_id":case["episode_id"],"valid":rows[-1]["valid"]}),flush=True)
                    except Exception as ex:
                        error=ex;report["error_type"]=type(ex).__name__;report["error_code"]=getattr(ex,"code",None);save()
                if error is not None:raise error
        assert len(rows)==244
        report["status"]="complete";report["groups"]=aggregate(rows)
        report["coverage"]={"A_forecast":"N/A: no independent dev","A_binary_decide":"N/A: only insufficient targets",
            "drift_supervision":"Controlled protocol only; natural drift generalization unverified"}
    except Exception as ex:
        report["status"]="blocked_error";report["error_type"]=type(ex).__name__;report["error_code"]=getattr(ex,"code",None)
    finally:
        report["actual_input_tokens"]=sum(r["input_tokens"] for r in rows)
        report["actual_output_tokens"]=sum(r["output_tokens"] for r in rows)
        report["actual_currency_cost"]=None;save()
        if run is not None:
            swanlab.log({"baseline/complete":int(report["status"]=="complete"),"baseline/records":len(rows),"training/started":0});swanlab.finish()
        print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
if __name__=="__main__":main()
