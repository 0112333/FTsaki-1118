"""One bounded trial. Fails closed before creating a client unless baseline/capacity passed."""
import hashlib,json,math,time
from pathlib import Path
import pytrio,swanlab
from pytrio.config.config import configuration
from transformers import AutoTokenizer
from model0_messages import ROOT,require_accepted,manifest_hash,token_sample
OUT=ROOT/"artifacts/preflight/a_forecast_v2";DATA=ROOT.parent/"data/model0_episodes_v1_1"
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def build_datum(sample):
    # PyTRiO cross_entropy uses sum reduction. Normalize each example's answer
    # weights to sum to 1 so a long matrix does not scale its step by token count.
    count=sample["assistant_tokens"]
    if count<=0 or sum(sample["weights"])!=count:raise ValueError("Invalid assistant mask")
    return pytrio.Datum(model_input=pytrio.ModelInput.from_ints(sample["input_tokens"]),
        loss_fn_inputs={"target_tokens":sample["target_tokens"],"weights":[w/count for w in sample["weights"]]})
def main():
    entry=json.loads((OUT/"trial-entry-acceptance.json").read_text())
    baseline=json.loads((OUT/"baseline/result.json").read_text())
    if not (entry.get("user_approved") and entry.get("service_call_passed") and entry.get("sequence_lengths_passed")) or baseline["status"]!="complete" or len(baseline["rows"])!=244:
        raise ValueError("Capacity/baseline gates are incomplete")
    expected_ids=set()
    with (DATA/"dev.jsonl").open() as f:
        expected_ids={e["episode_id"] for e in map(json.loads,f)}
    if len(baseline["rows"])!=len({r["episode_id"] for r in baseline["rows"]}) or {r["episode_id"] for r in baseline["rows"]}!=expected_ids:
        raise ValueError("Baseline coverage mismatch")
    acceptance=json.loads((OUT/"baseline/acceptance.json").read_text())
    if not acceptance.get("accepted") or acceptance["result_sha256"]!=hashlib.sha256((OUT/"baseline/result.json").read_bytes()).hexdigest():
        raise ValueError("Baseline acceptance missing or changed")
    require_accepted(DATA,ROOT/"artifacts/acceptance/model0_episodes_v1_1/acceptance.json")
    candidate=json.loads((OUT/"training-plan-candidate.json").read_text())
    frozen_path=OUT/"frozen-contract.json";frozen=json.loads(frozen_path.read_text())
    assert candidate["input_contract_sha256"]==hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert candidate["manifest_sha256"]==manifest_hash(DATA)
    assert entry["contract_sha256"]==candidate["input_contract_sha256"]
    assert acceptance["contract_sha256"]==candidate["input_contract_sha256"]
    for name,sha in frozen["files"].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    trial=candidate["trial"];assert trial["rank"]==16 and trial["max_optim_steps"]==16 and trial["batch_size"]==1
    plan={**trial,"max_forward_backward_calls":16,"max_optim_steps":16,"max_save_calls":18,
        "max_sequence_tokens":trial["max_sequence_token_sum"],"deadline_seconds":14400,
        "loss_normalization":"sum of assistant token weights per example = 1","auto_shift":False,
        "base_model":candidate["base_model"],"seed":42,"fresh_lora":True,
        "workspace":"workspace_1o9brly7x8y1","contract_sha256":candidate["input_contract_sha256"],
        "execution_files":{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ["scripts/train_model0_trial.py","scripts/model0_messages.py","scripts/model0_input.py"]},
        "pricing_verified":False,"budget_source":"Existing user-authorized quota; stop and contact on exhaustion"}
    dest=ROOT/"artifacts/training/trial_messages_v3";dest.mkdir(parents=True,exist_ok=True)
    with (dest/"run-plan.json").open("x") as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    report={"status":"starting","training_started":False,"updates":[],"forward_backward_calls":0,"sequence_tokens_submitted":0}
    def save():dump(dest/"result.json",report)
    deadline=time.monotonic()+plan["deadline_seconds"]
    def result(future):
        if time.monotonic()>=deadline:raise TimeoutError("Trial deadline")
        return future.result(timeout=max(1,deadline-time.monotonic()))
    run=None
    try:
        tokenizer=AutoTokenizer.from_pretrained(ROOT/"artifacts/preflight/tokenizer",local_files_only=True)
        with (DATA/"train.jsonl").open() as f:lookup={e["episode_id"]:e for e in map(json.loads,f)}
        samples=[token_sample(lookup[r["episode_id"]],tokenizer) for r in trial["episodes"]]
        assert len(samples)<=16 and sum(s["sequence_tokens"] for s in samples)<=plan["max_sequence_tokens"]
        datums=[build_datum(s) for s in samples] # SDK schema validation before any paid call.
        run=swanlab.init(workspace="ft_sakiko",project="model0-training",public=False,name="trial-messages-v3",
            job_type="train",mode="online",on_online_error="raise",log_dir=str(ROOT/"artifacts/swanlog"),
            settings=swanlab.Settings(interactive=False,probe={"hardware":False,"runtime":False,"requirements":False,"git":False,"monitor":False}),config=plan)
        report["swanlab_run_id"]=run.id;save()
        configuration.retry_max_attempts=1;configuration.training_retry_timeout_seconds=0;configuration.control_retry_timeout_seconds=0
        configuration.actor_capacity_wait_timeout_seconds=120
        client=pytrio.ServiceClient();assert client.workspace_cuid==plan["workspace"]
        trainer=client.create_lora_training_client(base_model=plan["base_model"],rank=16,seed=42)
        report["training_run_id"]=trainer.training_run_cuid
        report["initial_checkpoint"]=result(trainer.save_state("initial")).model_dump(mode="json");save()
        initial=report["initial_checkpoint"]
        if not initial.get("path") or initial.get("model")!=plan["base_model"] or initial.get("size",0)<=0:
            raise ValueError("Initial checkpoint artifact is not valid")
        capabilities={"training_run_id":trainer.training_run_cuid,"base_model":plan["base_model"],
            "rank":16,"learning_rate":5e-5,"workspace":plan["workspace"],"lora_initialization_passed":True,
            "initial_checkpoint":report["initial_checkpoint"],"python":"/home/awa_0112333/FTsaki/bin/python"}
        dump(ROOT/"artifacts/environment/training-capabilities.json",capabilities)
        dump(dest/"readiness.json",{"passed":True,"baseline_acceptance_sha256":hashlib.sha256((OUT/"baseline/acceptance.json").read_bytes()).hexdigest(),
            "contract_sha256":plan["contract_sha256"],"actual_lora_initialization_passed":True,"trial_limits":plan})
        tasks=ROOT/"openspec/changes/train-model0-with-pytrio/tasks.md"
        text=tasks.read_text()
        for task in ("2.4","3.8","4.1"):text=text.replace("- [ ] "+task+" ","- [x] "+task+" ")
        tasks.write_text(text)
        for index,(r,s,datum) in enumerate(zip(trial["episodes"],samples,datums)):
            if time.monotonic()>=deadline:raise TimeoutError("Trial deadline")
            report["status"]="forward_backward";report["training_started"]=True;report["forward_backward_calls"]+=1
            report["sequence_tokens_submitted"]+=s["sequence_tokens"];report["current_episode_id"]=r["episode_id"];save()
            fb=result(trainer.forward_backward([datum],loss_fn="cross_entropy",auto_shift=False))
            if not all(math.isfinite(v) for v in fb.metrics.values()):raise ValueError("Nonfinite training metric")
            report["status"]="optim_step_pending";save()
            opt=result(trainer.optim_step(pytrio.AdamParams(learning_rate=5e-5,grad_clip_norm=1.0)))
            update={"step":index+1,"episode_id":r["episode_id"],"metrics":fb.metrics,"optim":opt.model_dump(mode="json")}
            report["updates"].append(update);report["status"]="checkpoint_pending";save()
            update["checkpoint"]=result(trainer.save_state("step-%02d"%(index+1))).model_dump(mode="json");save()
            swanlab.log({"train/step":index+1,"train/assistant_tokens":s["assistant_tokens"],
                "train/sequence_tokens":s["sequence_tokens"],**{"train/"+k:v for k,v in fb.metrics.items()}})
            print(json.dumps({"step":index+1,"episode_id":r["episode_id"],"metrics":fb.metrics}),flush=True)
        report["sampler_weights"]=result(trainer.save_weights_for_sampler("trial-final")).model_dump(mode="json")
        report["status"]="trained_reload_and_dev_pending"
    except Exception as ex:
        report["status"]="blocked_error";report["error_type"]=type(ex).__name__;report["error_code"]=getattr(ex,"code",None)
    finally:
        save()
        if run is not None:swanlab.finish()
        print(json.dumps({k:v for k,v in report.items() if k!="updates"}),flush=True)
if __name__=="__main__":main()
