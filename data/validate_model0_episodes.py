"""Read-only, source-recomputing acceptance validator for model0 episodes."""
import argparse
import re
from model0_data import *

def require(condition,message):
    if not condition:raise ValueError(message)

def close(actual,expected,path="root"):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),"object fields: "+path)
        for k,v in expected.items():close(actual[k],v,path+"."+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),"list shape: "+path)
        for i,(a,b) in enumerate(zip(actual,expected)):close(a,b,path+"["+str(i)+"]")
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        require(type(actual) in (int,float) and math.isfinite(actual),"finite number: "+path)
        require(math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-11),"numeric mismatch: "+path)
    else:
        require(type(actual)==type(expected) and actual==expected,"value mismatch: "+path)

class Validator:
    def __init__(self,output):
        self.output=output
        self.ctx=contexts()
        # Independent full-source recomputation; never use reference_rows for scale validation.
        from statistics import fmean, pstdev
        for c in self.ctx.values():
            full=[vector(row) for row in rows(c["trainpath"])]
            cols=list(zip(*full))
            independently={"sample_count":len(full),"feature_mean":[fmean(col) for col in cols],"feature_std":[pstdev(col) for col in cols]}
            close(c["scale"],independently,"independent full industrial scale")
        self.m=read_json(output/"manifest.json")
        self.files={};self.hashes={};self.json_cache={};self.row_cache={}
        self.assignments=read_json(output/"protocols/split_assignment.json")["groups"]
        require(self.assignments==assignments(self.ctx),"split assignment differs from frozen rule")
        self.bygroup={a["group_key"]:a for a in self.assignments}
        self.obs={s:{r["id"]:r for r in c["observations"]} for s,c in self.ctx.items()}
        self.label_truth={s:{r["id"]:r["answer"] for r in rows(c["labelpath"])} for s,c in self.ctx.items()}
        self.pred_verified=set()
        self.natural_cache={}
        self.ids=set();self.families={};self.source_splits={}
        self.check_counts=collections.Counter()

    def desc(self,desc):
        require(isinstance(desc,dict) and bool(re.fullmatch("[0-9a-f]{64}",desc.get("sha256",""))),"invalid hash descriptor")
        p=resolve(desc["path"])
        if p not in self.hashes:self.hashes[p]=digest(p)
        require(self.hashes[p]==desc["sha256"],"hash mismatch "+str(p))
        return p

    def readrows(self,desc):
        p=self.desc(desc)
        if p not in self.row_cache:self.row_cache[p]=list(rows(p))
        return self.row_cache[p]

    def readjson(self,desc):
        p=self.desc(desc)
        if p not in self.json_cache:self.json_cache[p]=read_json(p)
        return self.json_cache[p]

    def initial(self):
        require(self.m["schema_version"]=="model0.dataset.v1","manifest schema")
        require(self.m["episode_schema"]=="model0.episode.v1" and self.m["evidence_schema"]=="model0.evidence.v1","manifest schemas")
        require(self.desc(self.m["upstream_manifest"])==ROOT/"harness_data_v1/manifest.json","upstream manifest")
        for f in self.m["files"]:
            p=(self.output/f["path"]).resolve()
            require(p.is_relative_to(self.output),"manifest path traversal")
            require(digest(p)==f["sha256"],"manifest file hash "+f["path"])
            if f["row_count"] is not None:
                require(sum(1 for _ in rows(p))==f["row_count"],"manifest line count")
            require(p.stat().st_size==f["bytes"],"manifest byte count")
            require(f["path"] not in self.files,"duplicate manifest file")
            self.files[f["path"]]=f
        for n,p in protocols().items():
            require(read_json(self.output/"protocols"/(n+".json"))==p,"protocol drift "+n)
        for p in self.m["protocols"]:
            require(digest(self.output/p["path"])==p["sha256"],"protocol manifest hash")
        freeze=read_json(self.output/"snapshots/freeze.json")
        require(freeze["upstream_manifest"]==self.m["upstream_manifest"],"freeze upstream")
        require(freeze["created_at"]==self.m["created_at"],"freeze time")
        require(self.readjson(freeze["split_assignment"])["groups"]==self.assignments,"freeze assignment")
        for p in freeze["protocols"]:self.desc(p)
        for p in self.m["tooling"]:self.desc(p)
        require(freeze["tooling"]==self.m["tooling"],"freeze tooling")
        self.check_counts["manifest_protocol_freeze_checks"]+=1

    def natural(self,c,a,group,audit):
        pp=self.desc(audit["industrial_predictions"])
        if pp in self.natural_cache:return self.natural_cache[pp]
        s=c["s"];b=BATCH[s]
        predictions=self.readrows(audit["industrial_predictions"])
        releases=self.readrows(audit["label_release_log"])
        ledger=self.readrows(audit["replay_ledger"])
        require([p["id"] for p in predictions]==[r["id"] for r in group],"prediction sample order")
        require(len({r["id"] for r in releases})==len(releases),"duplicate releases")
        preds=[r["prediction"] for r in predictions]
        expected=c["model"].predict(np.asarray([vector(r) for r in group],dtype=np.float64)).tolist()
        close(preds,expected,"recomputed industrial predictions")
        left=LABEL_BUDGET;seq=0;expected_releases=[];expected_ledger=[]
        for start in range(0,len(group),b):
            end=min(start+b,len(group));order=group[end-1]["replay_order"];seq+=1
            ids=[r["id"] for r in group[start:end]]
            for i in range(start,end):
                require(predictions[i]["group_step"]==i+1 and predictions[i]["predicted_at_order"]==order
                        and predictions[i]["event_sequence"]==seq,"prediction event order")
            expected_ledger.append({"event_sequence":seq,"event":"predictions_saved","as_of_order":order,"sample_ids":ids})
            selected=sorted(ids,key=lambda sid:rank(42,s,a["source_group"],sid))[:min(math.ceil(len(ids)*.25),left)]
            left-=len(selected);seq+=1
            for sid in sorted(selected):
                expected_releases.append({"id":sid,"answer":self.label_truth[s][sid],"predicted_at_order":order,
                  "released_at_order":order,"train_authorized":False,"event_sequence":seq})
            expected_ledger.append({"event_sequence":seq,"event":"labels_released","as_of_order":order,"sample_ids":sorted(selected)})
        close(releases,expected_releases,"label release replay")
        close(ledger,expected_ledger,"causal ledger")
        self.natural_cache[pp]=(preds,releases)
        self.check_counts["industrial_predictions_and_release_logs_recomputed"]+=1
        return preds,releases

    def episode(self,r,role):
        require(set(r)=={"schema_version","record_kind","episode_id","task","scenario_id","split","input","target","audit"},"episode fields")
        require(r["schema_version"]=="model0.episode.v1" and r["record_kind"]=="episode","episode marker")
        require(r["episode_id"] not in self.ids,"duplicate episode ID");self.ids.add(r["episode_id"])
        s=r["scenario_id"];require(s in self.ctx,"unknown scenario");c=self.ctx[s];b=BATCH[s]
        require(r["task"] in ("decide","forecast_next"),"task")
        require(r["split"] in ("train","dev"),"split")
        audit=r["audit"];sup=audit["supervision"];target=r["target"]
        eligible=audit["eligibility"]["sft_eligible"]
        require(type(eligible) is bool,"eligibility bool")
        require((role=="quarantine" and not eligible) or (role==r["split"] and eligible),"file eligibility")
        require((target is not None)==eligible,"target eligibility")
        require(not audit["eligibility"]["reason_codes"] if eligible else bool(audit["eligibility"]["reason_codes"]),"reason codes")
        sa=audit["split_assignment"];require(sa["group_key"] in self.bygroup,"unknown group")
        a=self.bygroup[sa["group_key"]]
        require(a["scenario_id"]==s and a["split"]==r["split"],"group split")
        require(sa["source_family_id"]==a["source_family_id"] and sa["assigned_before_windowing"] is True,"source family")
        require(self.readjson(sa["assignment_artifact"])["groups"]==self.assignments,"assignment descriptor")
        fam=sa["source_family_id"]
        require(self.families.setdefault(fam,r["split"])==r["split"],"family crossing")
        group=[self.obs[s][sid] for sid in a["sample_ids"]]
        end=r["input"]["as_of"]["group_step"]
        require(type(end) is int and 0<end<=len(group) and (end%b==0 or end==len(group)),"cutoff batch boundary")
        require(sa["event_key"]==fam+"-"+str(end),"event lineage")
        cutoff=group[end-1]["replay_order"]
        visible=[position(s,row,i+1) for i,row in enumerate(group[:end])]
        close(audit["visible_observations"],visible,"visible positions")
        for row in visible:require(row["replay_order"]<=cutoff,"future observation leak")
        future=group[end:end+32] if r["task"]=="forecast_next" else []
        expected_future=[{**position(s,row,end+i+1),"relative_step":i+1} for i,row in enumerate(future)]
        close(audit["future_targets"],expected_future,"future positions")
        require(not {v["id"] for v in visible}&{v["id"] for v in expected_future},"future overlaps visible")
        require(audit["source_manifest"]==self.m["upstream_manifest"],"source manifest identity")
        require(self.desc(audit["industrial_artifact"])==c["checkpoint"],"industrial lineage")
        scale_art=self.readjson(audit["feature_scale_artifact"])
        require(self.desc(scale_art["source"])==c["trainpath"],"scale source")
        require(scale_art["role"]=="industrial_train" and scale_art["row_count"]==c["scale"]["sample_count"],"scale full count")
        require(scale_art["feature_order"]==c["order"],"scale order")
        close(scale_art["statistics"],c["scale"],"full-scale artifact")
        ref=self.readjson(audit["reference_snapshot"])
        require(ref["sample_ids"]==[v["id"] for v in c["ref_rows"]],"reference selected IDs")
        require(self.desc(ref["source"])==c["trainpath"],"reference role")
        require(self.desc(ref["industrial_artifact"])==c["checkpoint"],"reference model")
        require(self.desc(ref["industrial_code"])==c["modelpath"],"reference code")
        close(ref["statistics"],c["reference"],"reference statistics")
        require(ref["feature_order"]==c["order"],"reference feature order")
        values=[vector(row) for row in group[:end]]
        intervention=sup["intervention"];stable=None
        if intervention is None:
            preds,releases=self.natural(c,a,group,audit)
            arrived_releases=[rel for rel in releases if rel["released_at_order"]<=cutoff]
            released={v["id"]:v for v in arrived_releases};remaining=LABEL_BUDGET-len(released)
        else:
            require(r["task"]=="decide" and end%(4*b)==0,"controlled cutoff")
            require(intervention["transform"]=="replay_prior_anchor_then_optional_location_shift","controlled transform")
            require(intervention["label_variant"]=="zero_feedback" and intervention["seed"]==42,"controlled protocol")
            ast=end-4*b;cst=end-2*b;anchor=values[ast:cst];stable=population(anchor)
            require(intervention["anchor_group_steps"]==[ast+1,cst] and intervention["applied_group_steps"]==[cst+1,end],"controlled ranges")
            token=rank("intervention.v1",42,s,a["group_key"],end)
            possible=[i for i,sd in enumerate(c["scale"]["feature_std"]) if sd>1e-6*max(1,abs(c["scale"]["feature_mean"][i]))]
            fi=possible[int(token[:8],16)%len(possible)];mult=3 if int(token[8:10],16)%2 else -3
            variant=intervention["variant"];require(variant in ("control","shift"),"controlled variant")
            require(intervention["feature"]==(c["order"][fi] if variant=="shift" else None),"controlled feature")
            require(intervention["std_multiplier"]==(mult if variant=="shift" else 0),"controlled multiplier")
            values[cst:end]=[list(v) for v in anchor]
            if variant=="shift":
                for v in values[cst:end]:v[fi]+=mult*c["scale"]["feature_std"][fi]
            expected_mapping=[{"id":group[i]["id"],"origin_id":group[i if i<cst else ast+i-cst]["id"],"group_step":i+1,"values":values[i]} for i in range(end)]
            require(audit["input_snapshot"]==intervention["transformed_inputs"],"controlled snapshot identity")
            close(self.readrows(audit["input_snapshot"]),expected_mapping,"executed transform")
            predrows=self.readrows(audit["industrial_predictions"])
            expected=c["model"].predict(np.asarray(values,dtype=np.float64)).tolist()
            close(predrows,[{"id":group[i]["id"],"group_step":i+1,"predicted_at_order":cutoff,"prediction":p} for i,p in enumerate(expected)],"controlled predictions")
            preds=[p["prediction"] for p in predrows];released={};remaining=0
            require(sup["reference_as_of"]==group[cst-1]["replay_order"],"controlled reference time")
            self.check_counts["controlled_transforms_reexecuted"]+=1
        expected_input=evidence(c,a,group,values,preds,released,end,remaining,r["task"],stable)
        close(r["input"],expected_input,"model-visible whitelist and recomputed evidence")
        starts=list(range(0,end,b));short_start=starts[max(0,len(starts)-2)]
        visible_labels=[released[row["id"]] for row in group[short_start:end] if row["id"] in released]
        expected_labels=[{k:v for k,v in rel.items() if k not in ("answer","event_sequence")} for rel in visible_labels]
        close(audit["released_labels"],expected_labels,"visible labels")
        for rel in expected_labels:
            require(rel["predicted_at_order"]<=rel["released_at_order"]<=cutoff,"label time leak")
        expected_sources=[
          {**record(c["trainpath"]),"role":"industrial_train","sample_ids":[v["id"] for v in c["ref_rows"]]},
          {**record(c["obspath"]),"role":"model0_source","sample_ids":[v["id"] for v in group[:end]+future]}]
        if visible_labels:expected_sources.append({**record(c["labelpath"]),"role":"model0_source","sample_ids":[v["id"] for v in visible_labels]})
        require(audit["sources"]==expected_sources,"source role/path/ID mismatch")
        for src in audit["sources"]:
            self.desc(src)
            require(src["role"] in ("industrial_train","model0_source"),"forbidden source")
            if src["role"]=="model0_source":
                for sid in src["sample_ids"]:
                    require(self.source_splits.setdefault((s,sid),r["split"])==r["split"],"source sample crossing")
        require(audit["action"] is None and audit["outcome"] is None,"unsupported plan supervision")
        protocol="controlled" if intervention else "forecast" if r["task"]=="forecast_next" else "coverage"
        require(sup["protocol_id"]==protocols()[protocol]["id"],"supervision protocol")
        require(self.desc({"path":sup["protocol_path"],"sha256":sup["protocol_sha256"]})==self.output/"protocols"/(protocol+".json"),"supervision protocol descriptor")
        review_artifact=self.readjson(sup["verification_artifact"])
        require(review_artifact["episode_id"]==r["episode_id"],"review identity")
        require(all(review_artifact["checks"].values()),"failed review proof")
        review=sup["review"]
        if eligible:
            require(review["status"]=="approved" and review["reviewer"]==REVIEWER and bool(review["reviewed_at"]) and bool(review["reason"]),"review approval")
            if r["task"]=="forecast_next":
                require(sup["source_type"]=="observed_future_features","forecast supervision")
                require(len(future)==32 and (s!="A" or end>=32),"forecast coverage")
                close(target,{"horizon":32,"feature_order":c["order"],"values":[vector(row) for row in future]},"true future target")
                require(target["values"]==[vector(row) for row in future],"source numeric precision")
            else:
                require(set(target)=={"decision","evidence_refs","hypotheses"},"decide target fields")
                require(set(target["evidence_refs"])<=set(r["input"]["evidence_refs"]),"target references")
                require(target["hypotheses"]==[],"unsupported causal hypotheses")
                if intervention:
                    require(sup["source_type"]=="controlled_intervention","controlled supervision")
                    require(target["decision"]==("no_drift" if intervention["variant"]=="control" else "drift"),"controlled target")
                    require(target["evidence_refs"]==["recent-reference-change","quality"],"controlled references")
                else:
                    require(end<2*b and target=={"decision":"insufficient","evidence_refs":["window-coverage"],"hypotheses":[]},"coverage target")
                    require(sup["source_type"]=="independent_annotation","coverage supervision")
        else:
            require(r["task"]=="decide" and intervention is None and end>=2*b,"quarantine scope")
            require(target is None and review["status"]=="pending" and sup["source_type"] is None,"quarantine not training")
        self.check_counts["episodes_recomputed"]+=1

def validate(output):
    output=Path(output).resolve();failures=[];v=None;records=[]
    try:
        v=Validator(output);v.initial()
        for role in ("train","dev","quarantine"):
            for row in rows(output/(role+".jsonl")):
                records.append(row)
                try:v.episode(row,role)
                except Exception as exc:
                    failures.append({"episode_id":row.get("episode_id"),"file":role,"reason":str(exc)})
                    if len(failures)>=20:raise ValueError("Stopped after20 failed records")
        require(counts(records)==v.m["counts"],"manifest counts mismatch")
    except Exception as exc:failures.append({"scope":"dataset","reason":str(exc)})
    totals=collections.Counter()
    for r in records:totals["eligible" if r["audit"]["eligibility"]["sft_eligible"] else "quarantine"]+=1
    return {"schema_version":"model0.validation.v1","status":"failed" if failures else "passed",
      "validator":REVIEWER,"read_only":True,"checked_records":len(records),
      "passed_records":v.check_counts["episodes_recomputed"] if v else 0,
      "failed_checks":len(failures),"failures":failures,"checks":dict(v.check_counts) if v else {},
      "eligible_records":totals["eligible"],"quarantined_records":totals["quarantine"],
      "quarantine_reason_counts":{"natural_decision_unreviewed":totals["quarantine"]},
      "counts":counts(records),"limitations":v.m["limitations"] if v else [],
      "scope":"Hashes, frozen protocols, source roles/IDs, split isolation, cutoff visibility, saved industrial predictions, actual release logs, input statistics, executed controlled transforms, true future targets and deterministic review. No claim of natural causal truth."}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset",default="model0_episodes_v1_1");args=p.parse_args()
    result=validate(ROOT/args.dataset);print(json.dumps(result,ensure_ascii=False,indent=2))
    sys.exit(0 if result["status"]=="passed" else 1)
