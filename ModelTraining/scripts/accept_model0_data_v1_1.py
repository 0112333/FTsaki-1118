"""Independent read-only dataset audit; writes reports only in ModelTraining."""
from pathlib import Path
import json, hashlib, math, statistics, collections, datetime, sys
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
DS = DATA / "model0_episodes_v1_1"
OUT = ROOT / "artifacts/acceptance/model0_episodes_v1_1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT.parent/"FTsaki_harness/src"))
from ftsaki_harness.data import flatten
from ftsaki_harness.forecast import Scale, evaluate
failures=[]; checks=collections.Counter(); cache={}
def check(ok, name, detail=""):
    checks[name]+=1
    if not ok:
        failures.append({"check":name,"detail":detail})
def parse(text):
    def unique(items):
        d={}
        for k,v in items:
            if k in d: raise ValueError("duplicate key "+k)
            d[k]=v
        return d
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def read(p):
    p=Path(p)
    if p not in cache:cache[p]=parse(p.read_text())
    return cache[p]
def rows(p):
    with Path(p).open() as f:
        for i,line in enumerate(f,1):
            if not line.strip():raise ValueError(f"blank row {p}:{i}")
            yield i,parse(line)
hash_cache={}
def sha(p):
    p=Path(p)
    if p not in hash_cache:
        h=hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
        hash_cache[p]=h.hexdigest()
    return hash_cache[p]
def desc(d,base=DATA):
    p=(base/d["path"]).resolve()
    check(p.is_relative_to(DATA),"path_scope",str(p))
    check(sha(p)==d["sha256"].removeprefix("sha256:"),"descriptor_hash",str(p))
    return p
def near(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def equal(a,b):
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool): return isinstance(b,(int,float)) and not isinstance(b,bool) and near(a,b)
    return type(a)==type(b) and a==b
def stats(v):
    cols=list(zip(*v))
    return ([statistics.fmean(c) for c in cols],[statistics.pstdev(c) for c in cols])
manifest=read(DS/"manifest.json")
file_records=manifest["files"]
for d in file_records:
    p=desc(d,DS)
    check(p.stat().st_size==d["bytes"],"manifest_bytes",d["path"])
    if d["row_count"] is not None:
        with p.open() as f:n=sum(1 for _ in f)
        check(n==d["row_count"],"manifest_row_count",d["path"])
for d in manifest["tooling"]:desc(d)
for d in manifest["protocols"]:desc(d,DS)
upstream=read(desc(manifest["upstream_manifest"]))
sources={};scales={};scale_comparison={};ref_maps={}
for s in "ABC":
    entry=upstream["scenarios"][s]
    trainp=desc(entry["splits"]["industrial_train"]["dataset"],DATA/"harness_data_v1")
    obsp=desc(entry["splits"]["model0_source"]["observations"],DATA/"harness_data_v1")
    labp=desc(entry["splits"]["model0_source"]["labels"],DATA/"harness_data_v1")
    train={r["id"]:r for _,r in rows(trainp)}
    obs={r["id"]:r for _,r in rows(obsp)}
    labels={r["id"]:r["answer"] for _,r in rows(labp)}
    order,_=flatten(next(iter(obs.values()))["inputs"])
    vec={k:flatten(r["inputs"])[1] for k,r in obs.items()}
    tv=[flatten(r["inputs"])[1] for r in train.values()]
    full=Scale.fit(tv);scales[s]=full
    ref=read(DS/"snapshots"/s/"reference.json");ref_maps[s]=ref
    rv=[flatten(train[k]["inputs"])[1] for k in ref["sample_ids"]]
    mu,sd=stats(rv)
    check(all(near(a,b) for a,b in zip(mu,ref["statistics"]["feature_mean"])),"reference_means",s)
    check(all(near(a,b) for a,b in zip(sd,ref["statistics"]["feature_std"])),"reference_stds",s)
    delivered=read(DS/"snapshots"/s/"feature_scale.json")
    check(delivered["role"]=="industrial_train" and delivered["row_count"]==len(train),"full_scale_source_count",s)
    check(desc(delivered["source"])==trainp,"full_scale_source",s)
    check(delivered["feature_order"]==order,"full_scale_order",s)
    check(equal(delivered["statistics"],{"sample_count":len(train),"feature_mean":list(full.mean),"feature_std":list(full.std)}),"full_scale_independent_recompute",s)
    rsd=delivered["statistics"]["feature_std"]
    ratios=[a/b for a,b in zip(rsd,full.std) if b>0]
    scale_comparison[s]={"full_train_count":len(train),"delivered_scale_count":delivered["row_count"],
      "same_population":len(train)==delivered["row_count"],"max_mean_difference":max(abs(a-b) for a,b in zip(delivered["statistics"]["feature_mean"],full.mean)),
      "std_ratio_min":min(ratios),"std_ratio_max":max(ratios),
      "sample_mean":mu,"sample_std":sd,"full_mean":list(full.mean),"full_std":list(full.std)}
    sources[s]={"train":train,"obs":obs,"labels":labels,"vec":vec,"order":order,
       "allowed":{trainp:"industrial_train",obsp:"model0_source",labp:"model0_source"}}
assign=read(DS/"protocols/split_assignment.json")["groups"]
assignment={x["group_key"]:x for x in assign}
seen_source={};ids=set();fam_split={};counts=collections.Counter();families=collections.defaultdict(set)
character_lengths=collections.defaultdict(list);controlled_bad=[];gate_changes=collections.Counter()
gate_examples=[];rows_by_file=collections.Counter();snapshot_rows={}
for role in ["train","dev","quarantine"]:
    for line,e in rows(DS/(role+".jsonl")):
        eid=e["episode_id"];s=e["scenario_id"];i=e["input"];a=e["audit"];t=e["target"];src=sources[s]
        detail=f"{role}.jsonl:{line}:{eid}"
        check(eid not in ids,"unique_episode",detail);ids.add(eid)
        check(e["record_kind"]=="episode","not_example",detail)
        check(e["schema_version"]=="model0.episode.v1" and i["schema_version"]=="model0.evidence.v1","versions",detail)
        eligible=a["eligibility"]["sft_eligible"]
        check((eligible and e["split"]==role and t is not None) or
              (role=="quarantine" and not eligible and t is None),"file_eligibility",detail)
        f=a["split_assignment"]["source_family_id"]
        check(fam_split.setdefault(f,e["split"])==e["split"],"family_isolation",detail)
        ass=assignment[a["split_assignment"]["group_key"]];group=[src["obs"][sid] for sid in ass["sample_ids"]]
        check(ass["split"]==e["split"] and ass["source_family_id"]==f,"assignment_match",detail)
        cut=i["as_of"]["group_step"];cutorder=i["as_of"]["replay_order"];b={"A":16,"B":8,"C":32}[s]
        check(0<cut<=len(group) and (cut%b==0 or cut==len(group)),"batch_cutoff",detail)
        check(group[cut-1]["replay_order"]==cutorder,"cutoff_source",detail)
        check(i["scenario_id"]==s and i["feature_order"]==src["order"],"feature_schema",detail)
        for source in a["sources"]:
            p=desc(source)
            check(src["allowed"].get(p)==source["role"],"source_allowlist",detail)
            allowedids=src["train"] if source["role"]=="industrial_train" else src["obs"]
            check(set(source["sample_ids"])<=set(allowedids),"source_ids_exist",detail)
            if source["role"]=="model0_source":
                for sid in source["sample_ids"]:
                    check(seen_source.setdefault((s,sid),e["split"])==e["split"],"source_split_isolation",detail)
        actual_visible=a["visible_observations"]
        check([x["id"] for x in actual_visible]==[r["id"] for r in group[:cut]],"visible_ids",detail)
        check(all(x["replay_order"]<=cutorder for x in actual_visible),"visible_cutoff",detail)
        refids=set(i["evidence_refs"]);items=i["evidence_items"]
        check(len(refids)==len(i["evidence_refs"]) and refids=={x["id"] for x in items},"evidence_ids",detail)
        check(all(x["predicted_at_order"]<=x["released_at_order"]<=cutorder for x in a["released_labels"]),"label_cutoff",detail)
        check(a["action"] is None and a["outcome"] is None,"no_fabricated_action",detail)
        sup=a["supervision"];inv=sup["intervention"]
        if eligible:
            check(sup["review"]["status"]=="approved" and bool(sup["review"]["reviewer"]),"supervision_review",detail)
            desc(sup["verification_artifact"])
        original=[src["vec"][r["id"]] for r in group[:cut]]
        values=[list(v) for v in original]
        if inv:
            ast=cut-4*b;cst=cut-2*b
            anchor=original[ast:cst];values[cst:cut]=[list(v) for v in anchor]
            check(cst>ast>=0,"anchor_before_current",detail)
            j=None
            if inv["variant"]=="shift":
                j=i["feature_order"].index(inv["feature"])
                delta=inv["std_multiplier"]*i["feature_scale"]["std"][j]
                for v in values[cst:cut]:v[j]+=delta
                tau=1e-6*max(1,abs(i["feature_scale"]["mean"][j]))
                if abs(delta)<=tau:controlled_bad.append({"episode":eid,"delta":delta,"tau":tau})
            p=desc(inv["transformed_inputs"])
            if p not in snapshot_rows:snapshot_rows[p]=[r for _,r in rows(p)]
            snap=snapshot_rows[p]
            check([x["values"] for x in snap]==values,"controlled_transform_exact",detail)
            check([x["origin_id"] for x in snap]==[group[q if q<cst else ast+q-cst]["id"] for q in range(cut)],"controlled_origin_ids",detail)
            mu,sd=stats(anchor)
            check(all(near(x,y) for x,y in zip(mu,i["recent_stable_reference"]["feature_mean"])),"anchor_reference",detail)
            check(t["decision"]==("drift" if inv["variant"]=="shift" else "no_drift"),"controlled_target",detail)
            check(i["label_feedback"]["visible_count"]==0 and i["label_feedback"]["metrics"] is None,"controlled_no_labels",detail)
        starts=list(range(0,cut,b));shortstart=starts[max(0,len(starts)-2)]
        current=values[shortstart:cut];mu,sd=stats(current)
        check(all(near(x,y) for x,y in zip(mu,i["current"]["feature_mean"])),"current_mean_recompute",detail)
        check(all(near(x,y) for x,y in zip(sd,i["current"]["feature_std"])),"current_std_recompute",detail)
        ref=ref_maps[s]["statistics"]
        check(equal(i["feature_scale"],{"fit_role":"industrial_train","count":len(src["train"]),"mean":list(scales[s].mean),"std":list(scales[s].std)}),"full_scale_episode_recompute",detail)
        artifact=read(desc(a["feature_scale_artifact"]))
        check(equal(artifact,read(DS/"snapshots"/s/"feature_scale.json")),"full_scale_artifact_consistency",detail)
        check(equal(i["training_reference"],ref),"training_reference_consistency",detail)
        longstart=starts[max(0,len(starts)-8)]
        expected_windows={"batch_size":b,"short":{"requested_batches":2,"available_batches":min(2,len(starts)),"sample_count":len(current)},"long":{"requested_batches":8,"available_batches":min(8,len(starts)),"sample_count":cut-longstart}}
        check(equal(i["windows"],expected_windows),"window_coverage_recompute",detail)
        indexed={x["id"]:x for x in items}
        delta=[(v-m)/d if d else None for v,m,d in zip(mu,ref["feature_mean"],i["feature_scale"]["std"])]
        check(equal(indexed["feature-location"]["standardized_mean_delta"],delta),"feature_location_recompute",detail)
        check(equal(indexed["quality"]["finite_counts"],[len(current)]*len(mu)),"quality_counts",detail)
        check(indexed["window-coverage"]["required_short_samples"]==2*b and indexed["window-coverage"]["available_short_samples"]==len(current),"coverage_item_recompute",detail)
        if inv:
            anchor_mu,anchor_sd=stats(anchor)
            check(equal(i["recent_stable_reference"],{"sample_count":len(anchor),"feature_mean":anchor_mu,"feature_std":anchor_sd}),"stable_reference_recompute",detail)
            delta=[(v-m)/d if d else None for v,m,d in zip(mu,anchor_mu,i["feature_scale"]["std"])]
            check(equal(indexed["recent-reference-change"]["standardized_mean_delta"],delta),"recent_reference_delta",detail)
        if len(starts)>=8:
            long_mu,long_sd=stats(values[longstart:cut]);longitem=indexed["long-window-summary"]
            expected={"group_step_start":longstart+1,"group_step_end":cut,"sample_count":cut-longstart,"finite_counts":[cut-longstart]*len(mu),"feature_mean":long_mu,"feature_std":long_sd}
            check(equal({k:longitem[k] for k in expected},expected),"long_window_recompute",detail)
        else:check("long-window-summary" not in indexed,"long_window_unavailable",detail)
        if e["task"]=="forecast_next":
            future=group[cut:cut+32]
            target=[src["vec"][r["id"]] for r in future]
            check(len(future)==32 and t["horizon"]==32 and t["feature_order"]==src["order"],"forecast_shape",detail)
            check(t["values"]==target,"future_values_exact",detail)
            check([x["id"] for x in a["future_targets"]]==[r["id"] for r in future],"future_ids_contiguous",detail)
            check([x["relative_step"] for x in a["future_targets"]]==list(range(1,33)),"future_positions",detail)
            history=values[max(0,cut-32):cut]
            check(i["recent_history"]["values"]==history,"history_exact",detail)
            check(i["recent_history"]["group_steps"]==list(range(max(0,cut-32)+1,cut+1)),"history_positions",detail)
            baseline=history[-32:] if s=="A" else [history[-1]]*32
            es=Scale(tuple(i["feature_scale"]["mean"]),tuple(i["feature_scale"]["std"]))
            full_result=evaluate(s,baseline,target,history,scales[s])
            subset_result=evaluate(s,baseline,target,history,es)
            if full_result["outcome"]!=subset_result["outcome"]:
                gate_changes[(role,s)]+=1
                if len(gate_examples)<8:gate_examples.append({"episode":eid,"file":role,"line":line,"full":full_result,"delivered":subset_result})
        elif t:
            check(t["decision"] in {"drift","no_drift","insufficient"},"decision_enum",detail)
            check(set(t["evidence_refs"])<=refids,"target_refs",detail)
            check(t["decision"]=="insufficient" or bool(t["evidence_refs"]),"binary_refs",detail)
            if not inv:check(t["decision"]=="insufficient" and cut<2*b,"coverage_supervision",detail)
        label="quarantine" if t is None else t.get("decision","forecast_next")
        counts[(role,s,label)]+=1;families[role].add(f);rows_by_file[role]+=1
        if eligible:character_lengths[(role,s,e["task"])].append(len(json.dumps(i,ensure_ascii=False,separators=(",",":")))+len(json.dumps(t,ensure_ascii=False,separators=(",",":"))))
old_counts=collections.Counter()
for role in ["train","dev","quarantine"]:
    old={e["episode_id"]:e for _,e in rows(DATA/"model0_episodes_v1"/(role+".jsonl"))}
    new={e["episode_id"]:e for _,e in rows(DS/(role+".jsonl"))}
    check(set(old)==set(new),"revision_episode_identity",role)
    for eid,e in new.items():
        new_assignment=e["audit"]["split_assignment"]; old_assignment=old[eid]["audit"]["split_assignment"]
        check({k:v for k,v in new_assignment.items() if k!="assignment_artifact"}=={k:v for k,v in old_assignment.items() if k!="assignment_artifact"} and new_assignment["assignment_artifact"]["sha256"]==old_assignment["assignment_artifact"]["sha256"],"revision_group_split_unchanged",eid)
        desc(new_assignment["assignment_artifact"])
        if e["task"]=="forecast_next":
            check(e["target"]==old[eid]["target"] and e["input"]["recent_history"]==old[eid]["input"]["recent_history"],"revision_natural_target_history_exact",eid)
manifest_before=sha(DS/"manifest.json")
report={"schema_version":"model0.consumer_acceptance.v1","checked_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "dataset_manifest_sha256":manifest_before,"file_count":len(file_records),"file_bytes":sum(x["bytes"] for x in file_records),
 "independent_structural_status":"passed" if not failures else "failed","failures":failures[:50],"failure_count":len(failures),
 "checks":dict(checks),"record_counts":dict(rows_by_file),"family_counts":{k:len(v) for k,v in families.items()},
 "breakdown":[{"file":k[0],"scenario":k[1],"target":k[2],"count":v} for k,v in sorted(counts.items())],
 "scale_comparison":scale_comparison,"gate_differences_using_last_value_or_empirical_baseline":[{"file":k[0],"scenario":k[1],"count":v} for k,v in sorted(gate_changes.items())],
 "gate_difference_examples":gate_examples,"controlled_nearconstant_shift_count":len(controlled_bad),
 "serialized_character_lengths_not_tokens":[{"file":k[0],"scenario":k[1],"task":k[2],"min":min(v),"median":statistics.median(v),"max":max(v)} for k,v in sorted(character_lengths.items())],
 "limitations":["This checker independently verified raw data, features, hashes, timestamps, grouping, transforms and statistics; industrial predictions and numerical label errors need separate model re-execution.",
 "Character counts are not tokenizer counts; actual model tokenizer/context limits remain unverified."]}
(OUT/"independent-checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ["scale_comparison","checks","serialized_character_lengths_not_tokens"]},ensure_ascii=False,indent=2))
print("SCALE",json.dumps({k:{x:y for x,y in v.items() if not isinstance(y,list)} for k,v in scale_comparison.items()},ensure_ascii=False))
