"""Build model0 episodes in a new version directory, preserving upstream data."""
import argparse
from datetime import datetime, timezone
import subprocess
from model0_data import *

LIMITATIONS = [
 "A只有一个采集域；64/26连续块隔离不等于独立采集域验证。文件顺序不是物理时钟。",
 "A开发集不足32历史+32未来，forecast_next未覆盖；A开发集drift/no_drift也未覆盖。",
 "B开发集仅Bearing2_2（37.5Hz11kN），训练为Bearing1_2（35Hz12kN）与Bearing3_2（40Hz10kN）。只有3个独立设备。",
 "重叠窗口高度相关，必须同时报告条数和来源家族数。",
 "自然drift/no_drift没有独立监督，全部隔离；自然insufficient仅表达冻结的最小窗口覆盖规则。",
 "受控案例只证明相对已知历史锚点的经验输入分布变化，不证明自然因果、设备故障或真实闭环收益。",
 "证据参考采用固定工业训练参考库，尺度独立使用完整industrial/train；参考库为：A全部267条，B每轴承最多128条，C每FD128条。未用模型效果挑样。",
 "受控案例明确采用零反馈协议；自然回放采用默认25%标签释放。没有工业续训授权。",
 "本交付不包含最优更新计划监督、model0训练、最终盲测或运行期集成。",
 "保留完整原单位预测矩阵；训练侧需验收token长度与多任务权重，不得截断补数。"
]

def label_worker(s):
    m=read_json(ROOT/"harness_data_v1/manifest.json")
    p=check_descriptor(ROOT/"harness_data_v1",m["scenarios"][s]["splits"]["model0_source"]["labels"])
    answers={r["id"]:r["answer"] for r in rows(p)}
    for line in sys.stdin:
        req=parse(line);ids=req["ids"]
        if len(set(ids))!=len(ids) or not set(ids)<=set(answers):raise ValueError("invalid label request")
        selected=sorted(ids,key=lambda sid:rank(42,s,req["source_group"],sid))[:min(math.ceil(len(ids)*.25),req["remaining"])]
        print(canonical({sid:answers[sid] for sid in selected}),flush=True)

def make_episode(c,a,group,end,task,variant,inp,target,upstream,proto,split_record,created_at,directory,released,intervention):
    s=c["s"];eligible=target is not None
    identity=f"m0v1-{s}-{a['group_id']}-{end:06d}-{task}-{variant}"
    future=group[end:end+32] if task=="forecast_next" else []
    visible=group[:end];b=BATCH[s];starts=list(range(0,end,b));begin=starts[max(0,len(starts)-2)]
    visible_labels=[released[r["id"]] for r in group[begin:end] if r["id"] in released]
    sources=[{**record(c["trainpath"]),"role":"industrial_train","sample_ids":[r["id"] for r in c["ref_rows"]]},
             {**record(c["obspath"]),"role":"model0_source","sample_ids":[r["id"] for r in visible+future]}]
    if visible_labels:sources.append({**record(c["labelpath"]),"role":"model0_source","sample_ids":[r["id"] for r in visible_labels]})
    protocol="controlled" if intervention else "forecast" if task=="forecast_next" else "coverage"
    stype="controlled_intervention" if intervention else "observed_future_features" if task=="forecast_next" else "independent_annotation" if eligible else None
    checks={"visible_before_cutoff":all(r["replay_order"]<=group[end-1]["replay_order"] for r in visible)}
    if task=="forecast_next":
        checks.update(next32=len(future)==32,exact_target=target["values"]==[vector(r) for r in future],
                      history_coverage=s!="A" or end>=32)
    elif eligible and not intervention:checks["minimum_coverage_rule"]=end<2*b
    if not all(checks.values()):raise ValueError("construction proof failed")
    proofpath=directory/f"{end:06d}-{task}-{variant}-review.json"
    write_json(proofpath,{"episode_id":identity,"reviewer":REVIEWER,"status":"approved" if eligible else "pending",
                          "checks":checks,"scope":"deterministic evidence only; no human causal review"})
    return {"schema_version":"model0.episode.v1","record_kind":"episode","episode_id":identity,
            "task":task,"scenario_id":s,"split":a["split"],"input":inp,"target":target,
            "audit":{"source_manifest":upstream,"sources":sources,
              "visible_observations":[position(s,r,i+1) for i,r in enumerate(visible)],
              "released_labels":[{k:v for k,v in r.items() if k not in ("answer","event_sequence")} for r in visible_labels],
              "industrial_artifact":record(c["checkpoint"]),"industrial_predictions":None,
              "reference_snapshot":record(directory.parent/"reference.json"),
              "feature_scale_artifact":record(directory.parent/"feature_scale.json"),
              "split_assignment":{"protocol_id":"model0-split.v1","group_key":a["group_key"],
                "event_key":a["source_family_id"]+"-"+str(end),"source_family_id":a["source_family_id"],
                "assigned_before_windowing":True,"assignment_artifact":split_record},
              "supervision":{"source_type":stype,"protocol_id":protocols()[protocol]["id"],
                "protocol_path":proto[protocol]["path"],"protocol_sha256":proto[protocol]["sha256"],
                "target_scope":"simulated_current_vs_prior_empirical_distribution" if intervention else "next_32_input_features" if task=="forecast_next" else "minimum_visible_window_coverage" if eligible else "unreviewed_natural_distribution",
                "reference_as_of":group[end-2*b-1]["replay_order"] if intervention else group[end-1]["replay_order"],
                "intervention":intervention,"verification_artifact":record(proofpath),
                "review":{"status":"approved" if eligible else "pending","reviewer":REVIEWER if eligible else None,
                  "reviewed_at":created_at if eligible else None,
                  "reason":"Versioned deterministic target checks; no human causal review." if eligible else "No independently verified natural drift/stability target."}},
              "future_targets":[{**position(s,r,end+i+1),"relative_step":i+1} for i,r in enumerate(future)],
              "action":None,"outcome":None,
              "eligibility":{"sft_eligible":eligible,"reason_codes":[] if eligible else ["natural_decision_unreviewed"]}}}

def inventory(output,manifest):
    manifest["files"]=[]
    for p in sorted(output.rglob("*")):
        if not p.is_file() or p.name=="manifest.json":continue
        rel=p.relative_to(output).as_posix()
        manifest["files"].append({"role":p.stem if p.parent==output else rel.split("/")[0],
          "path":rel,"sha256":digest(p),"bytes":p.stat().st_size,
          "row_count":sum(1 for _ in p.open(encoding="utf-8")) if p.suffix==".jsonl" else None})

def write_readme(output,records,created_at):
    text="""# model0 案例数据交付 v1

本目录是从冻结的 harness_data_v1 构建的真实来源案例及受控模拟案例。只有审核通过的记录进入 train/dev；缺少可信自然判断目标的记录进入 quarantine。

## 接收入口
入口是 manifest.json，格式为 model0.dataset.v1 / model0.episode.v1 / model0.evidence.v1。
ModelTraining 仅把 input 序列化为 user、target 序列化为 assistant；audit 绝不能进入提示。
两种任务分别是 decide 和 forecast_next，后者预测未来32条完整输入特征，不是类别或RUL。

## 分组与时间
只使用 industrial/train 公共历史和 model0_source 素材；未读取 closed_loop 或 industrial/test 内容构建或选择案例。
A前64条训练、后26条开发，窗口在块边界重置；B固定Bearing2_2开发，其余两根训练；C每FD哈希选择约20%整台设备开发。
同一家族的两任务、重叠窗口及模拟版本始终同侧。分配在窗口化之前冻结于 protocols/split_assignment.json。
A的group_step在新块从1计数，replay_order保留上游数值，source_step仍是原source_index；三者不可混用。
按A/B/C=16/8/32条一批，短/长窗口最多2/8批，不跨组补齐尾批。

## 证据与标签
固定参考库取A全部工业训练样本、B每根工业训练轴承最多128条、C每FD128条，由seed42哈希选择并登记全部ID。
feature_scale来自完整industrial/train（A267/B3152/C67798），training_reference保留抽样库；均采用总体标准差ddof=0。
参考库采样只影响training_reference，不决定feature_scale。完整尺度由共享工件追溯，不使用检查点防除零scale。
工业预测使用原检查点自身标准化与权重，不进行重训。特征顺序为字典键排序、数组数值下标顺序，C先sensors后settings。
自然回放每批先保存并flush预测，再向独立标签进程申请ceil(25%×n)标签，最后保存释放日志后构建输入。
抽样复用harness的SHA256排序语义，每设备/块固定4096标签额度；额度不由设备未来总寿命推算。
反馈只统计当前短窗口内已释放标签；输入不知道未来标签。train_authorized=false。

## 监督和审核
forecast_next使用同组后续32条真实原始特征，不能补齐、跨组或量化。输入历史最多32条，A至少32条。
自然insufficient由独立的版本化确定性覆盖规则标注，再由验收器复核，限定于最少2批覆盖要求；不声称自然稳定或因果。
其余自然decide缺少可信drift/no_drift目标，target=null并隔离；审核缺失不自动变成insufficient。
受控对照将截止点前的2批锚点按原序重放到当前2批位置，使有限经验分布相同；shift版本在同一锚点上执行确定性单特征±3完整工业训练标准差平移。
快照同时记录当前slot ID与原始origin ID，变换参数只在audit。归因hypotheses为空，避免无证据自然因果结论。
模拟明确采用零标签反馈，不把原始未变换标签配给变换输入。模拟目标只表示相对此前锚点的输入变化，不代表真实闭环收益。
审核是确定性程序审核，reviewer有版本；没有伪称人工审核或使用大模型伪标签。action/outcome为空。

## 复现与验收
依赖Python3.12、NumPy1.26.4。当前Windows Python使用data/.training-deps；
Linux应准备相同版本本机NumPy，不能加载Windows二进制依赖。
在data工作区执行（输出目录必须尚不存在）：

    python -B prepare_model0_episodes.py --output model0_episodes_reproduced --created-at CREATED_AT
    python -B validate_model0_episodes.py --dataset model0_episodes_reproduced
    python -B -m unittest discover -s tests -p test_model0_episodes.py

生成器先冻结协议/分组，再生成案例并运行验收；validator默认只读，把结果打印到stdout。
validation_report.json列出实际通过数、失败数、拒绝原因和覆盖限制。
复现的新目录名会改变path、dataset_id及文件哈希；应比较episode_id/input/target及分配等语义。
episode路径相对data根；manifest.files/protocols路径相对本manifest。跨机器保留这些相对布局与原包运行代码、获准上游数据和检查点。
SHA-256为无前缀64位小写十六进制。

## 统计
quarantine记录保留预分配split，但不参与训练或开发评分。不同目标行可能共享来源家族，不能把家族数跨行相加。

| split | 场景 | 任务 | 目标 | 条数 | 家族数 |
| --- | --- | --- | --- | ---: | ---: |
""".replace("CREATED_AT",created_at)
    for c in counts(records):
        text+=f"| {c['split']} | {c['scenario_id']} | {c['task']} | {c['target']} | {c['episodes']} | {c['source_families']} |\n"
    text+="\n## 局限\n\n"+"\n".join("- "+x for x in LIMITATIONS)+"\n"
    (output/"README.md").write_text(text,encoding="utf-8")

def build(output,created_at):
    if output.exists() or output.parent!=ROOT:raise ValueError("output must be a new direct child of data workspace")
    if np.__version__!="1.26.4":raise ValueError("Use NumPy1.26.4")
    ctx=contexts();assignment=assignments(ctx);output.mkdir()
    for k,v in protocols().items():write_json(output/"protocols"/(k+".json"),v)
    write_json(output/"protocols/split_assignment.json",{"protocol_id":"model0-split.v1","assigned_before_windowing":True,"groups":assignment})
    split_record=record(output/"protocols/split_assignment.json")
    tooling=[record(ROOT/n) for n in ("model0_data.py","prepare_model0_episodes.py","validate_model0_episodes.py")]
    upstream=record(ROOT/"harness_data_v1/manifest.json")
    write_json(output/"snapshots/freeze.json",{"created_at":created_at,"generator_version":VERSION,
      "upstream_manifest":upstream,"split_assignment":split_record,"tooling":tooling,
      "protocols":[record(p) for p in sorted((output/"protocols").glob("*.json"))],
      "environment":{"python":sys.version,"numpy":np.__version__}})
    proto={k:record(output/"protocols"/(k+".json")) for k in protocols()}
    all_records=[];skips=collections.Counter()
    for s,c in ctx.items():
        write_json(output/"snapshots"/s/"feature_scale.json", {"source":record(c["trainpath"]),"role":"industrial_train","feature_order":c["order"],"row_count":c["scale"]["sample_count"],"statistics":c["scale"]})
        write_json(output/"snapshots"/s/"reference.json",{"source":record(c["trainpath"]),
          "sample_ids":[r["id"] for r in c["ref_rows"]],"feature_order":c["order"],"statistics":c["reference"],
          "industrial_artifact":record(c["checkpoint"]),"industrial_code":record(c["modelpath"])})
        index={r["id"]:r for r in c["observations"]}
        proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--label-worker",s],
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8",bufsize=1)
        try:
            for a in [v for v in assignment if v["scenario_id"]==s]:
                group=[index[sid] for sid in a["sample_ids"]];values=[vector(r) for r in group];b=BATCH[s]
                directory=output/"snapshots"/s/a["group_id"];directory.mkdir(parents=True)
                predpath=directory/"natural_predictions.jsonl";releasepath=directory/"label_releases.jsonl";ledgerpath=directory/"replay_ledger.jsonl"
                predictions=[];released={};remaining=LABEL_BUDGET;seq=0;pending=[]
                with predpath.open("x",encoding="utf-8",newline="\n") as pf, releasepath.open("x",encoding="utf-8",newline="\n") as lf, ledgerpath.open("x",encoding="utf-8",newline="\n") as log:
                    for start in range(0,len(group),b):
                        end=min(start+b,len(group));cutoff=group[end-1]["replay_order"]
                        pred=c["model"].predict(np.asarray(values[start:end],dtype=np.float64)).tolist()
                        predictions.extend(pred);seq+=1
                        for i,p in enumerate(pred,start):
                            pf.write(canonical({"id":group[i]["id"],"group_step":i+1,"predicted_at_order":cutoff,"event_sequence":seq,"prediction":p})+"\n")
                        pf.flush()
                        log.write(canonical({"event_sequence":seq,"event":"predictions_saved","as_of_order":cutoff,"sample_ids":[r["id"] for r in group[start:end]]})+"\n");log.flush()
                        req={"ids":[r["id"] for r in group[start:end]],"source_group":a["source_group"],"remaining":remaining}
                        proc.stdin.write(canonical(req)+"\n");proc.stdin.flush();reply=proc.stdout.readline()
                        if not reply:raise RuntimeError("label worker failed")
                        answers=parse(reply);remaining-=len(answers);seq+=1
                        for sid,answer in answers.items():
                            release={"id":sid,"answer":answer,"predicted_at_order":cutoff,"released_at_order":cutoff,"train_authorized":False,"event_sequence":seq}
                            released[sid]=release;lf.write(canonical(release)+"\n")
                        lf.flush()
                        log.write(canonical({"event_sequence":seq,"event":"labels_released","as_of_order":cutoff,"sample_ids":sorted(answers)})+"\n");log.flush()
                        for task in ("decide","forecast_next"):
                            if task=="forecast_next" and (len(group)-end<32 or (s=="A" and end<32)):
                                skips[(s,a["split"],"future_short" if len(group)-end<32 else "history_short")]+=1;continue
                            target=None
                            if task=="forecast_next":target={"horizon":32,"feature_order":c["order"],"values":values[end:end+32]}
                            elif end<2*b:target={"decision":"insufficient","evidence_refs":["window-coverage"],"hypotheses":[]}
                            inp=evidence(c,a,group,values,predictions,released,end,remaining,task)
                            pending.append(make_episode(c,a,group,end,task,"natural",inp,target,upstream,proto,split_record,created_at,directory,released,None))
                        if end%(4*b)==0:
                            ast=end-4*b;cst=end-2*b;anchor=values[ast:cst];stable=population(anchor)
                            candidates=[i for i,sd in enumerate(c["scale"]["feature_std"]) if sd>1e-6*max(1,abs(c["scale"]["feature_mean"][i]))]
                            token=rank("intervention.v1",42,s,a["group_key"],end)
                            feature=candidates[int(token[:8],16)%len(candidates)]
                            multiplier=3 if int(token[8:10],16)%2 else -3
                            for variant in ("control","shift"):
                                transformed=[list(v) for v in values[:end]]
                                transformed[cst:end]=[list(v) for v in anchor]
                                if variant=="shift":
                                    for v in transformed[cst:end]:v[feature]+=multiplier*c["scale"]["feature_std"][feature]
                                altered=c["model"].predict(np.asarray(transformed,dtype=np.float64)).tolist()
                                name=f"{end:06d}-{variant}";transpath=directory/(name+"-inputs.jsonl")
                                mapping=[{"id":group[i]["id"],"origin_id":group[i if i<cst else ast+i-cst]["id"],"group_step":i+1,"values":transformed[i]} for i in range(end)]
                                write_rows(transpath,mapping)
                                altpred=directory/(name+"-predictions.jsonl")
                                write_rows(altpred,({"id":group[i]["id"],"group_step":i+1,"predicted_at_order":cutoff,"prediction":p} for i,p in enumerate(altered)))
                                intervention={"transform":"replay_prior_anchor_then_optional_location_shift","variant":variant,"seed":42,
                                  "anchor_group_steps":[ast+1,cst],"applied_group_steps":[cst+1,end],
                                  "feature":c["order"][feature] if variant=="shift" else None,"std_multiplier":multiplier if variant=="shift" else 0,
                                  "label_variant":"zero_feedback","transformed_inputs":record(transpath)}
                                inp=evidence(c,a,group,transformed,altered,{},end,0,"decide",stable)
                                target={"decision":"no_drift" if variant=="control" else "drift","evidence_refs":["recent-reference-change","quality"],"hypotheses":[]}
                                ep=make_episode(c,a,group,end,"decide",variant,inp,target,upstream,proto,split_record,created_at,directory,{},intervention)
                                ep["audit"]["industrial_predictions"]=record(altpred);ep["audit"]["input_snapshot"]=record(transpath)
                                expected=np.asarray(anchor).copy()
                                if variant=="shift":expected[:,feature]+=multiplier*c["scale"]["feature_std"][feature]
                                if not np.array_equal(expected,np.asarray(transformed[cst:end])):raise ValueError("transform mismatch")
                                reviewpath=resolve(ep["audit"]["supervision"]["verification_artifact"]["path"])
                                review=read_json(reviewpath)
                                review["checks"].update(executed_transform_matches=True,anchor_precedes_current=True,zero_label_feedback=True)
                                write_json(reviewpath,review);ep["audit"]["supervision"]["verification_artifact"]=record(reviewpath)
                                all_records.append(ep)
                for ep in pending:
                    ep["audit"]["industrial_predictions"]=record(predpath)
                    ep["audit"]["label_release_log"]=record(releasepath)
                    ep["audit"]["replay_ledger"]=record(ledgerpath)
                    all_records.append(ep)
                print(f"Built {s} {a['group_key']} ({len(group)} observations)",flush=True)
        finally:
            proc.stdin.close();proc.wait(timeout=30)
            if proc.returncode:raise RuntimeError("label worker failed")
    all_records.sort(key=lambda r:r["episode_id"])
    for split in ("train","dev"):
        write_rows(output/(split+".jsonl"),(r for r in all_records if r["split"]==split and r["audit"]["eligibility"]["sft_eligible"]))
    write_rows(output/"quarantine.jsonl",(r for r in all_records if not r["audit"]["eligibility"]["sft_eligible"]))
    write_json(output/"snapshots/generation_summary.json",{"counts":counts(all_records),
      "skipped_forecasts":[{"scenario_id":k[0],"split":k[1],"reason":k[2],"count":v} for k,v in sorted(skips.items())],
      "source_families":len(assignment),"natural_device_counts":{"A":1,"B":3,"C":100}})
    write_readme(output,all_records,created_at)
    manifest={"schema_version":"model0.dataset.v1","episode_schema":"model0.episode.v1","evidence_schema":"model0.evidence.v1",
      "dataset_id":output.name,"supersedes":{"dataset_id":"model0_episodes_v1","manifest_sha256":"4f532883a68cb9e41e2ef77b9a296b47c25293d3eb53005ae4d98bf99a747958","episode_id_mapping":"identity; existing IDs preserved"},"created_at":created_at,"generator_version":VERSION,"upstream_manifest":upstream,
      "protocols":[{"id":read_json(p).get("id",read_json(p).get("protocol_id")),"path":p.relative_to(output).as_posix(),"sha256":digest(p)} for p in sorted((output/"protocols").glob("*.json"))],
      "counts":counts(all_records),"limitations":LIMITATIONS,"coverage_status":"partial_by_design","tooling":tooling,"files":[]}
    inventory(output,manifest);write_json(output/"manifest.json",manifest)
    from validate_model0_episodes import validate
    report=validate(output);write_json(output/"validation_report.json",report)
    if report["status"]!="passed":raise ValueError("Dataset validation failed; inspect validation_report.json")
    inventory(output,manifest);write_json(output/"manifest.json",manifest)
    print(canonical({"manifest":str(output/"manifest.json"),"status":report["status"],"records":len(all_records)}))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="model0_episodes_v1_1");p.add_argument("--created-at")
    p.add_argument("--label-worker",choices=list("ABC"));args=p.parse_args()
    if args.label_worker:label_worker(args.label_worker)
    else:build((ROOT/args.output).resolve(),args.created_at or datetime.now(timezone.utc).isoformat())
