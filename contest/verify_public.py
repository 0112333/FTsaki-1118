"""Run one isolated SDK round per public development kit. Model replies are local fixtures."""
import argparse
import base64
import hashlib
import json
from itertools import islice
from pathlib import Path
import subprocess
import tempfile

IMAGE = "ftsaki-contest:model0-1118"
SAMPLER = "trio://run_ogvplibsgt3s/sampler_weights/trial-final"


def command():
    return ["docker", "run", "--rm", "--network=none", "--read-only", "-i",
            "--tmpfs", "/tmp", "--tmpfs", "/workspace:uid=65534,gid=65534,mode=0700"]


def mount(source, target):
    return ["--mount", f"type=bind,src={source.resolve()},dst={target},readonly"]


def project(source, target, count):
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in (stream if count is None else islice(stream, count))]
    if not rows or (count is not None and len(rows) != count):
        raise ValueError(f"insufficient records in {source}")
    target.write_text("".join(json.dumps({k: row[k] for k in ("id", "inputs", "answer")}) + "\n"
                              for row in rows), encoding="utf-8")
    return rows


def verify(key, kit, work, image, full):
    scenario = "classification" if key == "A" else "regression"
    checkpoint = kit / "checkpoint.npz"
    checkpoint_digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = project(kit / "public/train.jsonl", work / "train.jsonl", None if full else 64)
    validation = project(kit / "public/validation.jsonl", work / "validation.jsonl", None if full else 16)
    parent = {"artifact_id": "local-parent", "artifact_sha256": checkpoint_digest,
              "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                           "predict_argv": ["python3", "/app/harness.py", "predict"]},
              "files": [{"path": "checkpoint.npz", "sha256": checkpoint_digest,
                         "size_bytes": checkpoint.stat().st_size}]}
    task = {"task": scenario, "dataset_key": key, "metric": "macro_f1" if key == "A" else "rmse",
            "metric_direction": "higher" if key == "A" else "lower",
            "public_projection": {"train_path": "/data/train.jsonl", "validation_path": "/data/validation.jsonl"}}
    frame = {"protocol": "contest.harness.workspace.v2", "dataset_key": key, "round_index": 1,
             "stage_id": "public-smoke", "run_id": "local-test", "rounds_per_dataset": 1,
             "prompt": "Local protocol smoke test", "prompt_version": "industrial-workspace-flexible-v3",
             "task_contract": task, "parent_checkpoint_sha256": checkpoint_digest, "parent_artifact": parent,
             "artifacts": [parent], "model_name": SAMPLER, "max_output_tokens": 256,
             "resource_limits": {"candidate_upload_bytes_per_round": 64 * 1024 * 1024}}
    args = command() + mount(work / "train.jsonl", "/data/train.jsonl") + mount(work / "validation.jsonl", "/data/validation.jsonl")
    args += mount(checkpoint, "/previous/checkpoint.npz") + [image, "train"]
    selected = "local-candidate"
    local_call = "local-test-call"
    seen = []
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        try:
            proc.stdin.write(json.dumps(frame) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                event = json.loads(line)
                kind = event["type"]
                seen.append(kind)
                if kind == "candidate_request":
                    if event["parent_checkpoint_sha256"] != checkpoint_digest:
                        raise ValueError("candidate parent mismatch")
                    file, = event["files"]
                    if file["path"] != "checkpoint.npz":
                        raise ValueError("unexpected artifact")
                    candidate = base64.b64decode(file["data_base64"], validate=True)
                    if "sha256:" + hashlib.sha256(candidate).hexdigest() == checkpoint_digest:
                        raise ValueError("candidate unchanged")
                    (work / "candidate.npz").write_bytes(candidate)
                    reply = {"type": "candidate_response", "id": event["id"], "artifact_id": selected, "summary": {}}
                elif kind == "history_request":
                    reply = {"type": "history_response", "id": event["id"], "records": []}
                elif kind == "model_request":
                    selection = json.dumps({"selected_artifact_id": selected, "reason": "Local protocol fixture."})
                    reply = {"type": "model_response", "id": event["id"], "call_id": local_call,
                             "response": {"choices": [{"message": {"role": "assistant", "content": selection}}]}}
                elif kind == "round_complete":
                    record = event["record"]
                    if record["model_call_ids"] != [local_call] or record["selected_artifact_id"] != selected:
                        raise ValueError("invalid round receipt")
                    reply = {"type": "round_ack", "accepted": True}
                else:
                    raise RuntimeError(f"{key}: unexpected event {event.get('error', kind)}")
                proc.stdin.write(json.dumps(reply) + "\n")
                proc.stdin.flush()
                if kind == "round_complete":
                    break
        finally:
            proc.stdin.close()
        if proc.wait(timeout=30) != 0 or seen != ["candidate_request", "history_request", "model_request", "round_complete"]:
            raise RuntimeError(f"{key}: incomplete round {seen}")
    sample = {"task": scenario, "inputs": validation[0]["inputs"]}
    predictions = []
    for chosen in (checkpoint, work / "candidate.npz"):
        args = command() + mount(chosen, "/model/checkpoint.npz")
        args += ["-e", "CONTEST_ARTIFACT_PATH=/model/checkpoint.npz", image, "predict"]
        raw = subprocess.check_output(args, input=json.dumps(sample) + "\n", text=True, timeout=30)
        predictions.append(json.loads(raw)["outputs"])
    if key == "A" and not all(len(output) == 6 for output in predictions):
        raise ValueError("expected six gas scores")
    print(f"{key} OK | public rows {len(training)} train / {len(validation)} validation | candidate + local round ack | parent and candidate predict")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, help="directory containing the three cpu-a/b/c development kits")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--full", action="store_true", help="use all rows in both public splits")
    args = parser.parse_args()
    descriptor = json.loads(subprocess.check_output(command() + [args.image, "describe"], text=True))
    if descriptor["protocol"] != "contest.harness.workspace.v2" or descriptor["model_access"] != "host_gateway_rpc":
        raise ValueError("incompatible adapter")
    for key in "ABC":
        kit, = args.sources.glob(f"cpu-{key.lower()}-*")
        with tempfile.TemporaryDirectory(prefix="ftsaki-public-") as directory:
            verify(key, kit, Path(directory), args.image, args.full)


if __name__ == "__main__":
    main()
