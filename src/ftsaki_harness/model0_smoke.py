"""Single-call round-one acceptance for the frozen PyTRiO trial adapter."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from http.client import HTTPSConnection
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys

from .inference import validate_content, validate_hypotheses
from .policy import canonical, digest
from .records import Records

MODEL = "trio://run_ogvplibsgt3s/sampler_weights/trial-final"
BASE = "https://pytrio.com/api/openai/v1"
TASKS = ("decide", "forecast_next")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as handle:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_config(config):
    require(config["base_url"] == BASE and config["model"] == MODEL, "Unverified endpoint or weights")
    require(config["message_version"] == "model0.messages.v3", "Wrong message contract")
    require(config["credential_env"] == "PYTRIO_API_KEY", "Unexpected credential source")
    require(type(config["max_calls"]) is int and config["max_calls"] == 1, "Only one call permitted")
    for name in ("deadline_seconds", "max_output_tokens", "max_prompt_tokens"):
        require(type(config[name]) is int and config[name] > 0, "Missing positive limit: " + name)
    require(config["scenario"] in ("A", "B", "C") and config["task"] in TASKS, "Unknown grid cell")
    require(isinstance(config["episode_id"], str) and config["episode_id"], "Missing episode")
    require(isinstance(config["contract_sha256"], str) and len(config["contract_sha256"]) == 64, "Missing frozen digest")
    scope = config.get("evaluation_scope", "dev-format")
    require(scope in ("dev-format", "known-train-format-only", "2a-replay", "2a-attribution", "2b-reload",
                      "3-replay", "3-forecast"), "Unknown sample scope")
    if scope == "3-forecast":
        require(config["task"] == "forecast_next" and config.get("transport") == "text-completions",
                "Stage 3 forecast requires frozen text forecast input")
    if scope in ("2a-replay", "2a-attribution", "2b-reload", "3-replay"):
        require(config["task"] == "decide" and config.get("transport") == "text-completions", "2a requires text decision input")
    if config.get("evaluation_scope") == "known-train-format-only":
        require(config["scenario"] == "A" and config["task"] == "forecast_next", "Training probe is A/forecast_next only")


def plan_cell(base, plan, cell):
    require(cell in plan, "Cell is not in remaining plan")
    scenario, task = cell.split("/")
    options = plan[cell]
    require(set(options) == {"episode_id", "evaluation_scope", "max_prompt_tokens", "max_output_tokens", "deadline_seconds"},
            "Wrong plan fields")
    config = dict(base, scenario=scenario, task=task, **options)
    config["transport"] = "text-completions"
    check_config(config)
    return config


def prepare(config, training, dataset):
    from transformers import AutoTokenizer
    contract_path = training / "artifacts/preflight/a_forecast_v2/frozen-contract.json"
    require(file_hash(contract_path) == config["contract_sha256"], "Contract digest mismatch")
    contract = read(contract_path)
    require(contract["message_version"] == config["message_version"], "Version mismatch")
    for name, expected in contract["files"].items():
        require(file_hash(training / name) == expected, "Frozen file changed: " + name)
    manifest = dataset / "manifest.json"
    require(file_hash(manifest) == contract["manifest_sha256"], "Dataset manifest mismatch")
    descriptors = {item["path"]: item for item in read(manifest)["files"]}
    sys.path.insert(0, str(training / "scripts"))
    from model0_messages import project
    from model0_runtime import runtime_messages
    tokenizer = AutoTokenizer.from_pretrained(training / "artifacts/preflight/tokenizer", local_files_only=True)
    gate_dir = training / "artifacts/training/trial_messages_v3/acceptance"
    gate = read(gate_dir / "a-output-gate.json")
    train_probe = config.get("evaluation_scope") == "known-train-format-only"
    if train_probe:
        require(config["episode_id"] == gate["case_id"] and gate["independent_dev_metric"] is False,
                "Only the documented A format control may use the train split")
    selected_split = "train" if train_probe else "dev"
    selected = None
    examples = {}
    control = None
    for split in ("dev", "train"):
        filename = split + ".jsonl"
        require(file_hash(dataset / filename) == descriptors[filename]["sha256"], "Split digest mismatch")
        with (dataset / filename).open(encoding="utf-8") as handle:
            for line in handle:
                episode = json.loads(line)
                identity = episode["episode_id"]
                cell = episode["scenario_id"] + "/" + episode["task"]
                if split == selected_split and identity == config["episode_id"]:
                    require(episode["split"] == split, "Record split mismatch")
                    require(cell == config["scenario"] + "/" + config["task"], "Wrong selected cell")
                    selected = runtime_messages(episode["input"], episode["task"])
                    require(selected == project(episode)[:2], "Selected messages differ")
                is_control = identity == gate["case_id"] and split == "train"
                if is_control:
                    control = episode
                if cell not in examples or is_control:
                    messages = runtime_messages(episode["input"], episode["task"])
                    require(messages == project(episode)[:2], "Message mismatch: " + cell)
                    require([message["role"] for message in messages] == ["system", "user"], "Role leak")
                    projected = json.loads(messages[1]["content"])
                    require(not {"target", "audit", "episode_id"}.intersection(projected), "Episode metadata leak")
                    require(projected["feature_order"] == episode["input"]["feature_order"], "Changed feature order")
                    examples[cell] = {"episode_id": identity, "split": split, "messages_sha256": digest(messages)}
                if split == "train" and control and len(examples) == 6:
                    break
    require(selected and control and len(examples) == 6, "Incomplete preflight")
    filename = gate_dir / (gate["case_id"] + ".json")
    require(file_hash(filename) == gate["raw_response_sha256"], "Historical response changed")
    sequence = read(filename)["sequences"][0]
    tokens = list(sequence["tokens"])
    while tokens and tokens[-1] == tokenizer.eos_token_id:
        tokens.pop()
    content = tokenizer.decode(tokens, skip_special_tokens=False)
    require(hashlib.sha256(content.encode()).hexdigest() == gate["trial_text_sha256"], "Decode mismatch")
    rows = json.loads(content)["values"]
    require(len(rows) == 34 and all(len(row) == 128 for row in rows), "Unexpected historical shape")
    try:
        validate_content("forecast_next", control["input"], content, sequence["stop_reason"])
    except ValueError as exc:
        require(str(exc) == "Wrong horizon", "Unexpected rejection: " + str(exc))
    else:
        raise ValueError("Accepted 34 rows")
    validate_content("forecast_next", control["input"], canonical(control["target"]), "stop")
    ids = tokenizer.apply_chat_template(selected, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    count = len(ids)
    require(0 < count <= config["max_prompt_tokens"], "Input exceeds token budget")
    preflight = {"message_checks": examples, "prompt_tokens": count, "messages_sha256": digest(selected),
                      "manifest_sha256": contract["manifest_sha256"], "historical_raw_sha256": gate["raw_response_sha256"],
                      "historical_34x128": "rejected", "target_control_32x128": "accepted"}
    prompt = None
    if config.get("transport") == "text-completions":
        prompt = tokenizer.apply_chat_template(selected, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(tokenizer.encode(prompt, add_special_tokens=False) == ids, "Rendered text changed frozen token IDs")
        preflight.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), input_ids_sha256=digest(ids))
    return selected, preflight, prompt


def post(body, key, timeout, path):
    connection = HTTPSConnection("pytrio.com", timeout=timeout)
    try:
        connection.request("POST", path, body=canonical(body).encode(),
                           headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read(2_000_001)
        require(len(data) <= 2_000_000, "Oversized response")
        return response.status, data
    finally:
        connection.close()


def send_once(store, config, messages, preflight, key, *, prompt=None, validator=None, model=MODEL):
    require(bool(key), "Credential unavailable; no call sent")
    transport = config.get("transport", "chat-completions")
    require(transport in ("text-completions", "chat-completions"), "Unknown HTTP transport")
    body = {"model": model, "max_tokens": config["max_output_tokens"], "temperature": 0}
    if transport == "text-completions":
        require(isinstance(prompt, str) and prompt and hashlib.sha256(prompt.encode()).hexdigest() == preflight["prompt_sha256"],
                "Missing or changed frozen prompt")
        body["prompt"] = prompt
        path = "/api/openai/v1/completions"
    else:
        body.update(messages=messages, seed=42, chat_template_kwargs={"enable_thinking": False})
        path = "/api/openai/v1/chat/completions"
    seconds = config["deadline_seconds"]
    deadline = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    reservation = {"deadline_utc": deadline.isoformat(), "request_sha256": digest(body), "max_calls": 1,
                   "token_reservation": config["max_prompt_tokens"] + config["max_output_tokens"]}
    with store.db:
        try:
            store.db.execute("INSERT INTO records VALUES (?,?)", ("online/reservation", canonical(reservation)))
        except sqlite3.IntegrityError:
            raise ValueError("Call already reserved; inspect existing results, do not retry") from None
    report = {"source": "live-http", "episode_id": config["episode_id"], "cell": config["scenario"] + "/" + config["task"],
              "request_sha256": reservation["request_sha256"], "transport": transport, "expected_model": model, "usage": None,
              "local_prompt_tokens": preflight["prompt_tokens"], "currency_cost": None, "status": "blocked"}
    report["evaluation_scope"] = config.get("evaluation_scope", "dev-format")
    def expired(signum, frame):
        raise TimeoutError("Round deadline reached")
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.alarm(seconds)
        status, data = post(body, key, seconds, path)
        report["http_status"] = status
        report["response_sha256"] = hashlib.sha256(data).hexdigest()
        require(key.encode() not in data, "Credential echoed; raw response withheld")
        store.commit({"online/raw_response": data.decode("utf-8")})
        if status != 200:
            report["reason"] = "HTTP " + str(status)
        else:
            outer = json.loads(data)
            report.update(provider_model=outer.get("model"), usage=outer.get("usage"))
            choice = outer["choices"][0]
            reason = choice.get("finish_reason")
            report["finish_reason"] = reason
            report["status"] = "invalid-output"
            evidence = json.loads(messages[1]["content"])
            content = choice["text"] if transport == "text-completions" else choice["message"]["content"]
            if config["task"] == "forecast_next":
                parsed = json.loads(content)
                values = parsed.get("values") if isinstance(parsed, dict) else None
                if isinstance(values, list):
                    report.update(rows=len(values), row_widths=sorted({len(row) for row in values if isinstance(row, list)}),
                                  non_array_rows=sum(not isinstance(row, list) for row in values))
            if validator is not None:
                validator(evidence, content, reason)
            elif config["task"] == "decide":
                result = validate_content("decide", evidence, content, reason)
                validate_hypotheses(result, evidence)
                report.update(decision=result["decision"], evidence_refs=result["evidence_refs"])
            else:
                result = validate_content("forecast_next", evidence, content, reason)
                report.update(rows=len(result["values"]), width=len(result["feature_order"]))
            report["format_valid"] = True
            report["status"] = "passed" if outer.get("model") == model else "unverified-model-echo"
    except (ValueError, KeyError, TypeError, IndexError, OSError, RuntimeError, ArithmeticError) as exc:
        report["reason"] = str(exc)[:250].replace(key, "[redacted]")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    store.commit({"online/result": report})
    return report


def record_checkpoint_match(store, checkpoint):
    result = store.get("online/result")
    require(result and result["status"] == "unverified-model-echo" and result.get("format_valid") is True
            and result["http_status"] == 200, "No valid response awaiting identity verification")
    identity = checkpoint["checkpoint_id"]
    require(checkpoint["path"] == result["expected_model"] == MODEL and checkpoint["checkpoint_type"] == "sampler",
            "Wrong checkpoint artifact")
    require(isinstance(identity, str) and identity.startswith("ckpt_") and identity + "@0" == result["provider_model"],
            "Checkpoint ID mismatch")
    store.commit({"online/checkpoint_match": {"source": "pytrio.list_checkpoints", "path": checkpoint["path"],
                  "checkpoint_id": identity, "provider_model": result["provider_model"],
                  "response_sha256": result["response_sha256"]}})


def grid_report(store):
    initial = store.get("online/initial_grid")
    require(initial is not None, "Missing acceptance grid")
    result = store.get("online/result")
    reserved = store.get("online/reservation") is not None
    config = store.get("identity")
    match = store.get("online/checkpoint_match")
    if reserved:
        cell = config["scenario"] + "/" + config["task"]
        initial[cell] = result["status"] if result else "reserved-unknown"
        if result and match and match["response_sha256"] == result["response_sha256"]:
            initial[cell] = "passed"
    return {"source": "live-http", "grid": initial, "reservations": int(reserved),
            "result": result, "checkpoint_match": match, "offline": store.get("offline/preflight")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "training", "dataset", "database"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--send", action="store_true", help="Submit one real request (Linux only)")
    parser.add_argument("--text-completions", action="store_true", help="Use the frozen rendered prompt and a separate database")
    parser.add_argument("--grid-plan", type=Path, help="Frozen remaining single-call cells")
    parser.add_argument("--cell", help="Select one planned cell, e.g. C/forecast_next; does not imply --send")
    args = parser.parse_args()
    config = read(args.config)
    if args.grid_plan or args.cell:
        require(args.grid_plan and args.cell, "Both --grid-plan and --cell are required")
        config = plan_cell(config, read(args.grid_plan), args.cell)
    if args.text_completions:
        config["transport"] = "text-completions"
    check_config(config)
    messages, preflight, prompt = prepare(config, args.training.resolve(), args.dataset)
    store = Records(args.database)
    try:
        require(not store.export() or store.get("identity") == config, "Not this acceptance database")
        cells = {scenario + "/" + task: "not-run" for scenario in "ABC" for task in TASKS}
        store.commit({"identity": config, "offline/preflight": preflight, "online/initial_grid": cells})
        if args.send:
            report = send_once(store, config, messages, preflight, os.environ.get(config["credential_env"]), prompt=prompt)
            print(canonical(grid_report(store)))
            return int(report["status"] != "passed")
        print(canonical(grid_report(store)))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
