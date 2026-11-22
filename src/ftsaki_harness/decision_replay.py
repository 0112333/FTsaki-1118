"""Replay model0 decisions using a frozen window allowlist and no training client."""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import os
from pathlib import Path

from .decision_review import DecisionReview, SOURCES, validate_plan
from .model0_smoke import MODEL, check_config, read, require, send_once
from .policy import Policy, canonical, digest
from .records import Records
from .training_intent import timestamp
from .window_evidence import load_runtime, window_stream


def render(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    tokens = tokens["input_ids"] if hasattr(tokens, "keys") else tokens
    require(tokens and tokenizer.encode(prompt, add_special_tokens=False) == tokens, "Frozen tokenization changed")
    return prompt, {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_tokens": len(tokens)}


def attribution_messages(review, record, evidence):
    cutoff = record["as_of"]["replay_order"]
    window_id = f"{cutoff:012d}"
    vote = review.store.get("vote/" + window_id)
    require(vote and vote["pool"]["trigger"] and vote["event_id"]
            and vote["input_sha256"] == record["evidence_sha256"] == digest(evidence)
            and vote["group"] == evidence["group_id"], "Attribution requires the recorded triggering evidence")
    source_options = {lane: sorted(sources) for lane, sources in SOURCES.items()}
    template = {"evidence_refs":["visible evidence ID"], "hypothesis":"possible cause",
                "uncertainty":"gaps and alternatives", "training_objects":["decision or industrial"],
                "sources":{"decision":["allowed source"], "industrial":["allowed source"]},
                "budget_suggestion":"bounded recommendation"}
    instructions = ("Review the recorded drift event and return one JSON attribution proposal shaped as "
                    + canonical(template) + ". Suggest one or both training lanes for review. Allowed source kinds: "
                    + canonical(source_options)
                    + ". Cite only visible evidence. Treat causes as hypotheses. Budgets and data permissions are set by the controller.")
    request = {"event_id":vote["event_id"], "evidence":evidence, "recorded_decision":vote["decision_output"]}
    return vote, [{"role":"system", "content":instructions},{"role":"user", "content":canonical(request)}]


def run_attribution(review, call_dir, record, evidence, base, grant, tokenizer, key):
    vote, messages = attribution_messages(review, record, evidence)
    event = vote["event_id"]
    prompt, preflight = render(tokenizer, messages)
    require(grant.get("approval") and grant["event"] == event and grant["prompt_sha256"] == preflight["prompt_sha256"],
            "Approve this exact event attribution prompt")
    config = dict(base, transport="text-completions", scenario=review.identity["scenario"], task="decide",
                  episode_id=f"{review.identity['run']}/{event}/attribution", max_prompt_tokens=preflight["prompt_tokens"],
                  max_output_tokens=grant["max_output_tokens"], deadline_seconds=grant["deadline_seconds"],
                  evaluation_scope="2a-attribution")
    check_config(config)
    require(type(grant["max_tokens"]) is int and preflight["prompt_tokens"] + grant["max_output_tokens"] <= grant["max_tokens"],
            "Attribution exceeds approved token budget")
    replay = review.store.get("replay-plan")
    require(replay and replay["checkpoint_id"].startswith("ckpt_"), "Missing parent identity mapping")
    end = timestamp(grant["deadline"])
    review.store.commit({"attribution-grant/" + event:grant})
    leaf = Records(call_dir / f"attribution-{event}.sqlite")
    try:
        leaf.commit({"identity":config, "offline/preflight":preflight})
        receipt = leaf.get("online/result")
        receipt_key = "attribution-result/" + event
        if receipt is None:
            if leaf.get("online/reservation") or review.store.get(receipt_key):
                return {"state":"pending-reconciliation", "event":event}
            require((end - datetime.now(timezone.utc)).total_seconds() > config["deadline_seconds"], "Attribution deadline elapsed")
            def validate(_evidence, content, reason):
                require(reason == "stop", "Incomplete attribution")
                validate_plan(json.loads(content), vote["visible_refs"])
            receipt = send_once(leaf, config, messages, preflight, key, prompt=prompt, validator=validate)
        review.store.commit({receipt_key:receipt})
        if (receipt.get("format_valid") is not True or
                receipt.get("provider_model") not in (MODEL, replay["checkpoint_id"] + "@0")):
            return {"state":"attribution-blocked", "event":event, "receipt":receipt}
        choice = json.loads(leaf.get("online/raw_response"))["choices"][0]
        proposal = json.loads(choice["text"])
        checks = {"decision":{"parent_eligible":True}}
        return {"state":"reviewed", "event":event, "receipt":receipt,
                "plan":review.review_plan(vote["window_id"], proposal, checks)}
    finally:
        leaf.close()


def review_report(store, plan, reason):
    records = store.export()
    rows = [item for name, item in records.items() if name.startswith("vote/")]
    calls = [item for name, item in records.items() if name.startswith("call-result/")]
    events = [{"event": row["event_id"], "cutoff": row["cutoff"],
               "plan": records.get("plan/" + row["event_id"]),
               "attribution_receipt":records.get("attribution-result/" + row["event_id"]), "attribution": "pending"
               if "plan/" + row["event_id"] not in records else "reviewed"}
              for row in rows if row["pool"]["trigger"]]
    return {"run": plan["run"], "source": plan["source"], "stop_reason": reason, "windows": rows, "events": events,
            "reservations": sum(name.startswith("call/") for name in records), "calls": calls,
            "training_submissions": {"industrial": 0, "decision": 0}}


def run_windows(store, call_dir, windows, base, plan, tokenizer, key):
    require(plan["approval"] and plan["parent_authorized"] is True, "Explicit replay and parent approval required")
    require(plan["source"] in ("controlled", "live-http") and plan["scenario"] in ("A", "B", "C"), "Invalid replay scope")
    require(type(plan["max_calls"]) is int and isinstance(plan["windows"], list) and plan["windows"]
            and len(plan["windows"]) == plan["max_calls"],
            "Freeze the exact authorized windows")
    require(plan["checkpoint_id"].startswith("ckpt_"), "Frozen checkpoint mapping required")
    require(type(plan["max_tokens"]) is int and plan["max_tokens"] > 0, "Positive aggregate token budget required")
    limits = []
    for allowed in plan["windows"]:
        require(all(type(allowed[name]) is int and allowed[name] > 0 for name in
                    ("cutoff", "prompt_tokens", "max_output_tokens", "deadline_seconds")), "Invalid item limits")
        limits.append(allowed["prompt_tokens"] + allowed["max_output_tokens"])
    require(sum(limits) <= plan["max_tokens"], "Authorized windows exceed aggregate token budget")
    require(all(first["cutoff"] < second["cutoff"] for first, second in zip(plan["windows"], plan["windows"][1:])),
            "Expected increasing window cutoffs")
    end = timestamp(plan["deadline"])
    review = DecisionReview(store, run=plan["run"], scenario=plan["scenario"], source=plan["source"],
                            model=MODEL, parent_authorized=True)
    store.commit({"replay-plan": plan, "replay-contract": base["contract_sha256"]})
    reason = "authorized-windows-complete"
    for allowed, reserved_tokens in zip(plan["windows"], limits):
        try:
            record, evidence, messages = next(windows)
        except StopIteration:
            reason = "source-ended-early"
            break
        cutoff = allowed["cutoff"]
        require(record["as_of"]["replay_order"] == cutoff and record["scenario"] == plan["scenario"]
                and record["messages_sha256"] == allowed["messages_sha256"] == digest(messages), "Window differs from approved evidence")
        prompt, preflight = render(tokenizer, messages)
        require(preflight["prompt_tokens"] == allowed["prompt_tokens"], "Window differs from approved prompt token count")
        call = str(cutoff)
        saved_vote = store.get(f"vote/{cutoff:012d}")
        if saved_vote:
            if saved_vote["failure"] == "transport-error":
                reason = "transport-error"
                break
            if saved_vote["pool"]["trigger"] and not store.get("plan/" + saved_vote["event_id"]):
                reason = "attribution-pending"
                break
            continue
        remaining = int((end - datetime.now(timezone.utc)).total_seconds())
        if remaining <= 0:
            reason = "deadline"
            break
        leaf = Records(call_dir / f"window-{cutoff}.sqlite")
        try:
            config = dict(base, transport="text-completions", scenario=plan["scenario"], task="decide",
                          episode_id=f"{plan['run']}/{cutoff}", max_prompt_tokens=allowed["prompt_tokens"],
                          max_output_tokens=allowed["max_output_tokens"], deadline_seconds=allowed["deadline_seconds"],
                          evaluation_scope="2a-replay")
            check_config(config)
            leaf.commit({"identity":config, "offline/preflight":preflight})
            prior = leaf.get("online/result")
            reserved = leaf.get("online/reservation")
            if prior is None and (reserved or store.get("call/" + call)):
                reason = "pending-reconciliation"
                break
            if not reserved and remaining < config["deadline_seconds"]:
                reason = "deadline"
                break
            if prior is None:
                with store.db:
                    store.db.execute("BEGIN IMMEDIATE")
                    if store.get("call/" + call):
                        reason = "pending-reconciliation"
                        break
                    store.db.execute("INSERT INTO records VALUES (?,?)", ("call/" + call, canonical({
                        "cutoff":cutoff, "database":f"window-{cutoff}.sqlite", "reserved_tokens":reserved_tokens})))
            receipt = prior or send_once(leaf, config, messages, preflight, key, prompt=prompt)
            store.commit({"call-result/" + call: receipt})
            failure = None
            raw, stop = None, None
            if receipt.get("http_status") == 200:
                supplied = receipt.get("provider_model")
                if supplied not in (MODEL, plan["checkpoint_id"] + "@0"):
                    reason = "unverified-model-identity"
                    break
                if receipt.get("format_valid") is True:
                    choice = json.loads(leaf.get("online/raw_response"))["choices"][0]
                    raw, stop = choice["text"], choice["finish_reason"]
                else:
                    failure = "invalid-response"
            else:
                failure = "transport-error"
            vote = review.accept_window(record, evidence, messages, content=raw, stop=stop, failure=failure)
            if failure == "transport-error":
                reason = failure
                break
            if vote["pool"]["trigger"] and not store.get("plan/" + vote["event_id"]):
                reason = "attribution-pending"
                break
        finally:
            leaf.close()
    return review_report(store, plan, reason)


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "training", "config", "directory", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--attribution", type=Path, help="Review one triggering event with its separate grant")
    args = parser.parse_args()
    plan, base = read(args.plan), read(args.config)
    training = args.training.resolve()
    producer, context, replay, builder, manifest = load_runtime(args.data.resolve(), training, plan["scenario"], base["contract_sha256"])
    require(plan["manifest"] == manifest, "Approved manifest changed")
    tokenizer = AutoTokenizer.from_pretrained(training / "artifacts/preflight/tokenizer", local_files_only=True)
    inputs = Records(args.directory / "evidence.sqlite")
    try:
        inputs.commit({"identity":{"manifest":manifest, "scenario":plan["scenario"],
                                  "policy_sha256":Policy().sha256, "contract":base["contract_sha256"], "label_budget":0}})
        windows = window_stream(inputs, producer, context, replay, builder)
        if args.attribution:
            grant = read(args.attribution)
            review = Records(args.directory / "review.sqlite")
            try:
                require(review.get("replay-plan") == plan, "Resume with the original plan")
                reviewer = DecisionReview(review, run=plan["run"], scenario=plan["scenario"], source=plan["source"],
                                          model=MODEL, parent_authorized=plan["parent_authorized"])
                matching = [value for name, value in review.export().items() if name.startswith("vote/")
                            and value["event_id"] == grant["event"] and value["pool"]["trigger"]]
                require(len(matching) == 1, "Unknown attribution event")
                for record, evidence, _ in islice(windows, plan["max_calls"]):
                    if record["as_of"]["replay_order"] == matching[0]["cutoff"]:
                        if args.send:
                            require(plan["source"] == "live-http", "Live attribution requires live review")
                            report = run_attribution(reviewer, args.directory, record, evidence, base, grant, tokenizer,
                                                     os.environ.get(base["credential_env"]))
                        else:
                            _, messages = attribution_messages(reviewer, record, evidence)
                            _, preflight = render(tokenizer, messages)
                            report = {"event":grant["event"], **preflight}
                        print(canonical(report))
                        break
                else:
                    raise ValueError("Triggering evidence not found")
            finally:
                review.close()
        elif args.send:
            require(plan["source"] == "live-http" and os.environ.get(base["credential_env"]), "Live credentials and plan required")
            review = Records(args.directory / "review.sqlite")
            try:
                print(canonical(run_windows(review, args.directory, windows, base, plan, tokenizer,
                                            os.environ[base["credential_env"]])))
            finally:
                review.close()
        else:
            for record, _, messages in islice(windows, plan["max_calls"]):
                _, info = render(tokenizer, messages)
                print(canonical({"cutoff":record["as_of"]["replay_order"], "messages_sha256":record["messages_sha256"], **info}))
    finally:
        inputs.close()


if __name__ == "__main__":
    main()
