"""Shared inference budget for a bounded, durable real acceptance run."""
from datetime import datetime, timezone
from pathlib import Path

from .decision_replay import render
from .model0_smoke import check_config, require, send_once
from .policy import canonical, digest
from .records import Records
from .training_intent import timestamp


class BoundedInference:
    """Reserve all model0/model1 calls in one immutable ledger before HTTP."""
    def __init__(self, store, directory, grant, base, tokenizer, key):
        self.store, self.directory, self.grant = store, Path(directory), grant
        self.base, self.tokenizer, self.key = base, tokenizer, key
        require(grant.get("approval") and type(grant.get("max_inference_calls")) is int
                and type(grant.get("max_inference_tokens")) is int
                and grant["max_inference_calls"] > 0 and grant["max_inference_tokens"] > 0,
                "Approved aggregate inference limits required")
        store.commit({"identity":{"grant_sha256":digest(grant), "base":base}})

    def reserve(self, name, spec, preflight):
        require(name and all(character.isalnum() or character in "-_" for character in name), "Invalid call name")
        require(type(spec.get("max_output_tokens")) is int and spec["max_output_tokens"] > 0
                and type(spec.get("deadline_seconds")) is int and spec["deadline_seconds"] > 0,
                "Positive per-call limits required")
        claim = dict(spec, preflight=preflight, reserved_tokens=preflight["prompt_tokens"]+spec["max_output_tokens"])
        key = "call/" + name
        with self.store.db:
            self.store.db.execute("BEGIN IMMEDIATE")
            previous = self.store.get(key)
            if previous:
                require(previous == claim, "Frozen call changed")
                return claim
            seconds = (timestamp(self.grant["inference_deadline"]) - datetime.now(timezone.utc)).total_seconds()
            require(seconds > spec["deadline_seconds"], "Inference deadline elapsed")
            entries = [row for name,row in self.store.export().items() if name.startswith("call/")]
            require(len(entries) < self.grant["max_inference_calls"]
                    and sum(row["reserved_tokens"] for row in entries)+claim["reserved_tokens"]
                    <= self.grant["max_inference_tokens"], "Aggregate inference budget exhausted")
            self.store.db.execute("INSERT INTO records VALUES (?,?)", (key, canonical(claim)))
        return claim

    def request(self, name, messages, *, task, artifact, checkpoint_id, output_tokens, seconds):
        prompt, preflight = render(self.tokenizer, messages)
        scope = "3-forecast" if task == "forecast_next" else "3-replay"
        config = dict(self.base, transport="text-completions", task=task, scenario="C", episode_id=name,
                      evaluation_scope=scope, max_prompt_tokens=preflight["prompt_tokens"],
                      max_output_tokens=output_tokens, deadline_seconds=seconds)
        check_config(config)
        spec = {"artifact":artifact, "checkpoint_id":checkpoint_id, "task":task,
                "max_output_tokens":output_tokens, "deadline_seconds":seconds, "messages_sha256":digest(messages)}
        self.reserve(name, spec, preflight)
        leaf = Records(self.directory / (name + ".sqlite"))
        try:
            leaf.commit({"identity":{"config":config,"artifact":artifact,"checkpoint_id":checkpoint_id},
                         "offline/preflight":preflight})
            receipt = leaf.get("online/result")
            if receipt is None:
                require(not leaf.get("online/reservation"), "Call requires reconciliation; do not resend")
                remaining = (timestamp(self.grant["inference_deadline"])-datetime.now(timezone.utc)).total_seconds()
                require(remaining > seconds, "Inference deadline elapsed")
                receipt = send_once(leaf, config, messages, preflight, self.key, prompt=prompt, model=artifact)
            self.store.commit({"receipt/"+name:receipt})
            if receipt.get("http_status") == 200:
                require(receipt.get("expected_model") == artifact and
                        receipt.get("provider_model") in (artifact,checkpoint_id+"@0"), "Unverified artifact identity")
            return receipt, leaf.get("online/raw_response")
        finally:
            leaf.close()

    def summary(self):
        records = self.store.export()
        claims = [row for name,row in records.items() if name.startswith("call/")]
        receipts = [row for name,row in records.items() if name.startswith("receipt/")]
        usage = [row["usage"] for row in receipts if isinstance(row.get("usage"),dict)]
        return {"reserved_calls":len(claims), "reserved_tokens":sum(row["reserved_tokens"] for row in claims),
                "receipts":len(receipts), "reported_total_tokens":sum(row.get("total_tokens",0) for row in usage),
                "unknown_usage_calls":len(claims)-len(usage), "currency_cost":None}
