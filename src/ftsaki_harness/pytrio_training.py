"""Bounded PyTRiO SFT, with one submission and metadata-only reconciliation."""
from contextlib import contextmanager
from datetime import datetime, timezone
from math import isfinite
import signal
from urllib.parse import urlsplit

from .decision_replay import render
from .model0_smoke import check_config, send_once
from .policy import canonical, digest
from .training_intent import require, timestamp


@contextmanager
def deadline(seconds):
    def expire(signum, frame):
        raise TimeoutError("PyTRiO deadline elapsed")

    previous = signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sft_datums(sdk, intent, token_cases):
    require(set(token_cases) == {case["id"] for case in intent["cases"]}, "Token cases differ from frozen snapshot")
    datums = []
    for case in intent["cases"]:
        sample = token_cases[case["id"]]
        prompt, answer = sample["prompt_ids"], sample["answer_ids"]
        require(isinstance(prompt, list) and isinstance(answer, list) and prompt and answer
                and all(type(token) is int and token >= 0 for token in prompt + answer), "Invalid token sequence")
        require(digest(prompt) == case["input_sha256"] and digest(answer) == case["target_sha256"], "Frozen tokens changed")
        ids = prompt + answer
        require(len(ids) == case["sequence_tokens"], "Frozen token count changed")
        weights = [0.] * (len(prompt) - 1) + [1. / len(answer)] * len(answer)
        datums.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(ids[:-1]),
                                loss_fn_inputs={"target_tokens": ids[1:], "weights": weights}))
    return datums


class PyTrioTraining:
    def __init__(self, ledger, sdk, configuration):
        self.ledger, self.sdk, self.configuration = ledger, sdk, configuration

    def _client(self, intent):
        base = urlsplit(self.configuration.base_url)
        require((base.scheme, base.netloc) == ("https", "pytrio.com"), "Unapproved PyTRiO origin")
        require(intent["budget"]["service"] == "pytrio", "Intent targets another service")
        for name, value in {"retry_max_attempts": 1, "training_retry_timeout_seconds": 0,
                            "control_retry_timeout_seconds": 0, "actor_capacity_wait_timeout_seconds": 30}.items():
            setattr(self.configuration, name, value)
        client = self.sdk.ServiceClient()
        require(client.workspace_cuid == intent["budget"]["workspace"], "Wrong PyTRiO workspace")
        return client

    def _verify_parent(self, rest, parent, wait):
        checkpoint = parent["checkpoint"]
        sampler = parent["model"]
        parsed = urlsplit(checkpoint)
        require(parsed.scheme == "trio" and parsed.netloc and parsed.path.startswith("/weights/")
                and sampler.startswith(f"trio://{parsed.netloc}/sampler_weights/"), "Expected parent state and sampler from one run")
        require(parent.get("binding_approval"), "Approval linking sampler to optimizer state required")
        run = wait(rest.get_training_run(parsed.netloc))
        require(run.is_lora and not run.corrupted and run.training_run_id == parsed.netloc
                and run.base_model == parent["base_model"] and run.lora_rank == parent["rank"], "Parent architecture mismatch")
        checkpoints = {item.path: item for item in wait(rest.list_checkpoints(parsed.netloc)).checkpoints}
        require(checkpoint in checkpoints and sampler in checkpoints
                and checkpoints[checkpoint].checkpoint_type == "training" and checkpoints[sampler].checkpoint_type == "sampler",
                "Parent state or sampler metadata missing")

    def submit(self, run, event, token_cases):
        intent = self.ledger.intent(run, event)
        budget, method, parent = (intent[key] for key in ("budget", "method", "parent"))
        require(method["algorithm"] == "sft-cross-entropy-one-pass", "Training method is unsupported")
        require(len(intent["cases"]) <= budget["max_updates"], "Update budget exceeded")
        parameters = method["parameters"]
        require(set(parameters) == {"learning_rate", "grad_clip_norm", "seed"}, "Freeze optimizer parameters")
        require(type(parameters["seed"]) is int and 0 <= parameters["seed"] < 2**32
                and all(type(parameters[key]) in (int, float) and isfinite(parameters[key]) and parameters[key] > 0
                        for key in ("learning_rate", "grad_clip_norm")), "Invalid optimizer parameters")
        datums = sft_datums(self.sdk, intent, token_cases)
        optimizer = self.sdk.AdamParams(learning_rate=parameters["learning_rate"], grad_clip_norm=parameters["grad_clip_norm"])
        end = timestamp(budget["deadline"])

        def left():
            seconds = (end - datetime.now(timezone.utc)).total_seconds()
            require(seconds > 0, "Training deadline elapsed")
            return seconds

        def wait(future):
            return future.result(timeout=min(left(), 120))

        left()
        claim = f"submission/{run}/{event}"
        with self.ledger.store.db:
            self.ledger.store.db.execute("BEGIN IMMEDIATE")
            require(self.ledger.store.get(claim) is None and self.ledger.reconcile(run, event)["handle"] is None,
                    "Submission already started. Reconcile its handle.")
            self.ledger.store.db.execute("INSERT INTO records VALUES (?,?)",
                                         (claim, canonical({"intent_sha256": digest(intent), "deadline": budget["deadline"]})))
        phase = "parent-verification"
        try:
            with deadline(left()):
                client = self._client(intent)
                rest = client.create_rest_client()
                self._verify_parent(rest, parent, wait)
                phase = "create-training-run"
                trainer = client.create_lora_training_client(base_model=parent["base_model"], rank=parent["rank"],
                                                               seed=parameters["seed"])
                handle = trainer.training_run_cuid
                self.ledger.record_handle(run, event, handle)
                phase = "restore-parent"
                wait(trainer.load_state_with_optimizer(parent["checkpoint"]))
                for step, (case, datum) in enumerate(zip(intent["cases"], datums), start=1):
                    phase = "forward-backward"
                    self.ledger.store.commit({f"step-intent/{run}/{event}/{step}": {
                        "case": case["id"], "sequence_tokens": case["sequence_tokens"]}})
                    backward = wait(trainer.forward_backward([datum], loss_fn="cross_entropy", auto_shift=False))
                    require(all(isfinite(metric) for metric in backward.metrics.values()), "Nonfinite training metrics")
                    phase = "optimizer"
                    wait(trainer.optim_step(optimizer))
                    self.ledger.store.commit({f"step-done/{run}/{event}/{step}": {"case": case["id"], "metrics": backward.metrics}})
                phase = "save-sampler-weights"
                output = wait(trainer.save_weights_for_sampler("stage2b-model1"))
                phase = "verify-artifact"
                require(output.path == f"trio://{handle}/sampler_weights/stage2b-model1" and output.path != parent["model"]
                        and output.model == parent["base_model"] and output.size > 0,
                        "Unexpected sampler artifact")
                assets = wait(rest.list_checkpoints(handle)).checkpoints
                artifact, = [item for item in assets if item.path == output.path and item.checkpoint_type == "sampler"]
                result = {"status":"artifact-ready", "handle":handle, "artifact":output.path, "model_id":output.path,
                          "checkpoint_id":artifact.checkpoint_id, "parent":parent["model"], "updates":len(datums),
                          "sequence_tokens_submitted":intent["sequence_tokens"], "currency_cost":None}
        except Exception as exc:
            result = {"status":"awaiting-reconciliation", "phase":phase, "error_type":type(exc).__name__,
                      "handle":self.ledger.reconcile(run, event)["handle"]}
        self.ledger.store.commit({f"submission-result/{run}/{event}":result})
        return result

    def reconcile(self, run, event):
        pending = self.ledger.reconcile(run, event)
        if not pending["handle"]:
            return pending
        client = self._client(self.ledger.intent(run, event))
        rest = client.create_rest_client()
        handle = pending["handle"]
        remote = rest.get_training_run(handle).result(timeout=30)
        require(remote.training_run_id == handle, "Query returned the wrong run")
        checkpoints = rest.list_checkpoints(handle).result(timeout=30).checkpoints
        return {"status":"metadata-reconciled", "handle":handle, "corrupted":remote.corrupted,
                "sampler_artifacts":[item.path for item in checkpoints if item.checkpoint_type == "sampler"],
                "local_result":self.ledger.store.get(f"submission-result/{run}/{event}")}

    def probe_model1(self, run, event, probe_store, base, messages, tokenizer, grant, key):
        intent = self.ledger.intent(run, event)
        handle = self.ledger.reconcile(run, event)["handle"]
        job = self.ledger.store.get(f"submission-result/{run}/{event}")
        require(job and job["status"] == "artifact-ready" and handle and job["handle"] == handle
                and job["artifact"] == job["model_id"] == f"trio://{handle}/sampler_weights/stage2b-model1"
                and job["parent"] == intent["parent"]["model"] and job["artifact"] != job["parent"],
                "Verified stage 2b child artifact required")
        require([part.get("role") for part in messages] == ["system", "user"], "Reload requires system/user input")
        prompt, preflight = render(tokenizer, messages)
        require(grant.get("approval") and grant["artifact"] == job["artifact"] and grant["event"] == event
                and grant["prompt_sha256"] == preflight["prompt_sha256"], "Approve the child and exact reload prompt")
        config = dict(base, transport="text-completions", scenario=intent["scenario"], task="decide",
                      episode_id=f"{run}/{event}/model1-reload", evaluation_scope="2b-reload",
                      max_prompt_tokens=preflight["prompt_tokens"], max_output_tokens=grant["max_output_tokens"],
                      deadline_seconds=grant["deadline_seconds"])
        check_config(config)
        require(type(grant["max_tokens"]) is int and preflight["prompt_tokens"] + grant["max_output_tokens"] <= grant["max_tokens"],
                "Child reload exceeds approved token budget")
        approval_key = f"reload-grant/{run}/{event}"
        self.ledger.store.commit({approval_key:{"grant":grant,"config":config,"checkpoint_id":job["checkpoint_id"]}})
        probe_store.commit({"identity":{"config":config,"artifact":job["artifact"],"checkpoint_id":job["checkpoint_id"]},
                            "offline/preflight":preflight})
        terminal = probe_store.get("online/result")
        if terminal is None:
            if probe_store.get("online/reservation"):
                return {"status":"awaiting-reconciliation","artifact":job["artifact"]}
            require((timestamp(grant["deadline"]) - datetime.now(timezone.utc)).total_seconds()
                    > config["deadline_seconds"], "Reload deadline elapsed")
            claim = f"reload-submission/{run}/{event}"
            with self.ledger.store.db:
                self.ledger.store.db.execute("BEGIN IMMEDIATE")
                if self.ledger.store.get(claim):
                    return {"status":"awaiting-reconciliation","artifact":job["artifact"]}
                self.ledger.store.db.execute("INSERT INTO records VALUES (?,?)", (claim, canonical({
                    "artifact":job["artifact"],"prompt_sha256":preflight["prompt_sha256"]})))
            terminal = send_once(probe_store, config, messages, preflight, key, prompt=prompt, model=job["artifact"])
        compatible_id = (job["artifact"], job["checkpoint_id"] + "@0")
        success = terminal.get("http_status") == 200 and terminal.get("expected_model") == job["artifact"] and (
            terminal.get("format_valid") is True and terminal.get("provider_model") in compatible_id)
        outcome = {"status":"model1-inference-confirmed" if success else "reload-blocked",
                   "artifact":job["artifact"], "parent":job["parent"], "handle":handle,
                   "checkpoint_id":job["checkpoint_id"], "receipt":terminal}
        self.ledger.store.commit({f"reload-result/{run}/{event}":outcome})
        return outcome
