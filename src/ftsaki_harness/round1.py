"""Run or inspect the frozen first-round model0 acceptance grid."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from .model0_smoke import MODEL, TASKS, grid_report, plan_cell, prepare, read, record_checkpoint_match, require, send_once
from .policy import canonical, digest
from .records import Records
from .training_intent import timestamp


FIRST = "C/decide"


def cell_database(directory, cell):
    name = "model0-text-smoke-01" if cell == FIRST else "round1-grid-" + cell.replace("/", "-")
    return directory / (name + ".sqlite")


def stored_grid(directory):
    cells = {}
    for scenario in "ABC":
        for task in TASKS:
            cell = f"{scenario}/{task}"
            path = cell_database(directory, cell)
            item = {"status": "not-run", "database": path.name}
            if path.is_file():
                store = Records(path)
                try:
                    state = grid_report(store)
                    item.update(status=state["grid"][cell], reservations=state["reservations"], result=state["result"],
                                checkpoint_match=state["checkpoint_match"])
                finally:
                    store.close()
            cells[cell] = item
    match = cells[FIRST].get("checkpoint_match")
    if cells[FIRST]["status"] == "passed" and match and match["path"] == MODEL:
        for cell, item in cells.items():
            if cell == FIRST or item["status"] not in ("invalid-output", "unverified-model-echo"):
                continue
            result = item["result"]
            if result.get("provider_model") not in (MODEL, match["provider_model"]):
                item["status"] = "unverified-model-identity"
            elif item["status"] == "unverified-model-echo" and result.get("format_valid") is True:
                item.update(status="passed", checkpoint_match=dict(match, response_sha256=result["response_sha256"]))
    return cells


def approved_order(plan, grant):
    order = grant.get("cells")
    require(grant.get("approval") and grant.get("plan_sha256") == digest(plan), "Confirm this exact round-one plan")
    require(isinstance(order, list) and order and len(set(order)) == len(order)
            and all(cell in plan and cell != FIRST for cell in order), "Invalid or repeated authorized cell")
    continuation = grant.get("continue_after_format_failure", [])
    require(isinstance(continuation, list) and set(continuation) <= set(order), "Invalid continuation scope")
    budget = sum(plan[cell]["max_prompt_tokens"] + plan[cell]["max_output_tokens"] for cell in order)
    require(type(grant.get("max_tokens")) is int and budget <= grant["max_tokens"], "Authorized grid exceeds token budget")
    return order


def next_cell(cells, order, continuation):
    format_failure = False
    for cell in order:
        status = cells[cell]["status"]
        if status == "passed":
            continue
        if status == "invalid-output":
            format_failure = True
            continue
        if status == "not-run":
            if format_failure and cell not in continuation:
                return None, "format-failure-continuation-not-approved"
            return cell, None
        return None, status
    return None, "requested-cells-recorded"


def run_grid(directory, base, plan, grant, training, dataset, key):
    require(bool(key), "Missing PyTRiO credentials")
    order = approved_order(plan, grant)
    end = timestamp(grant["deadline"])
    ledger = Records(directory / "round1-authorization.sqlite")
    try:
        ledger.commit({"authorization":grant, "plan_sha256":digest(plan), "base_sha256":digest(base)})
        while True:
            cells = stored_grid(directory)
            first = cells[FIRST]
            mapping = first.get("checkpoint_match")
            require(first["status"] == "passed" and mapping and mapping["path"] == MODEL, "Verified model0 mapping missing")
            cell, reason = next_cell(cells, order, grant.get("continue_after_format_failure", []))
            if reason:
                return {"cells":cells, "stop_reason":reason}
            config = plan_cell(base, plan, cell)
            if (end - datetime.now(timezone.utc)).total_seconds() <= config["deadline_seconds"]:
                return {"cells":cells, "stop_reason":"deadline"}
            messages, preflight, prompt = prepare(config, training, dataset)
            if (end - datetime.now(timezone.utc)).total_seconds() <= config["deadline_seconds"]:
                return {"cells":cells, "stop_reason":"deadline"}
            store = Records(cell_database(directory, cell))
            try:
                require(store.get("identity") == config and store.get("offline/preflight") == preflight,
                        "Missing or changed offline cell preflight")
                result = send_once(store, config, messages, preflight, key, prompt=prompt)
                if result["status"] == "unverified-model-echo" and result.get("provider_model") == mapping["provider_model"]:
                    record_checkpoint_match(store, {"checkpoint_id":mapping["checkpoint_id"], "checkpoint_type":"sampler",
                                                    "path":mapping["path"]})
                if result["status"] == "invalid-output" and result.get("provider_model") not in (MODEL, mapping["provider_model"]):
                    return {"cells":stored_grid(directory), "stop_reason":"unverified-model-identity"}
            finally:
                store.close()
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    for name in ("config", "plan", "authorization", "training", "dataset"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.authorization:
        require(all((args.config, args.plan, args.training, args.dataset)), "Missing live-run paths")
        base = read(args.config)
        report = run_grid(args.directory, base, read(args.plan), read(args.authorization), args.training.resolve(),
                          args.dataset, os.environ.get(base["credential_env"]))
    else:
        report = {"cells":stored_grid(args.directory), "stop_reason":"inspection"}
    print(canonical(report))


if __name__ == "__main__":
    main()
