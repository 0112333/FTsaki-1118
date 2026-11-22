import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .data import Handoff, Replay, Windows, flatten
from .policy import Policy, DEPENDENCIES, canonical
from .records import Records

def inspect_replay(args):
    handoff = Handoff(args.manifest)
    policy = Policy.load(args.config) if args.config else Policy()
    spec = handoff.scenario(args.scenario)
    path = handoff.descriptor(spec["splits"]["closed_loop"]["observations"])
    records = Records(args.database)
    try:
        identity = {"manifest_sha256": handoff.sha256, "scenario": args.scenario,
                    "policy": asdict(policy), "policy_sha256": policy.sha256,
                    "mode": "data-only", "model_inference": False, "labels_released": 0}
        records.commit({"identity": identity})
        prior = records.export()
        completed = sorted((v for k,v in prior.items() if k.startswith("batch/")), key=lambda x: x["index"])
        cursor = sum(x["source_count"] for x in completed)
        windows = Windows()
        replay = Replay(path, policy.batch_size(args.scenario))
        keys = None
        seen = set()
        for batch in replay.batches():
            group = batch[0]["group_id"]
            valid, errors = [], []
            for row in batch:
                try:
                    order, vector = flatten(row["inputs"])
                    if len(vector) != spec["input_features"] or (keys is not None and keys != order):
                        raise ValueError("Feature schema mismatch")
                    if row["id"] in seen:
                        raise ValueError("Duplicate ID")
                    keys = order
                    seen.add(row["id"])
                    # No source metadata or labels are forwarded to consumers.
                    valid.append({"id": row["id"], "features": vector})
                except (ValueError, KeyError) as exc:
                    errors.append({"id": row.get("id"), "reason": str(exc)})
            windows.add(group, valid)
            next_cursor = replay.cursor + len(batch)
            index = next_cursor - len(batch)
            record = {"index": index, "group": group, "source_count": len(batch),
                      "valid_ids": [r["id"] for r in valid], "errors": errors,
                      "window_batches": windows.context()["batches"]}
            # Re-scan committed prefix to rebuild windows without exposing it as new evidence.
            if next_cursor > cursor:
                records.commit({f"batch/{index:09d}": record})
            replay.commit(len(batch))
        summary = {"scenario": args.scenario, "cursor": replay.cursor,
                   "batches": sum(k.startswith("batch/") for k in records.export()),
                   "mode": "data-only", "model_inference": False,
                   "labels_released": 0, "training_submitted": False,
                   "note": "Data-path verification only; no drift or adaptation result"}
        records.commit({"end": summary})
        return summary
    finally:
        records.close()

def main():
    parser = argparse.ArgumentParser(description="FTsaki Python harness development tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dependencies", help="Show unresolved real-service dependencies")
    check = sub.add_parser("verify-data", help="Verify manifest hashes, roles, dimensions and rows")
    check.add_argument("--manifest", required=True)
    check.add_argument("--scenario", choices=list("ABC"), required=True)
    replay = sub.add_parser("replay", help="Data-only grouped replay; no model or label access")
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--scenario", choices=list("ABC"), required=True)
    replay.add_argument("--database", required=True)
    replay.add_argument("--config")
    export = sub.add_parser("export")
    export.add_argument("--database", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "dependencies":
            result = DEPENDENCIES
        elif args.command == "verify-data":
            result = Handoff(args.manifest).verify(args.scenario)
        elif args.command == "replay":
            result = inspect_replay(args)
        else:
            store = Records(args.database)
            try:
                result = store.export()
            finally:
                store.close()
            with Path(args.output).open("x", encoding="utf-8") as handle:
                handle.write(canonical(result) + "\n")
            result = {"exported": args.output}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, str(exc) + "\n")

if __name__ == "__main__":
    main()
