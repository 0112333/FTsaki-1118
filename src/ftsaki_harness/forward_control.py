"""Offline persistence control on the first real B/C batches; never loads a model0 client."""
import argparse
from itertools import islice
from pathlib import Path

from .data import Handoff, Replay, flatten
from .forecast import Scale
from .policy import Policy, canonical
from .prospective import ProspectiveRun, require
from .records import Records


def run_control(manifest, scenario, database):
    require(scenario in ("B", "C"), "Select regression scenario")
    handoff = Handoff(manifest)
    spec = handoff.scenario(scenario)
    order, training = handoff.training_features(scenario)
    scale = Scale.fit(training)
    batch_size = Policy().batch_size(scenario)
    observations = handoff.descriptor(spec["splits"]["closed_loop"]["observations"])
    store = Records(database)
    try:
        name = "first-window-persistence-" + scenario
        run = ProspectiveRun(store, run=name, scenario=scenario, source="controlled", manifest=handoff.sha256,
                             feature_order=order, scale=scale, start=0, parent_artifact="controlled-stable")
        def freeze_at_first_boundary():
            group = run.state["group"]
            receipt = {"source":"controlled", "artifact":"persistence-control", "parent_artifact":"controlled-stable",
                       "parent":0, "version":1, "run":name, "scenario":scenario, "event":"controlled-event",
                       "ready_group":group, "trained_through":0, "ready_at":batch_size,
                       "snapshot_sha256":"controlled-no-training"}
            run.freeze(receipt, [run.state["history"][-1]] * 32, order)
        if run.cursor == batch_size and not store.get("forecast/1"):
            freeze_at_first_boundary()
        source = Replay(observations, batch_size, cursor=run.cursor)
        batches_left = 1 + 32 // batch_size - run.cursor // batch_size
        for batch in islice(source.batches(), max(0, batches_left)):
            samples = []
            for observation in batch:
                keys, values = flatten(observation["inputs"])
                require(keys == order, "Source feature schema changed")
                samples.append({"row": values, "order": observation["replay_order"],
                                "sample_id": observation["id"], "group": observation["group_id"]})
            run.advance(samples, {0:"insufficient"})
            source.commit(len(batch))
            if run.cursor == batch_size:
                freeze_at_first_boundary()
        run.end()
        return {"source":"controlled", "input_source":"closed_loop", "scenario":scenario,
                "start":batch_size + 1, "last":batch_size + 32, "cursor":run.cursor,
                "candidate":store.get("finish/1"), "remote_calls":0}
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scenario", required=True, choices=("B", "C"))
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()
    print(canonical(run_control(args.manifest, args.scenario, args.database)))
