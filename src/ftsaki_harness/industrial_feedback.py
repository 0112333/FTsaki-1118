"""Development label worker. Only the child opens the closed-loop labels."""
import json
import multiprocessing
from pathlib import Path
import sqlite3

from .data import Handoff, LabelProvider
from .model0_smoke import require
from .policy import canonical


def worker(pipe, database, manifest, scenario):
    try:
        with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=rw", uri=True) as connection:
            def get(key):
                row = connection.execute("SELECT value FROM records WHERE key=?", (key,)).fetchone()
                return json.loads(row[0]) if row else None
            identity = get("identity")
            require(identity and identity["mode"] == "industrial-paired" and identity["scenario"] == scenario,
                    "Unbound industrial ledger")
            handoff = Handoff(manifest)
            require(identity["handoff_sha256"] == handoff.sha256, "Label source not bound to experiment")
            labels = LabelProvider(handoff.descriptor(handoff.scenario(scenario)["splits"]["closed_loop"]["labels"]))
            pipe.send(("ready", None))
            while True:
                first_id = pipe.recv()
                if first_id is None:
                    return
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    first = get("pair/" + first_id)
                    require(first is not None, "No paired prediction")
                    start = str(first["step"])
                    batch = get("batch/" + start)
                    require(batch and batch["ids"][0] == first_id, "Not a saved batch boundary")
                    release_key = "release/" + start
                    answers = get(release_key)
                    if answers is None:
                        spent = sum(len(json.loads(row[0])) for row in connection.execute(
                            "SELECT value FROM records WHERE key LIKE 'release/%'"))
                        saved = [get("pair/" + sample_id) for sample_id in batch["ids"]]
                        require(all(item and item["group"] == first["group"] and "stable" in item and "candidate" in item
                                    for item in saved), "Batch predictions missing")
                        arrived = [{"id": item["id"], "group_id": item["group"]} for item in saved]
                        answers = labels.release(arrived, set(batch["ids"]), scenario, identity["label_budget"] - spent)
                        connection.execute("INSERT INTO records VALUES (?,?)", (release_key, canonical(answers)))
                pipe.send(("ok", answers))
    except (EOFError, BrokenPipeError):
        return
    except Exception as exc:
        pipe.send(("error", type(exc).__name__))
    finally:
        pipe.close()


class IndustrialFeedback:
    """Local development separation; file ACLs remain a deployment responsibility."""
    def __init__(self, database, manifest, scenario):
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, database, manifest, scenario), daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive("ready")
        except Exception:
            self.close()
            raise

    def _receive(self, expected):
        if not self.pipe.poll(30):
            raise TimeoutError("Industrial label worker timed out")
        state, value = self.pipe.recv()
        if state != expected:
            raise ValueError("Industrial label worker refused request: " + str(value))
        return value

    def release(self, batch, predicted_ids, scenario, remaining):
        self.pipe.send(batch[0]["id"])
        return self._receive("ok")

    def close(self):
        try:
            if self.process.is_alive():
                self.pipe.send(None)
        except (EOFError, OSError):
            pass
        self.pipe.close()
        self.process.join(3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
