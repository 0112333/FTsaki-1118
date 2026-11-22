"""Reopen actual ledgers with remote sends disabled; assert no logical changes."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytrio
from pytrio.config.config import configuration
from ftsaki_harness.model0_smoke import read, require
from ftsaki_harness.policy import canonical, digest
from ftsaki_harness.pytrio_training import PyTrioTraining
from ftsaki_harness.records import Records
from ftsaki_harness.training_intent import TrainingIntent

root=Path.cwd()
directory=root/"runs/stage23-C-model1-2026-11-19"
paths=list(directory.glob("*.sqlite"))+[root/"runs/2a-C-first-device-2026-11-19/review.sqlite"]
def snapshot():
    values={}
    for path in paths:
        store=Records(path)
        values[str(path.relative_to(root))]=digest(store.export())
        store.close()
    return values
before=snapshot()
original_argv=list(sys.argv)
try:
    with patch("ftsaki_harness.model0_smoke.post",side_effect=AssertionError("HTTP disabled")) as http, \
         patch("ftsaki_harness.live_acceptance.send_once",side_effect=AssertionError("New inference disabled")) as sender, \
         patch.object(pytrio,"ServiceClient",side_effect=AssertionError("Training service disabled")) as service:
        for phase in ("reload","replay"):
            sys.argv=["run_model1_acceptance_replay.py",phase]
            with redirect_stdout(io.StringIO()):
                runpy.run_path(str(root/"scripts/run_model1_acceptance_replay.py"),run_name="__main__")
        store=Records(directory/"training.sqlite")
        review=Records(root/"runs/2a-C-first-device-2026-11-19/review.sqlite")
        try:
            run=review.get("identity")["run"]
            event=read(root/"runs/remaining-tests-proposal-2026-11-19.json")["event"]
            adapter=PyTrioTraining(TrainingIntent(store,review),pytrio,configuration)
            try:
                adapter.submit(run,event,read(root/"runs/remaining-tests-token-cases-2026-11-19.json"))
            except ValueError as error:
                require("Submission already started" in str(error),"Unexpected duplicate guard")
            else:
                raise AssertionError("Duplicate training was not rejected")
        finally:
            store.close()
            review.close()
        counts={"http":http.call_count,"new_inference":sender.call_count,"training_service":service.call_count}
finally:
    sys.argv=original_argv
after=snapshot()
require(before==after and all(value==0 for value in counts.values()),"Recovery changed logical records or attempted remote calls")
result={"status":"passed","logical_snapshots_before":before,"logical_snapshots_after":after,
        "new_remote_calls":counts,"duplicate_training_submission":"rejected-before-service-client"}
path=directory/"recovery-audit.json"
if path.exists():
    require(read(path)==result,"Recovery audit changed")
else:
    with path.open("x") as handle:handle.write(canonical(result)+"\n")
print(canonical({"recovery":"passed","unchanged_databases":len(paths),"new_remote_calls":counts}))
