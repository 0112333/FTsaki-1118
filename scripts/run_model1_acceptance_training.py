"""Execute the single approved job, or reconcile metadata without resubmitting."""
import argparse
from pathlib import Path
import pytrio
from pytrio.config.config import configuration
from ftsaki_harness.model0_smoke import read, require
from ftsaki_harness.policy import canonical, digest
from ftsaki_harness.pytrio_training import PyTrioTraining
from ftsaki_harness.records import Records
from ftsaki_harness.training_intent import TrainingIntent

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--send',action='store_true')
args=parser.parse_args()
root=Path('runs')
proposal=read(root/'remaining-tests-proposal-2026-11-19.json')
grant=read(root/'remaining-tests-grant-2026-11-19.json')
require(grant['approval'] and grant['proposal_sha256']==digest(proposal) and grant['controller_experiment_approved'] is True, 'Approval differs from reviewed proposal')
event=proposal['event']
review=Records(root/'2a-C-first-device-2026-11-19/review.sqlite')
store=Records(root/'stage23-C-model1-2026-11-19/training.sqlite')
try:
    original=review.get('plan/'+event)
    require(digest(original)==proposal['original_review_sha256'],'Original review changed')
    supplement={'approval':grant['approval'],'purpose':'decision-connectivity-experiment','event':event,
                'original_plan_sha256':digest(original),'parent_model':proposal['parent']['model'],
                'scenario':'C','training_objects':['decision'],'grant_sha256':digest(grant)}
    review.commit({'decision-authorization/'+event:supplement})
    parent=dict(proposal['parent'],authorized=True,eligible=True,binding_approval=grant['approval'])
    method=dict(proposal['method'],approval=grant['approval'])
    snapshot=dict(proposal['snapshot'],authorization=grant['approval'])
    budget={key:value for key,value in proposal['training_budget'].items() if key!='deadline_seconds_after_approval'}
    budget.update(approval=grant['approval'],deadline=grant['training_deadline'])
    require(budget['max_jobs']==grant['max_jobs']==1 and budget['max_updates']==grant['max_updates']==2
            and budget['max_sequence_tokens']==grant['max_sequence_tokens']==28796,'Budget changed')
    ledger=TrainingIntent(store,review)
    ledger.reserve(event,parent,method,snapshot,{case['id']:digest(case) for case in snapshot['cases']},budget)
    run=review.get('identity')['run']
    adapter=PyTrioTraining(ledger,pytrio,configuration)
    if args.send:
        result=adapter.submit(run,event,read(root/'remaining-tests-token-cases-2026-11-19.json'))
    else:
        result=adapter.reconcile(run,event)
    print(canonical(result),flush=True)
finally:
    store.close()
    review.close()
