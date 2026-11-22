"""Finalize the local audit using saved responses only; no cloud client or training."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/training/trial_messages_v3/acceptance'
BASE = ROOT / 'artifacts/preflight/a_forecast_v2/baseline'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, text):
    temp = path.with_suffix('.new')
    temp.write_text(text)
    temp.replace(path)


def finalize():
    state = read(OUT/'result.json')
    if state['status'] == 'blocked':
        return {'decision':'evaluation_blocked', 'formal_training_eligible':False, 'error_type':state.get('error_type'), 'error_code':state.get('error_code'), 'dev_completed':state['dev_completed']}
    if state['status'] != 'evaluation_complete' or not state.get('upload_verified'):
        return None
    comparison = read(OUT/'comparison.json')
    rows = read(OUT/'dev-rows.json')
    baseline = read(BASE/'result.json')
    ids = {r['episode_id'] for r in baseline['rows']}
    if not (state['dev_completed'] == len(ids) == len(rows) == 244 and ids == {r['episode_id'] for r in rows} and state['a_probe']['episode_id'] not in ids and read(OUT/'baseline-replay.json')['raw_metrics_match']):
        raise ValueError('Dev evidence incomplete')
    token_keys = ('input_tokens','output_tokens')
    for row in [*rows, state['a_probe']]:
        raw = read(OUT/(row['episode_id']+'.json'))
        if not raw['sequences'] or any(raw[key]!=row[key] for key in token_keys):
            raise ValueError('Raw result mismatch: '+row['episode_id'])
    if any(state[key]!=sum(r[key] for r in rows)+state['a_probe'][key] for key in token_keys):
        raise ValueError('Usage ledger mismatch')
    reasons=[]
    if not state['artifacts_accepted']:
        reasons.append('artifact_reload')
    if not state['a_probe']['valid']:
        reasons.append('a_full_output')
    if not comparison['quality_criteria_met']:
        reasons.append('paired_quality')
    if state['formal_quality_gate_passed'] != (not reasons):
        raise ValueError('Gate status mismatch')
    return {'decision':'trial_quality_gates_passed' if not reasons else 'not_accepted_for_formal_training', 'formal_training_eligible':not reasons, 'failed_gates':reasons, 'dev_completed':244, 'baseline_valid':comparison['baseline_valid'], 'trial_valid':comparison['trial_valid'], 'a_probe':state['a_probe'], 'dev_tokens':comparison['dev_cost_tokens'], 'actual_currency_cost':None, 'formal_run_started':False, 'formal_budget_frozen':False,
            'evidence_sha256':{path.name:digest(path) for path in (OUT/'result.json', OUT/'comparison.json',OUT/'dev-rows.json',OUT/'baseline-replay.json')}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    start = datetime.fromisoformat(read(OUT/'result.json')['started_at']).timestamp()
    end = start + read(OUT/'started.json')['deadline_seconds'] + 300
    while True:
        result = finalize()
        if result is not None:
            break
        if not args.wait:
            print('Evaluation pending'); return
        if time.time() >= end:
            result = {'decision':'incomplete_at_deadline', 'formal_training_eligible':False,'dev_completed':read(OUT/'result.json')['dev_completed']}; break
        time.sleep(30)
    result['reviewed_at']=datetime.now(timezone.utc).isoformat()
    write(OUT/'verdict.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if result['decision'] in ('trial_quality_gates_passed','not_accepted_for_formal_training'):
        tasks=ROOT/'openspec/changes/train-model0-with-pytrio/tasks.md'
        text=tasks.read_text()
        if '- [ ] 4.4 ' in text:
            write(tasks,text.replace('- [ ] 4.4 ','- [x] 4.4 ',1))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
