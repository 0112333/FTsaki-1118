"""Run one bounded acceptance of saved LoRA artifacts; never update weights."""
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytrio
import swanlab
from pytrio.config.config import configuration
from transformers import AutoTokenizer

from evaluate_model0_baseline import aggregate
from model0_evaluation import score, validate_output_policy
from model0_messages import ROOT, require_accepted
from model0_response import response_text
from model0_runtime import runtime_messages

BASE = ROOT / 'artifacts/preflight/a_forecast_v2'
TRIAL = ROOT / 'artifacts/training/trial_messages_v3'
OUT = TRIAL / 'acceptance'
DATA = ROOT.parent / 'data/model0_episodes_v1_1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def prepare():
    dest = read(ROOT / 'configs/training-destinations.json')
    check(sys.prefix == str(Path(dest['python']).parent.parent), 'Wrong virtual environment')
    accepted = require_accepted(DATA, ROOT / 'artifacts/acceptance/model0_episodes_v1_1/acceptance.json')
    hashes = {file['path']: file['sha256'] for file in read(DATA / 'manifest.json')['files']}
    for name in ('train.jsonl','dev.jsonl'):
        check(digest(DATA/name) == hashes[name], 'Split changed: ' + name)
    trial, plan = read(TRIAL/'result.json'), read(TRIAL/'run-plan.json')
    base, receipt = read(BASE/'baseline/result.json'), read(BASE/'baseline/acceptance.json')
    protocol, probe = read(BASE/'baseline/started.json'), read(BASE/'probe-plan.json')
    frozen, ready = read(BASE/'frozen-contract.json'), read(TRIAL/'readiness.json')
    check(digest(BASE/'frozen-contract.json') == plan['contract_sha256'] == protocol['contract_sha256'] == probe['contract_sha256'] == receipt['contract_sha256'], 'Protocol mismatch')
    check(frozen['manifest_sha256'] == accepted['dataset_manifest_sha256'] == receipt['manifest_sha256'], 'Data provenance')
    check(ready['passed'] and receipt['accepted'] and receipt['swanlab_upload_verified'] and digest(BASE/'baseline/result.json') == receipt['result_sha256'] and digest(BASE/'baseline/acceptance.json') == ready['baseline_acceptance_sha256'], 'Baseline receipt')
    for files in (plan['execution_files'], frozen['files'], protocol['execution_files']):
        for name, expected in files.items():
            check(digest(ROOT/name) == expected, 'Code or tokenizer changed: ' + name)
    forecast_hashes = {r['harness_forecast_sha256'] for r in base['rows'] if 'harness_forecast_sha256' in r}
    check(forecast_hashes == {digest(ROOT.parent/'FTsaki_harness/src/ftsaki_harness/forecast.py')}, 'Harness scorer changed')
    updates, cases = trial['updates'], plan['episodes']
    check(trial['status'] == 'trained_reload_and_dev_pending' and len(updates) == trial['forward_backward_calls'] == plan['max_optim_steps'] == 16, 'Training steps')
    check([r['step'] for r in updates] == list(range(1,17)) and [r['episode_id'] for r in updates] == [c['episode_id'] for c in cases], 'Training ledger')
    check(sum(c['sequence_tokens'] for c in cases) == trial['sequence_tokens_submitted'] == plan['max_sequence_tokens'], 'Training token ledger')
    check(all(all(math.isfinite(v) for v in (*r['metrics'].values(), *r['optim']['metrics'].values())) for r in updates), 'Nonfinite metric')
    with (DATA/'dev.jsonl').open() as stream:
        dev = {ep['episode_id']:ep for ep in map(json.loads, stream)}
    ids = {c['episode_id'] for c in cases}
    with (DATA/'train.jsonl').open() as stream:
        selected = {ep['episode_id']:ep for ep in map(json.loads, stream) if ep['episode_id'] in ids}
    check(len(ids) == 16 and ids == set(selected) and not ids.intersection(dev) and all(ep['split'] == 'train' and ep['audit']['eligibility']['sft_eligible'] for ep in selected.values()), 'Train split/unique IDs')
    check(base['status'] == 'complete' and len(protocol['cases']) == len(base['rows']) == len(dev) == 244 and set(dev) == {c['episode_id'] for c in protocol['cases']} == {r['episode_id'] for r in base['rows']}, 'Dev coverage')
    a_case, = [c for c in probe['cases'] if c['scenario']=='A' and c['task']=='forecast_next' and c['split']=='train']
    check(a_case['episode_id'] in selected, 'A probe lineage')
    check(plan['workspace'] == dest['pytrio_workspace_id'] == protocol['workspace'] and plan['base_model'] == dest['model'] == protocol['model'], 'Destination mismatch')
    check(sum(c['prompt_tokens']+c['max_output_tokens'] for c in protocol['cases']) == protocol['max_billable_text_tokens'] and protocol['concurrency'] == 4, 'Dev budget')
    return dest, trial, plan, base, protocol, a_case, dev | selected


def compare(base_rows, rows):
    groups, changes = [], []
    for base, trial in zip(aggregate(base_rows), aggregate(rows), strict=True):
        check((base['scenario'], base['task'], base['count']) == (trial['scenario'], trial['task'], trial['count']), 'Pairing')
        metrics = ('macro_f1', 'drift_false_positive_rate') if base['task'] == 'decide' else ('gate_passed',)
        delta = {'format_rate': trial['format_rate'] - base['format_rate']}
        for key in metrics:
            if base[key] is not None and trial[key] is not None:
                delta[key] = trial[key] - base[key]
                changes.append(-delta[key] if key == 'drift_false_positive_rate' else delta[key])
        groups.append({'scenario':base['scenario'], 'task':base['task'], 'delta':delta, 'baseline':base, 'trial':trial})
    return groups, min(changes) >= 0 and max(changes) > 0 and all(g['delta']['format_rate'] >= 0 for g in groups)


def evaluation_row(episode, sequence, tokenizer):
    text = response_text(sequence, tokenizer)
    probe = episode['split'] == 'train'
    row = {'episode_id':episode['episode_id'], 'scenario':episode['scenario_id'], 'task':episode['task'], 'truth':episode['target'].get('decision'), 'stop_reason':sequence['stop_reason']}
    try:
        row.update(score(episode,text))
        if probe:
            check(sequence['stop_reason'] == 'stop','Abnormal A completion')
            validate_output_policy(episode,text)
            def decimal_literal(value):
                check('e' not in value.lower() and not value.endswith('0') and len(value.split('.')[1]) <= 6, 'A decimal representation')
                return float(value)
            json.loads(text, parse_float=decimal_literal)
    except (ValueError,KeyError,TypeError,ArithmeticError) as error:
        row.update(valid=False,prediction='invalid',failure_type=type(error).__name__)
        if probe:
            row['failure'] = str(error)[:160]
    if probe:
        row['independent_dev_metric'] = False
        try:
            matrix = json.loads(text)['values']
            if isinstance(matrix,list):
                row.update(observed_rows=len(matrix), observed_widths=sorted({len(r) for r in matrix if isinstance(r,list)}))
        except (ValueError,KeyError,TypeError):
            pass
    return row


def main():
    dest, trial, plan, base, protocol, a_case, episodes = prepare()
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'artifacts/preflight/tokenizer', local_files_only=True)
    limits = {'training_run_id':trial['training_run_id'], 'contract_sha256':plan['contract_sha256'], 'model_path':trial['sampler_weights']['path'], 'resume_path':trial['updates'][-1]['checkpoint']['path'], 'trial_result_sha256':digest(TRIAL/'result.json'), 'dev_plan_sha256':digest(BASE/'baseline/started.json'), 'script_sha256':digest(Path(__file__)), 'dev_max_calls':244, 'dev_max_text_tokens':protocol['max_billable_text_tokens'], 'a_max_calls':1, 'a_max_text_tokens':a_case['prompt_tokens'] + a_case['max_output_tokens'], 'max_steps':0, 'max_optimizer_restores':1, 'max_samplers':1, 'max_concurrency':4, 'attempts_per_case':1, 'deadline_seconds':protocol['deadline_seconds'] + 1800, 'funding':'Existing authorized workspace quota; currency price and balance unverified. No retries.'}
    if '--check-only' in sys.argv:
        print(json.dumps(limits)); return
    OUT.mkdir(exist_ok=True)
    restored = None
    if '--continue-after-restore' in sys.argv:
        previous = read(OUT/'started.json')
        check(all(limits[k] == previous[k] for k in limits if k != 'script_sha256'), 'Resume limits or lineage changed')
        restored = read(OUT/'result.json')
        check(restored['status']=='blocked' and restored.get('optimizer_restored') and restored.get('error_type')=='AttributeError' and restored['calls_reserved']==0 and 'sampling_run_id' not in restored and restored.get('upload_verified'), 'Ambiguous previous actions; cannot continue')
        with (OUT/'restore-receipt.json').open('x') as stream:
            json.dump(restored, stream, indent=2)
        with (OUT/'sampling-started.json').open('x') as stream:
            json.dump(limits,stream,indent=2)
    else:
        with (OUT/'started.json').open('x') as stream:
            json.dump(limits,stream,indent=2)
    deadline = time.monotonic() + limits['deadline_seconds']
    state = {'status':'verifying', 'started_at':datetime.now(timezone.utc).isoformat(), 'training_run_id':trial['training_run_id'], 'dev_completed':0, 'dev_total':244, 'calls_reserved':0, 'text_tokens_reserved':0, 'input_tokens':0, 'output_tokens':0, 'new_optimizer_steps':0, 'artifacts_accepted':False, 'currency_cost':None, 'formal_training_started':False}
    if restored:
        elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(restored['started_at'])).total_seconds()
        deadline -= max(0,elapsed)
        state.update(started_at=restored['started_at'], restore_run_id=restored['restore_run_id'], optimizer_restored=True, restore_swanlab_id=restored['swanlab_run_id'])
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        dump(OUT/'result.json',state)
    def left():
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Acceptance deadline')
        return remaining
    save()
    run = None
    try:
        configuration.retry_max_attempts = 1
        configuration.training_retry_timeout_seconds = configuration.control_retry_timeout_seconds = configuration.sampling_retry_timeout_seconds = 0
        configuration.actor_capacity_wait_timeout_seconds = 120
        configuration.sampling_timeout_seconds = protocol['deadline_seconds']
        client = pytrio.ServiceClient()
        check(client.workspace_cuid == dest['pytrio_workspace_id'], 'Wrong workspace')
        rest = client.create_rest_client()
        rid = trial['training_run_id']
        remote = rest.get_training_run(rid).result(timeout=min(left(),90))
        check(remote.training_run_id == rid and remote.base_model == plan['base_model'] and remote.is_lora and remote.lora_rank == plan['rank'] and not remote.corrupted, 'Remote LoRA lineage')
        inventory = {item.path:item for item in rest.list_checkpoints(rid).result(timeout=min(left(),90)).checkpoints}
        entries = [trial['initial_checkpoint'], *[r['checkpoint'] for r in trial['updates']], trial['sampler_weights']]
        check(len({e['path'] for e in entries}) == 18, 'Duplicate checkpoint')
        for entry in entries:
            item = inventory[entry['path']]
            check(entry['path'].startswith('trio://'+rid+'/') and item.size_bytes == entry['size'] > 0 and item.checkpoint_type == ('sampler' if entry['type']=='save-weights' else 'training') and item.expires_at is None, 'Remote checkpoint mismatch')
        project = dest['swanlab_workspace'] + '/' + dest['swanlab_project'] + '/'
        cloud = swanlab.Api().run(project + trial['swanlab_run_id'])
        check(cloud.name == 'trial-messages-v3' and cloud.state == 'FINISHED','Trial log mismatch')
        state.update(trial_steps=16, trial_sequence_tokens=trial['sequence_tokens_submitted'], remote_artifacts=len(entries), trial_swanlab_state=cloud.state, trial_swanlab_id=trial['swanlab_run_id'])
        save()
        run = swanlab.init(workspace=dest['swanlab_workspace'], project=dest['swanlab_project'], name='trial-acceptance-messages-v3', public=False, job_type='evaluation', mode='online', on_online_error='raise', log_dir=str(ROOT/'artifacts/swanlog'), settings=swanlab.Settings(interactive=False,probe={'hardware':False,'runtime':False,'requirements':False,'git':False,'monitor':False}), config=limits)
        state.update(swanlab_run_id=run.id, status='creating_sampler' if restored else 'restoring_optimizer')
        save()
        if not restored:
            trainer = client.create_training_client_from_state_with_optimizer(limits['resume_path'])
            state['restore_run_id'] = trainer.training_run_cuid
            save()
            trainer.load_state_with_optimizer(limits['resume_path']).result(timeout=min(left(),600))
            state['optimizer_restored'] = True
            save()
        sampler = client.create_sampling_client(base_model=plan['base_model'], model_path=limits['model_path'])
        state.update(sampling_run_id=sampler.sampling_run_cuid, status='sampling')
        save()
        def sample(case):
            ep = episodes[case['episode_id']]
            ids = tokenizer.apply_chat_template(runtime_messages(ep['input'],ep['task']), tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if hasattr(ids,'keys'):
                ids = ids['input_ids']
            check(len(ids)==case['prompt_tokens'] and len(ids)+case['max_output_tokens']<=65536,'Request capacity')
            response = sampler.sample(pytrio.ModelInput.from_ints(ids), num_samples=1, sampling_params=pytrio.SamplingParams(max_tokens=case['max_output_tokens'], temperature=protocol['temperature'], seed=protocol['seed'])).result(timeout=left()).model_dump(mode='json')
            dump(OUT/(case['episode_id']+'.json'), response)
            row = evaluation_row(ep,response['sequences'][0],tokenizer)
            row.update(input_tokens=response['input_tokens'],output_tokens=response['output_tokens'])
            return row

        rows = []
        cases = [a_case, *protocol['cases']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0,len(cases),4):
                left()
                batch = cases[start:start+4]
                state['calls_reserved'] += len(batch)
                state['text_tokens_reserved'] += sum(c['prompt_tokens']+c['max_output_tokens'] for c in batch)
                check(state['calls_reserved'] <= 245 and state['text_tokens_reserved'] <= limits['a_max_text_tokens']+limits['dev_max_text_tokens'],'Budget exceeded')
                save()
                futures = {pool.submit(sample,c):c for c in batch}
                errors = []
                for future in concurrent.futures.as_completed(futures):
                    case = futures[future]
                    try:
                        row = future.result()
                    except Exception as error:
                        errors.append(error); continue
                    state['input_tokens'] += row['input_tokens']
                    state['output_tokens'] += row['output_tokens']
                    state['artifacts_accepted'] = state['optimizer_restored']
                    if case['split']=='train':
                        state['a_probe'] = row
                    else:
                        rows.append(row)
                        state['dev_completed'] = len(rows)
                        dump(OUT/'dev-rows.json',rows)
                    save()
                    swanlab.log({'acceptance/dev_completed':len(rows),'acceptance/artifacts_loaded':1,**({'acceptance/a_complete':int(row['valid'])} if case['split']=='train' else {})})
                if errors:
                    raise errors[0]
        check(len(rows)==244 and {r['episode_id'] for r in rows}=={r['episode_id'] for r in base['rows']},'Incomplete pairs')
        groups, quality = compare(base['rows'],rows)
        report = {'groups':groups,'quality_criteria_met':quality,'baseline_valid':sum(r['valid'] for r in base['rows']), 'trial_valid':sum(r['valid'] for r in rows), 'dev_ids_verified':True, 'dev_cost_tokens':{k:sum(r[k] for r in rows) for k in ('input_tokens','output_tokens')}, 'baseline_cost_tokens':{k:sum(r[k] for r in base['rows']) for k in ('input_tokens','output_tokens')},'currency_cost':None, 'coverage':base['coverage'], 'a_probe':state['a_probe']}
        dump(OUT/'comparison.json',report)
        state.update(status='evaluation_complete',quality_criteria_met=quality,baseline_valid=report['baseline_valid'],trial_valid=report['trial_valid'])
        swanlab.log({'acceptance/dev_total':len(rows),'acceptance/dev_valid':report['trial_valid'],'acceptance/base_valid':report['baseline_valid'],'acceptance/quality_passed':int(quality),'acceptance/input_tokens':state['input_tokens'],'acceptance/output_tokens':state['output_tokens']})
    except Exception as error:
        state.update(status='blocked',error_type=type(error).__name__,error_code=getattr(error,'code',None))
        if isinstance(error,(ValueError,TimeoutError,KeyError)):
            state['diagnosis'] = str(error)[:180]
        raise
    finally:
        save()
        if run is not None:
            swanlab.finish()
            cloud = swanlab.Api().run(project+run.id)
            state['upload_verified'] = cloud.name=='trial-acceptance-messages-v3' and cloud.state=='FINISHED'
            if state['status']=='evaluation_complete' and state['upload_verified']:
                state['formal_quality_gate_passed'] = bool(state['quality_criteria_met'] and state['a_probe']['valid'] and state['artifacts_accepted'])
            save()


if __name__ == '__main__':
    main()
