"""Offline review only: no service client, no labels, no observations after 96."""
import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import AutoTokenizer
import pytrio
from ftsaki_harness.data import Handoff, read_jsonl
from ftsaki_harness.decision_replay import render
from ftsaki_harness.forecast import Scale
from ftsaki_harness.inference import validate_content
from ftsaki_harness.model0_smoke import read
from ftsaki_harness.policy import canonical, digest
from ftsaki_harness.records import Records
from ftsaki_harness.replay_handoff import reviewed_anchor
from ftsaki_harness.window_evidence import load_runtime, window_stream

root = Path.cwd()
training = (root.parent / 'ModelTraining').resolve()
data = (root.parent / 'data').resolve()
base = read(root / 'config/model0-smoke.json')
prior = root / 'runs/2a-C-first-device-2026-11-19'
review = Records(prior / 'review.sqlite')
event = 'event-c11ee4f12c93c28261f8'
plan = review.get('plan/' + event)
producer, context, replay, builder, manifest = load_runtime(data, training, 'C', base['contract_sha256'])
handoff = Handoff(data / 'harness_data_v1/manifest.json')
source = handoff.descriptor(handoff.scenario('C')['splits']['closed_loop']['observations'])
# islice bounds the reader before constructing any future target.
observations = list(islice(read_jsonl(source), 96))
assert len(observations) == 96 and len({row['group_id'] for row in observations}) == 1
anchor = reviewed_anchor(review, iter(observations), manifest=manifest, scenario='C',
                         feature_order=context['order'], run='stage3-C-model1-2026-11-19')
tokenizer = AutoTokenizer.from_pretrained(training / 'artifacts/preflight/tokenizer', local_files_only=True)
def tokens(messages, generation):
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, enable_thinking=False)
    return result['input_ids'] if hasattr(result, 'keys') else result

cases, samples = [], {}
with TemporaryDirectory() as temporary:
    local = Records(Path(temporary) / 'evidence.sqlite')
    for record, evidence, messages in islice(window_stream(local, producer, context, replay, builder), 3):
        cutoff = evidence['as_of']['replay_order']
        assert cutoff <= 96
        vectors = [producer.vector(row) for row in observations[:cutoff]]
        forecast_evidence = dict(evidence, recent_history={'group_steps':list(range(max(1,cutoff-31), cutoff+1)), 'values':vectors[-32:]})
        forecast_messages = builder(forecast_evidence, 'forecast_next')
        if cutoff < 96:
            target = {'horizon':32,'feature_order':context['order'], 'values':[producer.vector(row) for row in observations[cutoff:cutoff+32]]}
            validate_content('forecast_next', forecast_evidence, canonical(target), 'stop')
            prompt_ids = tokens(forecast_messages, True)
            complete = tokens(forecast_messages + [{'role':'assistant','content':canonical(target)}], False)
            assert complete[:len(prompt_ids)] == prompt_ids
            answer_ids = complete[len(prompt_ids):]
            case = {'id':f'C-closed-loop-{cutoff}-forecast-next','scenario':'C','task':'forecast_next',
                    'supervision_source':'observed_future_features','input_cutoff':cutoff,'supervision_cutoff':cutoff+32,
                    'input_sha256':digest(prompt_ids),'target_sha256':digest(answer_ids), 'sequence_tokens':len(complete)}
            reference, = [item for item in read(prior / 'supervision-preflight.json')['cases'] if item['input_cutoff'] == cutoff]
            assert all(case[key] == reference[key] for key in ('input_sha256','target_sha256','sequence_tokens'))
            datum = pytrio.Datum(model_input=pytrio.ModelInput.from_ints(complete[:-1]),
                loss_fn_inputs={'target_tokens':complete[1:],'weights':[0.]*(len(prompt_ids)-1)+[1./len(answer_ids)]*len(answer_ids)})
            cases.append(case)
            samples[case['id']] = {'prompt_ids':prompt_ids,'answer_ids':answer_ids}
        else:
            _, reload_prompt = render(tokenizer, messages)
            _, forecast_prompt = render(tokenizer, forecast_messages)
    local.close()
old_plan = review.get('replay-plan')
windows = [item for item in old_plan['windows'] if item['cutoff'] > 96]
stable_tokens = sum(item['prompt_tokens']+256 for item in windows)
child_tokens = sum(item['prompt_tokens']+256 for item in windows[1:])
ceiling = reload_prompt['prompt_tokens']+256+forecast_prompt['prompt_tokens']+8192+stable_tokens+child_tokens
proposal = {'status':'awaiting-scope-and-budget-approval', 'change':'validate-model0-harness-integration',
 'event':event, 'original_review_sha256':digest(plan), 'original_training_objects':plan['proposal']['training_objects'],
 'original_budget_suggestion':plan['proposal']['budget_suggestion'],
 'required_scope_decision':'Append explicit controller authorization for a decision-model connectivity experiment on this natural event; preserve original model attribution.',
 'parent':{'model':base['model'],'checkpoint':'trio://run_ogvplibsgt3s/weights/step-16','base_model':'Qwen/Qwen3.8-27B','rank':16,'version':0,'authorized':False,'eligible':False},
 'method':{'algorithm':'sft-cross-entropy-one-pass','target':'forecast_next','minimum_cases':2,
           'parameters':{'learning_rate':0.00005,'grad_clip_norm':1.0,'seed':42}},
 'snapshot':{'cutoff':96,'cases':cases,'sha256':digest(cases),'authorization':None},
 'training_budget':{'workspace':'workspace_1o9brly7x8y1','service':'pytrio','max_jobs':1,'max_updates':2,
                    'max_sequence_tokens':sum(case['sequence_tokens'] for case in cases),'deadline_seconds_after_approval':3600},
 'inference_budget':{'max_calls':11,'max_tokens':ceiling,'reload_output_tokens':256,'forecast_output_tokens':8192,
                     'decide_output_tokens':256,'reload_deadline_seconds':120,'forecast_deadline_seconds':600,
                     'decide_deadline_seconds':120,'deadline_seconds_after_approval':7200},
 'reload_prompt':reload_prompt,'candidate_forecast_prompt':forecast_prompt,
 'replay':{'run':anchor['run'],'scenario':'C','source':'closed_loop','manifest':manifest,'cursor':96,
           'group':anchor['group'],'validation_first':97,'validation_last':128,'stop_at':240,
           'window_cutoffs':[item['cutoff'] for item in windows],
           'labels_budget':0,'industrial_training_jobs':0,'retry_training_jobs':0,
           'on_candidate_failure':'Record terminal result and continue stable model0 within the bounded replay.',
           'coverage':'Report naturally observed branches separately from controlled rule coverage.'},
 'remote_calls_during_preflight':0,'observations_read_through':96,'currency_cost':None,
 'currency_cost_note':'Service currency price and invoice unavailable; token/update limits are not a monetary cap.'}
output = root / 'runs/remaining-tests-proposal-2026-11-19.json'
for path, value in ((output,proposal),(root / 'runs/remaining-tests-token-cases-2026-11-19.json',samples)):
    if path.exists():
        assert read(path) == value, 'Frozen preflight changed: ' + str(path)
    else:
        with path.open('x',encoding='utf-8') as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2)
review.close()
print(canonical({'proposal':str(output),'cases':len(cases),'training_sequence_tokens':proposal['training_budget']['max_sequence_tokens'],
                 'inference_token_ceiling':ceiling,'reload_prompt':reload_prompt,'forecast_prompt':forecast_prompt,
                 'observations_read_through':96,'remote_calls':0,'original_training_objects':proposal['original_training_objects']}))
