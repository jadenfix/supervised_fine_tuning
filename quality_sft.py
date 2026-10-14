"""Reproducible CPU baseline/adaptation comparison; no generated code is executed."""
import argparse
import ast
import hashlib
import html
import importlib.metadata
import json
import platform
import random
import re
import time
from pathlib import Path

import requests
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
DATASET = 'iamtarun/python_code_instructions_18k_alpaca'
DATA_URL = 'https://datasets-server.huggingface.co/rows'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(row):
    # Same instruction/input cannot occur on both sides with different answers.
    return digest(' '.join((row['instruction'] + ' ' + row.get('input', '')).lower().split()).encode())


def encode(row, tokenizer, max_length=512):
    messages = [{'role': 'user', 'content': row['instruction'] + ('\n' + row['input'] if row.get('input') else '')}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    response = tokenizer.encode(row['output'], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Skip rather than silently cutting instructions, answers or the assistant marker.
    if len(prefix) + len(response) > max_length:
        return None
    return {'input_ids': prefix + response, 'labels': [-100] * len(prefix) + response}


def split_rows(rows, tokenizer, train_size=32, test_size=8):
    eligible = []
    seen_prompts, seen_answers = set(), set()
    skipped = {'duplicate_prompt': 0, 'duplicate_answer': 0, 'too_long': 0}
    for index, row in enumerate(rows):
        key = fingerprint(row)
        answer = digest(' '.join(row['output'].split()).encode())
        if key in seen_prompts:
            skipped['duplicate_prompt'] += 1
            continue
        if answer in seen_answers:
            skipped['duplicate_answer'] += 1
            continue
        if encode(row, tokenizer) is None:
            skipped['too_long'] += 1
            continue
        seen_prompts.add(key)
        seen_answers.add(answer)
        eligible.append(dict(row, source_index=index, prompt_sha256=key, answer_sha256=answer))
    random.Random(42).shuffle(eligible)
    if len(eligible) < train_size + test_size:
        raise ValueError(f'Only {len(eligible)} complete unique examples; need {train_size + test_size}')
    return eligible[:train_size], eligible[train_size:train_size + test_size], skipped


def batch(rows, tokenizer):
    items = [encode(r, tokenizer) for r in rows]
    if any(i is None for i in items):
        raise ValueError('Untruncated record exceeds token budget')
    length = max(len(i['input_ids']) for i in items)
    return {key: torch.tensor([i[key] + ([tokenizer.pad_token_id] if key == 'input_ids' else [-100]) * (length-len(i[key])) for i in items]) for key in ('input_ids', 'labels')} | {'attention_mask': torch.tensor([[1] * len(i['input_ids']) + [0] * (length-len(i['input_ids'])) for i in items])}


def evaluate(model, rows, tokenizer):
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for row in rows:
            data = batch([row], tokenizer)
            n = int((data['labels'][:, 1:] != -100).sum())
            loss = model(**data).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            total += float(loss) * n
            count += n
    return {'response_token_nll': total/count, 'scored_tokens': count, 'records': len(rows)}


def code_quality(text):
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    code = blocks[0] if blocks else text.strip()
    try:
        tree = ast.parse(code)
        functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return {'nonblank': bool(code.strip()), 'parseable_python': bool(tree.body), 'function_names': functions}
    except (SyntaxError, ValueError):
        return {'nonblank': bool(code.strip()), 'parseable_python': False, 'function_names': []}


def generate(model, rows, tokenizer):
    model.eval()
    outputs = []
    for row in rows:
        content = row['instruction'] + ('\n' + row['input'] if row.get('input') else '')
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=192, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        continuation = tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        outputs.append({'source_index': row['source_index'], 'instruction': content, 'reference': row['output'], 'continuation': continuation, 'quality': code_quality(continuation), 'hit_token_budget': len(ids[0])-inputs['input_ids'].shape[1] == 192})
    return outputs


def write_report(out, result):
    cards = []
    for before, after in zip(result['baseline_generations'], result['adapted_generations']):
        cards.append('<article><h2>Held-out record '+str(before['source_index'])+'</h2><p>'+html.escape(before['instruction'])+'</p><div class="pair"><section><h3>Unchanged pretrained model</h3><pre>'+html.escape(before['continuation'])+'</pre></section><section><h3>Reloaded adapter</h3><pre>'+html.escape(after['continuation'])+'</pre></section></div><details><summary>Reference and diagnostics</summary><pre>'+html.escape(json.dumps({'reference': before['reference'], 'baseline': before['quality'], 'adapted': after['quality'], 'baseline_truncated': before['hit_token_budget'], 'adapted_truncated': after['hit_token_budget']}, indent=2))+'</pre></details></article>')
    page = '<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPU LoRA comparison</title><style>body{background:#faf7f0;color:#20372f;font:16px system-ui;max-width:1200px;margin:auto;padding:28px}h1{font-size:42px}article{background:white;border:1px solid #ddd8cc;border-radius:16px;padding:24px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f1;padding:16px;border-radius:8px}p{line-height:1.6}@media(max-width:650px){.pair{grid-template-columns:1fr}body{padding:14px}h1{font-size:30px}}</style><h1>Does adaptation improve the answer?</h1><p>All held-out records are shown. Complete examples, response-only loss, fixed greedy decoding, fixed training budget and exact adapter reload. Syntax alone does not establish functional correctness. Outputs are not executed.</p><pre>'+html.escape(json.dumps({k:result[k] for k in ('model','model_revision','baseline_evaluation','adapted_evaluation','reload_max_logit_error','limitations')},indent=2))+'</pre>'+''.join(cards)+'</html>'
    (out/'comparison.html').write_text(page)


def run(output, steps=64, rows_path=None):
    if not 1 <= steps <= 128:
        raise ValueError('Steps must be 1..128')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    started = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if rows_path:
        payload = Path(rows_path).read_bytes()
    else:
        response = requests.get(DATA_URL, params={'dataset':DATASET,'config':'default','split':'train','offset':0,'length':100}, timeout=60)
        response.raise_for_status()
        payload = response.content
    (out/'source-rows.json').write_bytes(payload)
    rows = [e['row'] for e in json.loads(payload)['rows']]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    train, test, skipped = split_rows(rows, tokenizer)
    (out/'split.json').write_text(json.dumps({'train':train, 'heldout':test, 'skipped':skipped}, indent=2))
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32).cpu()
    before = evaluate(base, test, tokenizer)
    baseline_generations = generate(base, test, tokenizer)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
    initial = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    frozen_hash = {n:digest(p.detach().numpy().tobytes()) for n,p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    history = []
    for step in range(steps):
        indices = [(2*step)%len(train), (2*step+1)%len(train)]
        optimizer.zero_grad()
        loss = model(**batch([train[i] for i in indices], tokenizer)).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        history.append(float(loss.detach()))
        if step%8 == 0:
            print(json.dumps({'step':step+1,'loss':history[-1]}), flush=True)
    after = evaluate(model, test, tokenizer)
    changed = [n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(p, initial[n])]
    if not changed or any(digest(p.detach().numpy().tobytes()) != frozen_hash[n] for n,p in model.named_parameters() if not p.requires_grad):
        raise RuntimeError('Adapter update/frozen base invariant failed')
    model.save_pretrained(out/'adapter')
    tokenizer.save_pretrained(out/'tokenizer')
    probe = batch(test[:1], tokenizer)
    with torch.inference_mode():
        expected = model(**probe).logits.clone()
    del model, base, optimizer
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True).cpu(), out/'adapter').eval()
    with torch.inference_mode():
        actual = reloaded(**probe).logits
    error = float((expected-actual).abs().max())
    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
        raise RuntimeError('Reload changed logits')
    adapted_generations = generate(reloaded, test, tokenizer)
    result = {'model':MODEL, 'model_revision':REVISION, 'dataset':DATASET, 'dataset_source':DATA_URL, 'dataset_sha256':digest(payload), 'seed':42, 'steps':steps, 'learning_rate':2e-4, 'max_complete_sequence_tokens':512, 'generation_max_new_tokens':192, 'train_indices':[r['source_index'] for r in train], 'heldout_indices':[r['source_index'] for r in test], 'skipped':skipped, 'baseline_evaluation':before, 'adapted_evaluation':after, 'training_losses':history, 'changed_parameter_tensors':len(changed), 'frozen_base_unchanged':True, 'reload_max_logit_error':error, 'baseline_generations':baseline_generations, 'adapted_generations':adapted_generations, 'elapsed_seconds':time.perf_counter()-started, 'source_code_sha256':digest(Path(__file__).read_bytes()), 'runtime':{'python':platform.python_version(), 'platform':platform.platform(), 'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate')}}, 'limitations':'32 training / 8 held-out public records; pretraining contamination unknown, near-duplicate/semantic leakage not ruled out. Syntax and reference likelihood do not prove executed code correctness, broad coding capability or retention. Fixed single run; no held-out checkpoint selection. Original 7B GPU notebook not reproduced.'}
    (out/'result.json').write_text(json.dumps(result, indent=2))
    write_report(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='quality-sft-output')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--rows')
    args = parser.parse_args()
    result = run(args.output, args.steps, args.rows)
    print(json.dumps({k:result[k] for k in ('baseline_evaluation','adapted_evaluation','reload_max_logit_error','elapsed_seconds')}, indent=2))
