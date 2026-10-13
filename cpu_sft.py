"""Real bounded CPU LoRA training/reload on public coding instruction records."""
import argparse, hashlib, json, math, time
from pathlib import Path
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

MODEL='roneneldan/TinyStories-1M'
REVISION='77f1b168e219585646439073245fe87e56b3023e'
DATASET='iamtarun/python_code_instructions_18k_alpaca'
DATA_URL='https://datasets-server.huggingface.co/rows'

def load_rows(path=None):
    if path:
        payload=Path(path).read_bytes();data=json.loads(payload)
    else:
        response=requests.get(DATA_URL,params={'dataset':DATASET,'config':'default','split':'train','offset':0,'length':24},timeout=30)
        response.raise_for_status();payload=response.content;data=response.json()
    rows=[entry['row'] for entry in data['rows']]
    if len(rows)<24:raise ValueError('Need 24 records for independent 16/8 split')
    return rows[:24],hashlib.sha256(payload).hexdigest()

def encode(row,tokenizer,max_length=96):
    prompt='### Instruction:\n'+row['instruction']
    if row.get('input'):prompt+='\n'+row['input']
    prompt+='\n### Response:\n'
    # Reserve response space: long instructions must not mask every target.
    p=tokenizer.encode(prompt,add_special_tokens=False)[:40]
    response=tokenizer.encode(row['output'],add_special_tokens=False)[:max_length-len(p)-1]+[tokenizer.eos_token_id]
    tokens=p+response;labels=[-100]*len(p)+response
    padding=max_length-len(tokens)
    return {'input_ids':tokens+[tokenizer.pad_token_id]*padding,'attention_mask':[1]*len(tokens)+[0]*padding,'labels':labels+[-100]*padding}

def batch(records,tokenizer):
    encoded=[encode(r,tokenizer) for r in records]
    return {key:torch.tensor([r[key] for r in encoded]) for key in encoded[0]}

def evaluate(model,records,tokenizer):
    model.eval();losses=[]
    with torch.inference_mode():
        for start in range(0,len(records),2):
            loss=model(**batch(records[start:start+2],tokenizer)).loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite heldout loss')
            losses.append(float(loss))
    return sum(losses)/len(losses)

def run(output,steps=20,rows_path=None):
    if not 1<=steps<=500:raise ValueError('Steps must be 1..500')
    torch.set_num_threads(1);torch.manual_seed(42)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows,data_hash=load_rows(rows_path);train,test=rows[:16],rows[16:24]
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION).cpu()
    model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    trainable={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    before=evaluate(model,test,tokenizer)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-3)
    history=[];start_time=time.perf_counter()
    model.train()
    for step in range(steps):
        offset=(2*step)%len(train);data=batch(train[offset:offset+2],tokenizer)
        optimizer.zero_grad();loss=model(**data).loss
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();history.append(float(loss))
    after=evaluate(model,test,tokenizer)
    changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(p,trainable[n])]
    if not changed:raise RuntimeError('Training did not change adapter parameters')
    if any(not torch.equal(p,frozen[n]) for n,p in model.named_parameters() if not p.requires_grad):raise RuntimeError('LoRA optimization changed frozen base weights')
    model.save_pretrained(out/'adapter');tokenizer.save_pretrained(out/'tokenizer')
    probe=batch(test[:1],tokenizer)
    with torch.inference_mode():expected=model(**probe).logits.detach().clone()
    del model,base
    reloaded=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True).cpu(),out/'adapter').eval()
    with torch.inference_mode():actual=reloaded(**probe).logits
    maximum_error=float((actual-expected).abs().max())
    if not torch.allclose(actual,expected,atol=1e-5,rtol=1e-5):raise RuntimeError('Adapter reload changed model logits')
    prompt='### Instruction:\nWrite a Python function to add two numbers.\n### Response:\n'
    inputs=tokenizer(prompt,return_tensors='pt')
    with torch.inference_mode():generated=reloaded.generate(**inputs,max_new_tokens=24,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    text=tokenizer.decode(generated[0],skip_special_tokens=True)
    result={'model':MODEL,'model_revision':REVISION,'dataset':DATASET,'dataset_source':DATA_URL,'dataset_sha256':data_hash,'train_indices':list(range(16)),'heldout_indices':list(range(16,24)),'steps':steps,'max_tokens':96,'instruction_token_cap':40,'trainable_parameters':sum(p.numel() for p in trainable.values()),'changed_parameter_tensors':len(changed),'frozen_base_unchanged':True,'heldout_loss_before':before,'heldout_loss_after':after,'training_losses':history,'reload_max_logit_error':maximum_error,'elapsed_train_reload_seconds':time.perf_counter()-start_time,'generation':text,'scope':'Real CPU LoRA pipeline on a tiny story pretrained model and truncated public coding records. Generated code quality is not established; not reproduction of 7B GPU QLoRA or capability retention.'}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(range(1,len(history)+1),history,color='#337565',linewidth=2);axes[0].set(xlabel='Optimization step',ylabel='Supervised batch loss',title='Actual LoRA optimization')
    axes[1].bar(['Before','After'],[before,after],color=['#a9b4a5','#337565']);axes[1].set(ylabel='8-record heldout loss',title='Tiny model / truncated coding inputs')
    fig.tight_layout();fig.savefig(out/'loss.png',dpi=150);plt.close(fig)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='cpu-sft-output');parser.add_argument('--steps',type=int,default=20);parser.add_argument('--rows');args=parser.parse_args()
    print(json.dumps(run(args.output,args.steps,args.rows),indent=2))
