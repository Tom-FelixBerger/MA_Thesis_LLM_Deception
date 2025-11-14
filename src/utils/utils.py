import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
import torch
from pathlib import Path
from utils import templates
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"

### Model Loading Utilities ###
MODELS = {
    "mistral-7b-v03": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "excluded": True,
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
    },
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b-it",
        "excluded": True,
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b-it",
        "excluded": False,
        "num_layers": 42,
        "num_heads": 16,
        "head_dim": 256,
        "start_layer": 21
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "excluded": False,
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "start_layer": 16
    },
}

def get_model_dims(model_key):
    num_layers = MODELS[model_key]["num_layers"]
    num_heads = MODELS[model_key]["num_heads"]
    head_dim = MODELS[model_key]["head_dim"]
    return num_layers, num_heads, head_dim



def load_huggingface_auth_token():
    token_file = PROJECT_ROOT / "credentials.txt"
    with open(token_file, "r") as f:
        for line in f:
            if line.startswith("HUGGINGFACE_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    return token

def load_model_and_tokenizer(model_key):
    print(f"Loading model and tokenizer for {model_key}")

    huggingface_auth_token = load_huggingface_auth_token()

    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_key]["model_id"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="bfloat16",
            bnb_4bit_use_double_quant=True,
        ),
        token=huggingface_auth_token,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
        dtype="bfloat16",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS[model_key]["model_id"],
        trust_remote_code=True,
        token=huggingface_auth_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

### Vignette Generation Utilities ###
def fix_capitalization(text):
    sentences = text.split('\n')
    fixed_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            if s[0] == '"':
                s = s[0] + s[1].upper() + s[2:] if s[1].isalpha() else s
            else:
                s = s[0].upper() + s[1:] if s[0].isalpha() else s
        fixed_sentences.append(s)
    return "\n".join(fixed_sentences)

def fill(template, instruction, fill_comb):
    scenario = template['scenario']
    response_a = template['response_a']
    response_b = template['response_b']
    question = template['question']
    for k, v in fill_comb.items():
        scenario = fix_capitalization(scenario.replace(f'{{{k}}}', v))
        response_a = fix_capitalization(response_a.replace(f'{{{k}}}', v))
        response_b = fix_capitalization(response_b.replace(f'{{{k}}}', v))

        if question:
            question = fix_capitalization(question.replace(f'{{{k}}}', v))

        if instruction:
            instruction = instruction.replace(f'{{{k}}}', v).replace('{response_a}', response_a).replace('{response_b}', response_b)
            instruction = fix_capitalization(instruction)

    return scenario, response_a, response_b, question, instruction

def generate_fill_combs(template):
    fill_combs = []

    if 'other_referencing_protagonists' in template.keys():
        protagonists = template['other_referencing_protagonists']
    else:
        protagonists = [(p, "") for p in template['protagonists']]
        
    for obj, (prot, verb), (attr_a, attr_b) in itertools.product(
        template['objects'],
        protagonists,
        [
            (template['attributes'][0], template['attributes'][1]),
            (template['attributes'][1], template['attributes'][0]),
        ]
    ):
        fill_comb = {
            'object': obj,
            'protagonist': prot,
            'verb': verb,
            'attribute_a': attr_a,
            'attribute_b': attr_b,
        }
        fill_combs.append(fill_comb)
    return fill_combs

def generate_deception_incentive_vignettes(template_ids, return_question):
    vignettes = []
    for t_id in template_ids:
        templ = templates.TEMPLATES[t_id]
        for fill_comb in generate_fill_combs(templ):
            scenario, response_a, response_b, question, instruction = fill(
                template=templ,
                instruction=templates.INSTRUCTION_DECEPTION_INCENTIVE,
                fill_comb=fill_comb,
            )
            vign = {
                'messages': [
                    {"role": "user", "content": scenario + instruction,},
                ],
                'response_a': response_a,
                'response_b': response_b,
                'template_id': t_id,
            }
            if return_question:
                vign['question'] = question
                vign['attributes'] = templ['attributes']
            vignettes.append(vign)
    return vignettes

def generate_belief_inference_vignettes(template_ids):
    vignettes = []
    for t_id in template_ids:
        templ = templates.TEMPLATES[t_id]
        for fill_comb in generate_fill_combs(templ):
            scenario, response_a, response_b, question, instruction = fill(
                template=templ,
                instruction=templates.INSTRUCTION_DECEPTION_INCENTIVE,
                fill_comb=fill_comb,
            )
            for target_c, quest_attr in [(False, 'attribute_a'), (True, 'attribute_b')]:
                repl_quest = question.replace('{question_attribute}', fill_comb[quest_attr])
                for target_p, response in [(quest_attr == 'attribute_a', response_a), (quest_attr == 'attribute_b', response_b)]:
                    vign = {
                        'messages': [
                            {"role": "user", "content": scenario + instruction},
                            {"role": "assistant", "content": response},
                            {"role": "user", "content": repl_quest},
                        ],
                        'target_c': target_c,
                        'target_p': target_p,
                        'template_id': t_id,
                    }
                    vignettes.append(vign)
    return vignettes

### Model Text Generation Utilities ###
def tokenize_batch(message_batch, tokenizer):
    encoded = tokenizer.apply_chat_template(
        message_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True, # This is required for batching
        truncation=False,
    )
    attention_mask = (encoded != tokenizer.pad_token_id).long()
    encoded = BatchEncoding({
        "input_ids": encoded,
        "attention_mask": attention_mask,
    })
    return encoded

def batch_generate_text(model, tokenizer, message_batch):
    encoded = tokenize_batch(message_batch, tokenizer)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    input_len = encoded["input_ids"].shape[1]
    for i in range(len(message_batch)):
        generated_ids = outputs[i, input_len:]

        only_new = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        full = tokenizer.decode(
            outputs[i],
            skip_special_tokens=False
        )
        results.append((full, only_new))

    return results

def classify_response(response, response_a, response_b):
    target_dec = clean_text(response_a)
    target_hon = clean_text(response_b)
    response_clean = clean_text(response)

    if target_dec == response_clean:
        return 'deceptive'
    elif target_hon == response_clean:
        return 'honest'
    else:
        return 'invalid'

def clean_text(text):
    text = text.replace("&#39;", "'")
    text = text.replace("&quot;", '"')
    return ' '.join(c for c in text if c.isalnum() or c.isspace()).lower().strip()


### Activation Extraction Utilities ###

def extract_batch_attention_outputs_pre_projection(message_batch, model, tokenizer, model_key):
    num_layers, num_heads, head_dim = get_model_dims(model_key)

    encoded = tokenize_batch(message_batch, tokenizer)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    batch_size = input_ids.shape[0]

    layers = model.model.layers

    # collected[l] = (B, H, D)
    collected = [None] * num_layers

    def extract_from_tensor(tensor, layer_idx):
        B, S, H = tensor.shape
        reshaped = tensor.view(B, S, num_heads, head_dim)

        # pick last non-pad token
        final_idx = attention_mask.sum(dim=1) - 1
        out = np.stack([
            reshaped[b, final_idx[b]].detach().cpu().float().numpy()
            for b in range(B)
        ])
        collected[layer_idx] = out  # (B, H, D)

    def make_hook(layer_idx):
        def hook(mod, inputs, outputs):
            extract_from_tensor(inputs[0], layer_idx)
        return hook

    hooks = []
    for i in range(num_layers):
        attn = layers[i].self_attn
        module = attn.o_proj
        hooks.append(module.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(**encoded)

    for h in hooks:
        h.remove()

    # Stack into (B, L, H, D)
    attention_tensor = np.stack(collected, axis=1)

    return attention_tensor