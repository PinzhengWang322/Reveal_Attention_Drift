from dsets import CounterFactDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import importlib
import numpy as np
import torch
from transformers import set_seed
import time
import os
from functools import wraps, partial
from rome import ROMEHyperParams
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)

from utils.tools import (
    ModelAndTokenizer,
    make_inputs, 
    layername, 
    get_topk_tokens_info, 
    get_module,
    find_token_range,
    plot_trace_heatmap,
    decode_tokens
)
from utils.casual_trace import(
    collect_embedding_std,
    trace_important_states,
    trace_important_window,
    trace_with_patch
) 
from utils.dsets import KnownsDataset

from copy import deepcopy
from rome.rome_main import get_context_templates, upd_matrix_match_shape
from rome.compute_u import compute_u
from rome.compute_v import compute_v
from util import nethook
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
import torch.nn as nn
from matplotlib import pyplot as plt

import torch.nn.functional as F
import icl.analysis.attentioner_for_attribution
importlib.reload(icl.analysis.attentioner_for_attribution)
from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.analysis.attentioner_for_attribution import AttentionAdapter, GPT2AttentionerManager



def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def test_case(case, tok, model, test=True, ori=True, gen=True):
    if ori:
        prompt = case['prompt']
        print("Testing with original prompt:")
        print("Generated output:", tok.batch_decode(model.generate(**tok(case['prompt'], return_tensors='pt').to('cuda'), max_new_tokens=10))[0])
        logits = model(tok(prompt, return_tensors='pt')['input_ids'].to('cuda')).logits[0, -1, :]
        probs = logits.softmax(dim=-1)
        top3_tokens = torch.topk(probs, 3)
        top3_probabilities = top3_tokens.values.tolist()
        top3_logits = torch.topk(logits, 3).values.tolist()
        top3_tokens_ids = top3_tokens.indices.tolist()
        top3_tokens = tok.convert_ids_to_tokens(top3_tokens_ids)
        result = list(zip(top3_tokens, top3_probabilities))
        print("Top 3 words and probabilities:", result)
        specific_token = case['target_true']
        print(f"Probability and output for specified word {specific_token} (target_true):", probs[tok(specific_token).input_ids[0]].item())
        specific_token = case['target_new']
        print(f"Probability for specified word {specific_token} (target_new):", probs[tok(specific_token).input_ids[0]].item())
        print('-' * 100)
    if test:
        prompt = case['test_prompt']
        print("Testing with test prompt:")
        if gen:
            print("Generated output:", tok.batch_decode(model.generate(**tok(case['test_prompt'], return_tensors='pt').to('cuda'), max_new_tokens=10))[0])
        logits = model(tok(prompt, return_tensors='pt')['input_ids'].to('cuda')).logits[0, -1, :]
        probs = logits.softmax(dim=-1)
        top3_tokens = torch.topk(probs, 3)
        top3_probabilities = top3_tokens.values.tolist()
        top3_logits = torch.topk(logits, 3).values.tolist()
        top3_tokens_ids = top3_tokens.indices.tolist()
        top3_tokens = tok.convert_ids_to_tokens(top3_tokens_ids)
        result = list(zip(top3_tokens, top3_probabilities))
        print("Top 3 words and probabilities:", result)
        specific_token = case['test_true']
        print(f"Probability and output for specified word {specific_token} (test_true):", \
            probs[tok(specific_token).input_ids[0]].item())
        specific_token = case['target_new']
        print(f"Probability and output for specified word {specific_token} (target_new):", \
            probs[tok(specific_token).input_ids[0]].item())
    
    
def run_rome(case, hparams, model, tok):
    def edit_one(requested_rewrite):
        deltas = execute_rome(model, tok, requested_rewrite, hparams)
        weights_copy = {}
        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                weights_copy[w_name] = w.detach().clone()
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                w[...] += upd_matrix
                

        return model, weights_copy

    requested_rewrite = {'prompt': case['prompt'].replace(case['subject'], '{}'),
    'target_new': {'str': case['target_new']},
    'subject': case['subject']}

    edited_model, weights_copy = edit_one(requested_rewrite)
    return weights_copy

def model_reset(model, weights_copy):
    with torch.no_grad():
        for k, v in weights_copy.items():
            nethook.get_parameter(model, k)[...] = v.to("cuda")

# change the model with knockout

def gpt2_attn_with_knockout(self, query, key, value, attention_mask=None, head_mask=None,
                      knockout_weight=None):
    
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    if knockout_weight is not None:
        attn_weights = attn_weights + knockout_weight.unsqueeze(0).to(attn_weights.device)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights
    
def add_knockout_attn(model, knockout_dic):
    for i, layer in enumerate(model.transformer.h):
        layer.attn._attn = partial(gpt2_attn_with_knockout, layer.attn,
                                knockout_weight = knockout_dic[i] if i in knockout_dic else None)
        
# Back to original attention
def ori_attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

def attn_reset(model):
    for i, layer in enumerate(model.transformer.h):
        layer.attn._attn = partial(ori_attn, layer.attn)

# Utilties for image visualize

def plot_trace_heatmap(differences, 
                       low_score, 
                       answer_token, 
                       kind, 
                       savepdf, 
                       window,
                       input_tokens,
                       noised_range,
                       title=None, 
                       annotation=None, 
                       xlabel=None, 
                       modelname=None):

    labels = list(input_tokens)
    for i in range(*noised_range):
        labels[i] = labels[i] + "*"

    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = kind
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if annotation is not None:
            ax.annotate(xlabel)
        elif answer_token is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer_token).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def noise_test(mt, case, trace_word, noise_level):
    test_prompt = case['test_prompt']
    noise_target = case['subject']
    inp = make_inputs(mt.tokenizer, [test_prompt])
    noised_range = find_token_range(mt.tokenizer, inp["input_ids"][0], noise_target)

    # Clean Run
    trace_word_id = mt.tokenizer(trace_word).input_ids[0]
    inp = make_inputs(mt.tokenizer, [test_prompt])
    token_logits = mt.model(**inp).logits[0,-1]
    clean_probs = torch.softmax(token_logits, dim=-1)
    ori_ans_token = clean_probs.argmax().item()
    top3_tokens = torch.topk(clean_probs, 3)
    top3_probabilities = top3_tokens.values.tolist()
    top3_tokens_ids = top3_tokens.indices.tolist()
    top3_tokens = mt.tokenizer.convert_ids_to_tokens(top3_tokens_ids)
    result = list(zip(top3_tokens, top3_probabilities))
    print("Next word probabilities:", result)
    print("Clean Run:")
    print(f"Probability of trace word \"{trace_word}\": {clean_probs[trace_word_id]}")
    print(f"Top words: {result}")
    print('-' * 50)

    # Noised Run
    inp = make_inputs(mt.tokenizer, [test_prompt] * (10 + 1))
    full_output = trace_with_patch(
        mt.model, inp, [], torch.tensor(ori_ans_token).cuda(), noised_range, noise=noise_level, full_output=True
    )
    low_score, dirty_probs = full_output["ans_prob"], full_output["probs"]
    top3_tokens = torch.topk(dirty_probs, 3)
    top3_probabilities = top3_tokens.values.tolist()
    top3_tokens_ids = top3_tokens.indices.tolist()
    top3_tokens = mt.tokenizer.convert_ids_to_tokens(top3_tokens_ids)
    result = list(zip(top3_tokens, top3_probabilities))
    print("Noised Run:")
    print(f"Probability of trace word \"{trace_word}\": {dirty_probs[trace_word_id]}")
    print(f"Top words: {result}")

def plot_casual_trace(mt, window_size, kinds, case, trace_word, noise_level):
    test_prompt = case['test_prompt']
    noise_target = case['subject']
    inp = make_inputs(mt.tokenizer, [test_prompt] * (10 + 1))
    noised_range = find_token_range(mt.tokenizer, inp["input_ids"][0], noise_target)
    trace_word_id = mt.tokenizer(trace_word).input_ids[0]
    res = {}
    for kind in kinds:

        start = time.time()

        differences = trace_important_window(
            mt.model, 
            mt.num_layers, 
            inp, 
            noised_range, 
            torch.tensor(trace_word_id).cuda(), 
            noise = noise_level,
            window = window_size,
            kind = kind,
            t_window=1,
        ).detach().cpu()

        end = time.time()
        print("cost time:", end - start)

        plot_trace_heatmap(
            differences=differences,
            low_score=-0.1,
            answer_token=decode_tokens(mt.tokenizer, [trace_word_id]),
            kind=str(kind),
            savepdf=f"",
            window=window_size,
            input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
            noised_range=noised_range,
            modelname="GPT",
        )
        res[kind] = differences
    return res

def plot_attn_knockout(window_size, case, model, tok, trace_word, layers_num=48, head_num=25):
    trace_word_id = tok(trace_word).input_ids[0]
    input_text = case['test_prompt']
    noise_target = case['subject']
    inp = tok(input_text, return_tensors='pt')
    w_sent = tok.batch_decode(inp.input_ids[0]) 
    noised_range = range(*find_token_range(tok, inp["input_ids"][0], noise_target))
    scores = []
    for sid in range(layers_num - window_size):
        knockout_dic = {}
        cut_layer_range = range(sid, sid + window_size)
        for layer in cut_layer_range:
            knockout_dic[layer] = torch.zeros(25, len(w_sent), len(w_sent))
            for head in range(head_num):
                for src in [len(w_sent) - 1]:
                    for tgt in noised_range:
                        knockout_dic[layer][head, src, tgt] = -torch.inf
                
        add_knockout_attn(model, knockout_dic)
        logits = model(tok(case['test_prompt'], return_tensors='pt')['input_ids'].to('cuda')).logits[0, -1, :]
        probs = logits.softmax(dim=-1)
        scores.append((probs[trace_word_id].item(), sid + window_size // 2))

    values, coordinates = zip(*scores)

    # 创建一个与数据长度相同的矩阵，用于绘制热力图
    heatmap = np.zeros((1, len(scores)))

    # 将值填充到热力图矩阵中
    for i, value in enumerate(values):
        heatmap[0, i] = value

    # 绘制热力图
    plt.figure(figsize=(8, 1))
    plt.imshow(heatmap, interpolation='nearest', aspect='auto', vmin=0, vmax=0.2)
    plt.colorbar()

    step = len(coordinates) // 10  # 选择步长，这里大概每五个点显示一个标签
    plt.xticks(ticks=np.arange(0, len(coordinates), step), labels=coordinates[::step], rotation=45)
    plt.yticks([])  # 隐藏y轴标签

    plt.title("1D Heatmap")
    plt.show()


def get_saliency_scores(model, tok, trace_word, ):
    wrap_model = LMForwardAPI(model=model, model_name="gpt2-xl", tokenizer=tok,
                              device='cuda',label_dict={})
    attentionermanger = GPT2AttentionerManager(wrap_model.model)
    pred_token = " English"
    input_text = "Michel Denisot spoke the language Russian. Stephen Hawking's language is"
    inp = tok(input_text, return_tensors='pt').to("cuda")
    label = tok(trace_word).input_ids[0]

    for p in wrap_model.parameters():
        p.requires_grad = False
    attentionermanger.zero_grad()

    output = wrap_model(**inp)
    loss = F.cross_entropy(output['ori_logits'], 
                        torch.tensor([label]).to(output['ori_logits'].device))
    print(loss.item())
    loss.backward()