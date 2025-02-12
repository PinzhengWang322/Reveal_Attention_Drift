from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

from functools import wraps, partial
import torch.nn as nn
from ..rome import repr_tools

# from .icl.lm_apis.lm_api_base import LMForwardAPI
# from .icl.analysis.attentioner_for_attribution import AttentionAdapter, GPT2AttentionerManager

from .attn_attr import (get_salinecies_score, 
                       choose_grad_head, 
                       gpt2_attn_with_detached_heads,
                       turn_to_origin_attn,
                       turn_to_grad_pune_attn)
import os

CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    select_head=False,
    topk = 3,
    label_type=None,
    choose_type=None,
    idx=None,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    
    deltas = execute_rome(model, tok, request, hparams, select_head=select_head, 
                          topk = topk, label_type=label_type, choose_type=choose_type, idx = idx)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layers_range = range(17,48),
    select_head = False,
    topk = 3,
    label_type = None,
    choose_type = None,
    idx = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    # Update target and print info
    request = deepcopy(request)
    if request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
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

    if select_head:
        subject_range = repr_tools.find_token_range(tok, tok(request['prompt'].format(request['subject'])).input_ids, request['subject'])
        inp = tok(request['prompt'].format(request['subject']), return_tensors='pt').to("cuda")
        
        if label_type in ["truth"]:
            label = tok(request['target_true']).input_ids[0]
        elif label_type == "pred":
            label = model(**inp).logits[0, -1, :].argmax()
        else:
            raise NotImplementedError
        
        wrap_model = LMForwardAPI(model=model, model_name="gpt2-xl", tokenizer=tok,
                     device='cuda',label_dict={})
        attentionermanger = GPT2AttentionerManager(wrap_model.model) 
        salinecies = get_salinecies_score(wrap_model, attentionermanger, inp, label)
    
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        torch.cuda.empty_cache()
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        
        if select_head:
            print(f"select head! mod: {choose_type}")
            partial_choose_func = partial(choose_grad_head, 
                                        choose_type = choose_type,
                                        layers_range=layers_range,
                                        subject_range=subject_range)
        
        if select_head:
            turn_to_grad_pune_attn(model, salinecies, partial_choose_func, topk = topk)

        torch.cuda.empty_cache()
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

        if select_head:
            turn_to_origin_attn(model)

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


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
