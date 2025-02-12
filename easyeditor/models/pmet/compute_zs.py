from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .pmet_hparams import PMETHyperParams


def compute_zs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: PMETHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """


    # Get model parameters
    if "gptj" or "gpt2" in model.config._name_or_path:
        ln_f = nethook.get_module(model, hparams.ln_f_module)
        lm_head_module = nethook.get_module(model, hparams.lm_head_module)
        lm_w = nethook.get_parameter(lm_head_module, "weight").T
    else:
        lm_w, ln_f = (
            nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
            nethook.get_module(model, hparams.ln_f_module),
        )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    is_llama = 'tinyllama' in hparams.model_name.lower() or 'llama13b' in hparams.model_name.lower()
    target_ids = tok(request["target_new"] if not is_llama else request["target_new"][1:], return_tensors="pt").to(f"cuda")[
        "input_ids"
    ][0]
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_ori_prompts = [
        context.format(request["prompt"])
        for context_types in context_templates
        for context in context_types
    ]
        
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + (' ' if (is_llama and len(target_ids) > 1)
                                             else '') + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    
    input_tok_lst = [tok(prompt.format(request["subject"])).input_ids for prompt in rewriting_ori_prompts]
    input_full_tok_lst = [tok(prompt.format(request["subject"])).input_ids for prompt in rewriting_prompts]

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
        


    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if "neo" or "llama" in model.config._name_or_path:
        delta_attn = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
        delta_mlp = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    else:
        delta_attn = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
        delta_mlp = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init_attn, target_init_mlp, kl_distr_init = None, None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init_attn, target_init_mlp

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init_mlp is None:
                print("Recording initial value of v* in mlp")
                # Initial value is recorded for the clean sentence
                target_init_mlp = cur_out[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta_mlp
        if cur_layer == hparams.attn_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init_attn is None:
                print("Recording initial value of v* in attn")
                # Initial value is recorded for the clean sentence
                target_init_attn = cur_out[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta_attn
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta_mlp, delta_attn], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    
    ori_attn_W = None
    ori_attn_K = None
    ori_hid = None
    
    def kl_divergence(p, q, input_lens, lookup_idxs):
        eps = 1e-8 if p.dtype != torch.float16 else 1e-7
        p = p[:len(input_lens)] + eps
        q = q[:len(input_lens)] + eps
        max_subj_p, max_subj_q = [], []
        for bidx, (ilen, sub_last) in enumerate(zip(input_lens, lookup_idxs)):
            max_subj_p.append(p[bidx,:,ilen, sub_last].max(dim = -1).values)
            max_subj_q.append(q[bidx,:,ilen, sub_last])
        max_subj_p,  max_subj_q = torch.stack(max_subj_p), torch.stack(max_subj_q)
        mask = max_subj_q > max_subj_p.unsqueeze(-1)
        res = (p * (p / q).log()) * mask.unsqueeze(-1).unsqueeze(-1)
        return sum([res[bidx,:,ilen,:].sum() for bidx, ilen in enumerate(input_lens)]) / len(input_lens)
    
    def js_divergence(p, q, input_lens):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m, input_lens) + 0.5 * kl_divergence(q, m, input_lens)
    
    def L2_loss(p, q, input_lens):
        res = (p - q)**2
        return sum([res[bidx,:,ilen,:].sum() for bidx, ilen in enumerate(input_lens)]) / len(input_lens)
    
    nll_loss_factor = hparams.nll_loss_factor
    kl_factor = hparams.kl_factor
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
                hparams.attn_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            if hparams.last_hid_restrain_layers is not None:
                restrain_layers = eval(hparams.last_hid_restrain_layers)
            else:
                restrain_layers = None
            
            predict_idxs = [len(i) - 1 for i in input_tok_lst]
            input_lens = [len(i) for i in input_full_tok_lst]
            output = model(**input_tok, output_attentions=True, output_hidden_states=True)
            logits = output.logits
            attn_W = output['attentions']
            attn_K = [i[0] for i in output['past_key_values']]

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        
        if ori_attn_W is None:
            ori_attn_W = attn_W
            attn_W_loss = torch.tensor(0).cuda()
        else:
            loss_function = dict(
                KL = kl_divergence,
                JS = js_divergence,
                L2 = L2_loss,
            )[hparams.loss_type]
            if hparams.high_attn_range is None:
                attn_W_loss = sum([loss_function(p.detach(), q, predict_idxs, lookup_idxs) for q, p in zip(attn_W, ori_attn_W)]) * hparams.attn_W_loss_weight * 1e-4
            else:
                attn_W_loss = sum([loss_function(p.detach(), q, predict_idxs, lookup_idxs) for lidx, (q, p) in enumerate(zip(attn_W, ori_attn_W))
                                   if lidx in eval(hparams.high_attn_range)]) * hparams.attn_W_loss_weight * 1e-4
            
        if ori_attn_K is None:
            ori_attn_K = attn_K
            attn_K_loss = torch.tensor(0).cuda()
        else:
            if hparams.high_attn_range is None:
                attn_K_loss = sum([sum([
                               torch.norm(i[bidx,:,:ilen] - j[bidx,:,:ilen].detach())
                               for bidx, ilen in enumerate(input_lens)
                               ])
                               for i, j in zip(attn_K, ori_attn_K)]) * hparams.attn_K_loss_weight * 1e-4
            else:
                attn_K_loss = sum([sum([
                               torch.norm(i[bidx,:,:ilen] - j[bidx,:,:ilen].detach())
                               for bidx, ilen in enumerate(input_lens)
                               ])
                               for lidx, (i, j) in enumerate(zip(attn_K, ori_attn_K)) 
                               if lidx in eval(hparams.high_attn_range)]) * hparams.attn_K_loss_weight * 1e-4
        last_hid_loss = torch.tensor(0).cuda()
        
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        max_probs = torch.max(log_probs, dim = 2)[0]
        max_prob = torch.exp((max_probs * mask).sum(1) / target_ids.size(0)).mean().item()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta_mlp) / torch.norm(target_init_mlp) ** 2 + torch.norm(delta_attn) / torch.norm(target_init_attn) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay + attn_W_loss + attn_K_loss + last_hid_loss
        prob = torch.exp(-nll_loss_each).mean().item()
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} + {np.round(attn_W_loss.item(), 3)} + {np.round(attn_K_loss.item(), 3)} + {np.round(last_hid_loss.item(), 3)} "
            f"loss_type: {hparams.loss_type} is_llama_tok:{is_llama}"
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        
        if loss < 5e-2:
            break
        if max_prob == prob:
            nll_loss_factor = 0.1 * hparams.nll_loss_factor
            if kl_loss <= 0.01:
                break
        else:
            nll_loss_factor = hparams.nll_loss_factor
            
        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init_mlp.norm()
        if delta_mlp.norm() > max_norm:
            with torch.no_grad():
                delta_mlp[...] = delta_mlp * max_norm / delta_mlp.norm()

        max_norm = hparams.clamp_norm_factor * target_init_attn.norm()
        if delta_attn.norm() > max_norm:
            with torch.no_grad():
                delta_attn[...] = delta_attn * max_norm / delta_attn.norm()


    target_attn = target_init_attn + delta_attn
    target_mlp = target_init_mlp + delta_mlp
    print(
        f"[ATTN]: Init norm {target_init_attn.norm()} | Delta norm {delta_attn.norm()} | Target norm {target_attn.norm()}",
        f"[MLP]: Init norm {target_init_mlp.norm()} | Delta norm {delta_mlp.norm()} | Target norm {target_mlp.norm()}",
    )

    return target_attn, target_mlp


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: PMETHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1]) #这里好像是个bug，target_ids[:-1]是空的  tok.decode(target_ids[0]) 
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    nll_loss_factor = hparams.nll_loss_factor
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        max_probs = torch.max(log_probs, dim = 2)[0]
        max_prob = torch.exp((max_probs * mask).sum(1) / target_ids.size(0)).mean().item()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        prob = torch.exp(-nll_loss_each).mean().item()
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{prob}"
        )
        if loss < 5e-2:
            break
        if max_prob == prob:
            nll_loss_factor = 0.1 * hparams.nll_loss_factor
            if kl_loss / hparams.kl_factor < 0.01:
                break
        else:
            nll_loss_factor = hparams.nll_loss_factor
            
        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target

def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()

def get_modules_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_templates: List[str],
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_templates,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input1, l_input2 = repr_tools.get_inputs_at_word_tokens(
            track="in", subtoken=subtoken, **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input1.detach(), l_input2.detach()

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
