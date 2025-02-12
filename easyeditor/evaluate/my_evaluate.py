import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch

import nltk
import scipy
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_generation_quality, 
    PPL,
    kl_loc_loss,
    es_sent
)
import time


def compute_myedit_quality(model, 
                           tok, 
                           request, 
                           snips = None,
                           vec = None,
                           text_generation = False,
                           ppl_text = None,
                           aft_edit_text = False):
    res = {}
    eval_time = {}
    run_types = [""] if not aft_edit_text else ["", "_aft_edit_text"]
    edit_text = f"{request['requested_rewrite']['prompt']} {request['requested_rewrite']['target_new']}."
    target_types = ["right", "wrong"]
    is_llama2 = ("LlamaForCausalLM" in str(type(model)) and (len(model.model.layers) == 22 or
                                                             len(model.model.layers) == 40)) # For tiny llama and llama 13b
    print("is_llama2:", is_llama2)
    for target_type in target_types:
        for run_type in run_types:
            for key in request.keys():
                if key in ['requested_rewrite', 'generation']: continue
                if run_type == "_aft_edit_text" and key.split('-')[0] not in ['l', 'o']: continue
                start = time.time()
                if target_type == "right":
                    if key.split('-')[0] == 'o':
                        targets = [request[key]['right']]
                    else: 
                        targets = request[key]["targets"]
                elif target_type == "wrong":
                    if key == "rewrite" or key.split('-')[0] == 'p':
                        targets = [request['requested_rewrite']['target_true'] for _ in request[key]["targets"]]
                    elif key.split('-')[0] == 'l':
                        targets = [request['requested_rewrite']['target_new'] for _ in request[key]["targets"]]
                    elif key.split('-')[0] == 'o':
                        targets = [request[key]['wrong']]

                prompts = request[key]["prompts"] if isinstance(request[key]["prompts"], list) else [request[key]["prompts"]]
                if prompts == []: continue
                if run_type == "_aft_edit_text": 
                    prompts = [f"{edit_text} {p}" for p in prompts]

                assert len(prompts) == len(targets)

                prefix_lens = [len(n) for n in tok(prompts)["input_ids"]]
                tgtlens = [len(tok(f" {n}" if not is_llama2 else n, add_special_tokens = False)["input_ids"]) for n in targets]

                input_tok = tok([f"{prefix} {suffix}" for prefix, suffix in zip(prompts, targets)]
                                , padding=True, return_tensors="pt").to('cuda')
                tgt_tok = tok([f" {t}" if not is_llama2 else t for t in targets], add_special_tokens = False).input_ids
                with torch.no_grad():
                    out = model(**input_tok)
                    if type(out) == torch.Tensor:
                        logits = out
                    else:
                        logits = out.logits

                results = np.zeros((logits.size(0),), dtype=np.float32)
                for i, tgtlen in enumerate(tgtlens):
                    for j in range(tgtlen):
                        cur_tok = tgt_tok[i][j]
                        results[i] += -torch.nn.functional.log_softmax(
                            logits[i, prefix_lens[i] + j - 1, :], dim=0
                        )[cur_tok].item()
                    results[i] /= tgtlen
                res[f"{key}{run_type}-{target_type}"] = [r.item() for r in results]
                end = time.time()
                eval_time[f"{key}{run_type}"] = end - start
                # if key.startswith('o'): import pdb; pdb.set_trace()

    
    if "generation" in request and text_generation:
        start = time.time()
        rel_id = request["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][request["requested_rewrite"]["target_new_id"]]]

        ret = test_generation(model, tok, 
                              prefixes = request["generation"]["prompts"], 
                              consistency_texts = consistency_texts,
                              essence_texts = ppl_text,
                              vec = vec)
        res.update(ret)
        end = time.time()
        eval_time['generation'] = end - start
    
    return res, eval_time

def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"ppl_score": ppl})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()

def perplexity(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        text: str,
        max_input_length: int = None,
    ):
        """
        Computes perplexity of a piece of text, measured on a reference model.
        Text is truncated to max_input_length tokens.
        """

        inputs = tok(
            [text], return_tensors="pt", max_length=max_input_length, truncation=True
        ).to("cuda")

        logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
        log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

        # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
        return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()

def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]
    torch.cuda.empty_cache()
    return txt

