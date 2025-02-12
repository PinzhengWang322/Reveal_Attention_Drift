import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import numpy as np


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

from ..evaluate import *
from ..evaluate.my_evaluate_human_eval import compute_myedit_quality




class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        if 'gpt' in self.model_name.lower():
            if self.model_name == "gpt2-xl": 
                model_path = "/path/to/hf_models/gpt2-xl"
            if self.model_name == "gptj-6B": 
                model_path = "/path/to/hf_models/gpt-j-6b"
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto' if hparams.model_parallel else None)
            self.tok = GPT2Tokenizer.from_pretrained(model_path)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'llama13b' in self.model_name.lower():
            model_path = "/path/to/hf_models/Llama-2-13b-hf"
            self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=None)
            self.tok = AutoTokenizer.from_pretrained(model_path)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'TinyLlama' in self.model_name:
            model_path = "/path/to/hf_models/TinyLlama-1.1B-pretrain-3T"
            self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map=None)
            self.tok = AutoTokenizer.from_pretrained(model_path)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'llama' in self.model_name.lower():
            # model_path = "/path/to/hf_models/llama-7b-hf"
            model_path = "/path/to/hf_models/Meta-Llama-3-8B"
            self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=None)
            self.tok = AutoTokenizer.from_pretrained(model_path)
            self.tok.pad_token_id = self.tok.eos_token_id
        else:
            raise KeyError(f"Unknown model {self.model_name}")
        
        self.model.to("cuda")
        self.hparams = hparams

    def edit(self, 
             requests,
             text_generation = True,
             ppl_text = None,
             aft_edit_text = False,
             snips = None,
             vec = None,
             cache_path = None,
             select_head = False,
             topk = 3,
             label_type = None,
             choose_type = None):
        
        all_metrics = []
        cache_metrics = {}
        load_flag = True

        if cache_path and os.path.exists(cache_path):
            loaded_metrics = json.load(open(cache_path))
        else: load_flag = False
            
        for i, request in tqdm(enumerate(requests), total=len(requests)):
            detail_time = {}
            self.model.eval()
            # if load_flag:
            #     pre, pre_time = loaded_metrics[str(i)], 0
            # else:
            #     pre, pre_time = compute_myedit_quality(self.model, 
            #                                         self.tok,
            #                                         request,
            #                                         snips = snips,
            #                                         vec = vec,
            #                                         text_generation = text_generation,
            #                                         ppl_text = ppl_text,
            #                                         aft_edit_text = aft_edit_text)
            #     cache_metrics[i] = pre
            #     print("pre metric:\n", json.dumps(pre, indent=4))
                
            pre, pre_time = compute_myedit_quality(self.model, 
                                                    self.tok,
                                                    request,
                                                    snips = snips,
                                                    vec = vec,
                                                    text_generation = text_generation,
                                                    ppl_text = ppl_text,
                                                    aft_edit_text = aft_edit_text)
            cache_metrics[i] = pre
            print("pre metric:\n", json.dumps(pre, indent=4))
            # self.model.train()
            
            detail_time['pre_eval_time'] = pre_time
            start_time = time.time()
            edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request["requested_rewrite"]],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=True,
                    train_ds=None,
                    select_head = select_head,
                    topk = topk,
                    label_type = label_type,
                    choose_type = choose_type,
                    idx = i,
                )
            end_time = time.time()

            detail_time['edit_time'] = end_time - start_time
            edited_model.eval()
            post, post_time = compute_myedit_quality(edited_model, 
                                                    self.tok,
                                                    request,
                                                    snips = snips,
                                                    vec = vec,
                                                    text_generation = text_generation,
                                                    ppl_text = ppl_text,
                                                    aft_edit_text = aft_edit_text)
            # print("pre metric:\n", json.dumps(pre, indent=4))
            print("post metric:\n", json.dumps(post, indent=4))
            # edited_model.train()
            detail_time['post_eval_time'] = post_time
            all_metrics.append({"id":i, "pre": pre, "post": post})

            if i % 11 == 0: print("detail_time", json.dumps(detail_time, indent=4))

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda")

        if cache_path and not load_flag:
            with open(cache_path, 'w') as f:
                f.write(json.dumps(cache_metrics, indent=4))

        return all_metrics