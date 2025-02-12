from functools import wraps, partial
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from .icl.analysis.attentioner_for_attribution import AttentionAdapter

def get_salinecies_score(wrap_model, attentionermanger, inp, label):
    for p in wrap_model.parameters():
        p.requires_grad = False
    attentionermanger.zero_grad()
    output = wrap_model(**inp)
    if isinstance(label, int):
        loss = F.cross_entropy(output['ori_logits'], 
                            torch.tensor([label]).to(output['ori_logits'].device))
    else:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output['ori_logits'][0,:], label)
        
    loss.backward()
    saliencies =  attentionermanger.grad(use_abs=True)
    return saliencies

def choose_grad_head(saliencies, topk = 3, choose_type = None, 
                     layers_range=None, subject_range=None):
    res = {}
    if choose_type.startswith("token_last_subject_"):
        subject_type = choose_type.split('_')[-1]
        for layer in layers_range:
            if subject_type == "all":
                saliency = saliencies[layer][:,:,-1:,range(*subject_range)].sum(dim=-1).sum(dim=-1)
            elif subject_type == "last":
                saliency = saliencies[layer][:,:,-1:,subject_range[-1]].sum(dim=-1)
            elif subject_type == "lastsmall":
                saliency = -saliencies[layer][:,:,-1:,subject_range[-1]].sum(dim=-1)
            elif subject_type == "allsmall":
                saliency = -saliencies[layer][:,:,-1:,range(*subject_range)].sum(dim=-1).sum(dim=-1)
            else:
                raise NotImplementedError
            sorted_indices = torch.argsort(saliency, dim=1, descending=True)
            res[layer] = {}
            res[layer]['indices'] = sorted_indices.squeeze().cpu().tolist()[:topk]
            res[layer]['scores'] = saliency[0][res[layer]['indices']].cpu().tolist()
    elif choose_type == "random":
        for layer in layers_range:
            saliency = saliencies[layer][:,:,-1:,subject_range].sum(dim=-1).sum(dim=-1)
            sorted_indices = torch.argsort(saliency, dim=1, descending=True).squeeze().cpu().tolist()
            random_indices = random.sample(sorted_indices, topk)
            res[layer] = {}
            res[layer]['indices'] = random_indices
    else:
        raise NotImplementedError
    
    return res

    
def gpt2_attn_with_detached_heads(self, query, key, value, attention_mask=None, head_mask=None, 
                                  topk=3, saliencies=None, layer=None, partial_choose_func=None):
    choosed_layers = partial_choose_func(saliencies, topk=topk)
    if layer in choosed_layers:
        not_detached_lst = choosed_layers[layer]['indices']
    else:
        not_detached_lst = None
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights,
                                   self.masked_bias.to(attn_weights.dtype))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask
        
    if not_detached_lst is None:
        new_attn_weights = attn_weights
    else:
        mask = torch.zeros_like(attn_weights)
        mask[:,not_detached_lst,:,:] = 1
        new_attn_weights = attn_weights * mask + attn_weights.detach() * (1 - mask)
    attn_output = torch.matmul(new_attn_weights, value)
    
#     attn_output = torch.matmul(attn_weights, value)
#     if not_detached_lst is None:
#         new_attn_output = attn_output
#     else:
#         mask = torch.zeros_like(attn_output)
#         mask[:,not_detached_lst,:,:] = 1
#         new_attn_output = attn_output * mask + attn_output.detach() * (1 - mask)

#     return neattn_output, attn_weights
    return attn_output, new_attn_weights

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

def turn_to_grad_pune_attn(model, saliencies, partial_choose_func, topk=3):
    for i, layer in enumerate(model.transformer.h):
        layer.attn._attn = partial(gpt2_attn_with_detached_heads, layer.attn,
                                topk=topk, saliencies=saliencies, layer=i,
                                partial_choose_func = partial_choose_func)
        
def turn_to_origin_attn(model):
    for i, layer in enumerate(model.transformer.h):
        layer.attn._attn = partial(ori_attn, layer.attn)