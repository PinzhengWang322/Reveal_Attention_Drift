import os.path
import sys
import json
import random
sys.path.append('..')
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    PMETHyperParams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset

from easyeditor.dsets import AttributeSnippets, get_tfidf_vectorizer
# from easyeditor.dataset import quick_process
import argparse
from transformers import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--start_id', default=0, type=int)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None, type=str)
    parser.add_argument('--edit_layer', default=None, type=int)
    parser.add_argument('--ppl_text', default=None, type=str)
    parser.add_argument('--aft_edit_text', action="store_true")
    parser.add_argument('--noinv', action="store_true")
    parser.add_argument('--select_head', action="store_true")
    parser.add_argument('--text_generation', action="store_true")
    parser.add_argument('--snips_dir', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--topk', default=3, type=int)
    parser.add_argument('--label_type', default="truth", type=str)
    parser.add_argument('--choose_type', default=None, type=str)
    parser.add_argument('--attn_W_loss_weight', default=0, type=float)
    parser.add_argument('--attnW_droupout', default=None, type=float)
    parser.add_argument('--attn_K_loss_weight', default=0, type=float)
    parser.add_argument('--last_hid_restrain_layers', default=None, type=str)
    parser.add_argument('--last_hid_restrain_weight', default=0, type=float)
    parser.add_argument('--lr', default=5e-1, type=float)
    parser.add_argument('--high_attn_range', default=None, type=str)
    parser.add_argument('--v_num_grad_steps', default=20, type=int)
    parser.add_argument('--loss_type', default="KL", type=str)
    parser.add_argument('--kl_factor', default=None, type=float)
    parser.add_argument('--hinge', action="store_true")
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--part', default=0, type=int)
    args = parser.parse_args()
    
    set_seed(42)
    print("-" * 100)
    print("args:")
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)
    random.seed(1234)
    
    os.makedirs(args.metrics_save_dir, exist_ok=True)
    if args.cache_dir: os.makedirs(args.cache_dir, exist_ok=True)
    
    if args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'PMET':
        editing_hparams =  PMETHyperParams
    else:
        raise NotImplementedError

    requests = json.load(open(args.data_path))
    part_num = len(requests) // args.gpu_num + 1
    requests = requests[args.part*part_num:(args.part+1)*part_num]
    
    print(os.path.join(args.metrics_save_dir, 
                                      f'{args.editing_method}_results_vstep{args.v_num_grad_steps}_{args.part*part_num}_{(args.part+1)*part_num}.json'))
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    if args.edit_layer: hparams.layers = [args.edit_layer]
    if args.noinv: hparams.mom2_adjustment = False
    hparams.attn_W_loss_weight = args.attn_W_loss_weight
    hparams.attn_K_loss_weight = args.attn_K_loss_weight
    if args.v_num_grad_steps is not None: hparams.v_num_grad_steps = args.v_num_grad_steps
    if args.attnW_droupout is not None: hparams.attnW_droupout = args.attnW_droupout
    if args.kl_factor is not None: hparams.kl_factor = args.kl_factor
    if args.hinge: hparams.hinge = True
    hparams.last_hid_restrain_weight = args.last_hid_restrain_weight
    hparams.last_hid_restrain_layers = args.last_hid_restrain_layers
    hparams.high_attn_range = args.high_attn_range
    hparams.v_lr = args.lr
    hparams.loss_type = args.loss_type
    editor = BaseEditor.from_hparams(hparams)
    
    snips = AttributeSnippets(args.snips_dir)
    vec = get_tfidf_vectorizer(args.snips_dir) if args.text_generation else None
    ppl_text = [i['Text'] for i in json.load(open(args.ppl_text))] if args.ppl_text else []

    model_name = os.path.basename(args.hparams_dir).split('.')[0]
    data_name = os.path.basename(args.data_path).split('.')[0]
    ppl_text_name = os.path.basename(args.ppl_text).split('.')[0]
    metrics = editor.edit(
        requests=requests,
        text_generation = args.text_generation,
        ppl_text = ppl_text,
        aft_edit_text = args.aft_edit_text,
        snips = snips,
        vec = vec,
        cache_path = os.path.join(args.cache_dir, 
                f"pre_{model_name}_None_{data_name}_{args.start_id}_{args.ds_size}_{ppl_text_name}.json") 
                if args.cache_dir else None,
        select_head = args.select_head,
        topk = args.topk,
        label_type = args.label_type,
        choose_type=args.choose_type,
    )

    json.dump(metrics, open(os.path.join(args.metrics_save_dir, 
                                         'result.json'), 'w'), indent=4)
