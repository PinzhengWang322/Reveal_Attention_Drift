# ADR method
CUDA_VISIBLE_DEVICES=6 python run_knowedit.py \
    --editing_method=PMET \
    --hparams_dir=../hparams/PMET/gpt-j-6B.yaml \
    --data_path=../data/merged_counterfact.json \
    --datatype='merged' \
    --metrics_save_dir=./results/gpt-j-6B/pmet-ADR \
    --ds_size=10000 \
    --ppl_text=../data/ppl_test/Corpus-50.json \
    --text_generation \
    --snips_dir=../data/status \
    --aft_edit_text \
    --cache_dir=./cache \
    --v_num_grad_steps=40 \
    --attn_W_loss_weight=800 \
    --hinge

# origin method
CUDA_VISIBLE_DEVICES=7 python run_knowedit.py \
    --editing_method=PMET \
    --hparams_dir=../hparams/PMET/gpt-j-6B.yaml \
    --data_path=../data/merged_counterfact.json \
    --datatype='merged' \
    --metrics_save_dir=./results/gpt-j-6B/pmet-ADR \
    --ds_size=10000 \
    --ppl_text=../data/ppl_test/Corpus-50.json \
    --text_generation \
    --snips_dir=../data/status \
    --aft_edit_text \
    --cache_dir=./cache \
    --v_num_grad_steps=20 \
    --attn_W_loss_weight=0 \
    --hinge

# attn_W_loss_weight is multiplied by 1e-4 in the code

