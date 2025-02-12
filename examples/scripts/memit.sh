# ADR method
CUDA_VISIBLE_DEVICES=7 python run_knowedit.py \
    --editing_method=MEMIT \
    --hparams_dir=../hparams/MEMIT/gpt-j-6B.yaml \
    --data_path=../data/merged_counterfact.json \
    --datatype='merged' \
    --metrics_save_dir=./results/gpt-j-6B/memit-ADR \
    --ds_size=10000 \
    --ppl_text=../data/ppl_test/Corpus-50.json \
    --text_generation \
    --aft_edit_text \
    --snips_dir=../data/status \
    --cache_dir=./cache \
    --v_num_grad_steps=20 \
    --attn_W_loss_weight=50 \
    --hinge

# origin method
CUDA_VISIBLE_DEVICES=6 python run_knowedit.py \
    --editing_method=MEMIT \
    --hparams_dir=../hparams/MEMIT/gpt-j-6B.yaml \
    --data_path=../data/merged_counterfact.json \
    --datatype='merged' \
    --metrics_save_dir=./results/gpt-j-6B/memit-ADR \
    --ds_size=10000 \
    --ppl_text=../data/ppl_test/Corpus-50.json \
    --text_generation \
    --aft_edit_text \
    --snips_dir=../data/status \
    --cache_dir=./cache \
    --v_num_grad_steps=20 \
    --attn_W_loss_weight=0 \
    --hinge

# attn_W_loss_weight is multiplied by 1e-4 in the code