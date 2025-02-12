# ADR method
CUDA_VISIBLE_DEVICES=7 python run_knowedit.py \
--editing_method=ROME \
--hparams_dir=../hparams/ROME/gpt-j-6B.yaml \
--data_path=../data/merged_counterfact.json \
--datatype='merged_counterfact' \
--metrics_save_dir=./results/gpt-j-6B/rome-ADR \
--text_generation \
--ds_size=10000 \
--ppl_text=../data/ppl_test/Corpus-50.json \
--aft_edit_text \
--snips_dir=../data/status \
--cache_dir=./cache \
--v_num_grad_steps=80 \
--attn_W_loss_weight=100 \
--hinge 2>&1 | tee ADR.log

# origin method
CUDA_VISIBLE_DEVICES=6 python run_knowedit.py \
--editing_method=ROME \
--hparams_dir=../hparams/ROME/gpt-j-6B.yaml \
--data_path=../data/merged_counterfact.json \
--datatype='merged_counterfact' \
--metrics_save_dir=./results/gpt-j-6B/rome-ADR \
--text_generation \
--ds_size=10000 \
--ppl_text=../data/ppl_test/Corpus-50.json \
--aft_edit_text \
--snips_dir=../data/status \
--cache_dir=./cache \
--v_num_grad_steps=20 \
--attn_W_loss_weight=0 \
--hinge 2>&1 | tee base.log

# attn_W_loss_weight is multiplied by 1e-4 in the code