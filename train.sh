source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate openpi

export WANDB_API_KEY=956bd553be1193c6fdfce758ad1eded5190eac56
export PYTHONPATH=/home/yuwenye/project/openpi:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

uv run reward_model/train_reward.py \
--dataset_path /data/yuwenye/reward_modeling/data/sarm/1122_kitchen_test_dinominlm_embeddings.hdf5 \
--output_path /data/yuwenye/reward_modeling/output \
--num_stages 7 \
--max_seq_len 32 \
--backbone dinov2_minilm \
--batch_size 256 \
--learning_rate 1e-4 \
--num_epochs 100 \
--num_workers 4 \
--clip-grad \
--video-rewind \
--device cuda \
--eval_every 10 \
--save_every 20 \
--wandb_project reward_model \
--exp_name dinominilm_rewind
