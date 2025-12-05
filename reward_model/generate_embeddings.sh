source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate openpi

export WANDB_API_KEY=956bd553be1193c6fdfce758ad1eded5190eac56
export PYTHONPATH=/home/yuwenye/project/openpi:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1

uv run reward_model/generate_embeddings.py \
    --data_path /data/yuwenye/reward_modeling/data/original/1113_kitchen.hdf5 \
    --output_path /data/yuwenye/reward_modeling/data/sarm/1127_kitchen_test_pi0_internal_embeddings.hdf5 \
    --backbone pi0_internal \
    --prompt "Put the items in the pot." \
    --batch_size 64