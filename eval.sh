source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate openpi
export PYTHONPATH=/home/yuwenye/project/openpi:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

uv run reward_model/eval_reward.py \
--dataset_path /data/yuwenye/reward_modeling/data/original/1120_novel_kitchen.hdf5 \
--output_path /data/yuwenye/reward_modeling/data/sarm/1201_novel_kitchen_cont_value_head \
--reward_model_path /data/yuwenye/reward_modeling/output/pi0_internal_rewind/reward_model_99.pt \
--prompt "Put the items in the pot." \
--batch_size 16 \
policy:checkpoint \
--policy.config pi05_flexiv_value \
--policy.dir /home/yuwenye/project/openpi/checkpoints/pi05_flexiv_value/kitchen_100_cont_value/39999
