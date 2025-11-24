source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate qwen
export PYTHONPATH=/home/yuwenye/project/openpi:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=1

python reward_model/eval_reward.py \
--dataset_path /data/yuwenye/reward_modeling/data/original/1113_kitchen.hdf5 \
--output_path /data/yuwenye/reward_modeling/data/sarm/1124_rollout_original_kitchen_qwen3vl_rewind_reward_eval_relative_pred \
--reward_model_path /data/yuwenye/reward_modeling/output/qwen3vl_rewind/reward_model_99.pt \
# --dino-ckpt-path /data/yuwenye/reward_modeling/dinov2_vitb14_reg4_pretrain.pth \
