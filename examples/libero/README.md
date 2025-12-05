# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero



# To run with glx for Mujoco instead (use this if you have egl errors):

CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glx python examples/libero/main.py 
# CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=glx python examples/libero/run_parallel.py --args.num-envs 20 --args.task_suite_name libero_90
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glx python examples/libero/run_parallel.py --args.num-envs 20 --args.task_suite_name libero_90

CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=glx python examples/libero/run_parallel_with_visualization.py --args.num-envs 5 --args.task_suite_name libero_spatial

CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=glx python examples/libero/run_parallel_with_hil.py --args.num-envs 5 --args.task_suite_name libero_spatial



# Run the simulation
CUDA_VISIBLE_DEVICES=2 python examples/libero/main.py --args.task_suite_name libero_90
CUDA_VISIBLE_DEVICES=2 python examples/libero/run_parallel.py --args.num-envs 20  --args.task_suite_name libero_90 # parallel policy rollout! 

CUDA_VISIBLE_DEVICES=2 python examples/libero/run_parallel_with_visualization.py --args.num-envs 5  --args.task_suite_name libero_90 # parallel policy rollout! 
```



Terminal window 2:

```bash
# Run the server
# export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"
CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir /data/yijin/openpi/checkpoints/pi05_libero # single env
CUDA_VISIBLE_DEVICES=6 uv run scripts/serve_policy.py --env LIBERO --is-batched policy:checkpoint --policy.config pi05_libero --policy.dir /data/yijin/openpi/checkpoints/pi05_libero # parallel env
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
