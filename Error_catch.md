# if pyenv and venv co-exist you may need:

```bash
conda deactivate
export PYENV_VERSION=system
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
source examples/libero/.venv/bin/activate
which python


export CUDA_VISIBLE_DEVICES=7
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


```

# if you meet matplotlib tkinter problems you may need: 

```bash
tplotlib/backends/_backend_tk.py", line 24, in <module>
    from . import _tkagg
AttributeError: module '_tkinter' has no attribute '__file__'

```

```bash
pip install PyQt5
```

then use

```bash
import matplotlib
matplotlib.use('Qt5Agg')  # change to Qt5 backend
import matplotlib.pyplot as plt
```




# IP Error: server didn't set IP forwarding rules：

```bash
ssh -L 127.0.0.1:18002:127.0.0.1:18002 user@server_ip
```

to establish ssh tunnel and bind local_host:18002 to the remote server:18002, use 18002 here cause it doesn't need sudo permissions

and use "localhost:18002" to connect to the remote server

# To use "convert data to lerobot" you need

```bash
uv pip install tensorflow(select your own version to match protobuf) tensorflow_dataset
```

# To enable the GPU-drived tensorflow (individually): 

```bash
export LD_LIBRARY_PATH=/opt/cudnn-8.6/lib:/usr/local/cuda-11.5/lib64:/usr/local/cuda-11.5/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

