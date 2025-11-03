from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import my_policy

config = _config.get_config("pi05_flexiv")
checkpoint_dir = "/home/yuwenye/project/openpi/checkpoints/pi05_flexiv/my_data/29999"

policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = my_policy.make_my_example()
action_chunk = policy.infer(example)["actions"]
import pdb; pdb.set_trace()
print(action_chunk)