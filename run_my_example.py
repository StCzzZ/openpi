from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import my_policy

config = _config.get_config("pi05_flexiv_value")
checkpoint_dir = "/home/wenye/project/openpi/checkpoints/pi05_flexiv_value/kitchen_100_discrete_value/30000"

policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = my_policy.make_my_example()
action_chunk = policy.infer(example)["actions"]
value_chunk = policy.infer(example)["value"]
print(action_chunk.shape, action_chunk)
print(value_chunk.shape, value_chunk)