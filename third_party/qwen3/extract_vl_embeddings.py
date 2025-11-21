from modelscope import Qwen3VLMoeForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info

# # default: Load the model on the available device(s)
# model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

# For batched inference
processor.tokenizer.padding_side = 'left'

# Batched messages
messages = []

for i in range(10):
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"/data/yuwenye/reward_modeling/data/qwen/1113_kitchen/episode_0/{i}.png"},
            {"type": "text", "text": "Put the items in the pot."},
        ],
    }]
    messages.append(message)

# Process input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, padding=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    do_resize=False,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Run model and get hidden states
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

# 4. Get the last layer's Hidden States
# shape: (batch_size, sequence_length, hidden_size)
last_hidden_state = outputs.hidden_states[-1]

# 5. Separate image features
# Since the input contains text and image tokens, you need to find the position of image tokens based on input_ids
# Qwen3-VL images are usually wrapped by special tokens or directly replaced with visual tokens
# A simple approach is to see which positions in input_ids are image placeholders (depends on tokenizer implementation)

import pdb; pdb.set_trace()
# print(f"Last Hidden State Shape: {last_hidden_state.shape}")