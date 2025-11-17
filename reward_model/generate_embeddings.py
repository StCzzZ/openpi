import torch
import dataclasses
import tyro
from transformers import AutoTokenizer, AutoModel
import h5py
import numpy as np
from tqdm import tqdm

from reward_model.util import dino_load_image, mean_pooling
from third_party.dinov2.model import vit_base


PROMPT = "Put the items in the pot."


@dataclasses.dataclass
class Args:
    data_path: str
    output_path: str
    ckpt_path: str
    batch_size: int = 32


def load_model(args: Args, device: torch.device):
    # load pretrained DINOv2 encoder
    dino_encoder = vit_base(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    ).to(device)
    dino_encoder.load_state_dict(torch.load(args.ckpt_path), strict=True)

    # load pretrained all-MiniLM-L12-v2 encoder following ReWIND
    minilm_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    )
    minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
        device
    )

    return dino_encoder, minilm_tokenizer, minilm_model


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_encoder, minilm_tokenizer, minilm_model = load_model(args, device)
    embedding_output_file = h5py.File(args.output_path, 'w')

    # Extract task prompt embedding
    with torch.no_grad():
        encoded_input = minilm_tokenizer(
                [PROMPT], padding=False, truncation=True, return_tensors="pt"
            ).to(device)
        model_output = minilm_model(**encoded_input)
        minlm_task_embedding = (
            mean_pooling(model_output, encoded_input["attention_mask"])
            .cpu()
            .detach()
            .numpy()
        )

    with h5py.File(args.data_path, 'r') as f:
        for key in tqdm(f, desc="Processing episodes"):
            if key not in embedding_output_file:
                embedding_output_file.create_group(key)
            episode = f[key]
            # ================ Extract DINO embeddings ================
            dino_embeddings = []
            for step in tqdm(range(0, episode["action"].shape[0], args.batch_size), desc="Extracting DINO embeddings"):
                batch_images = episode['side_cam'][step:min(step+args.batch_size, episode["action"].shape[0])]
                batch_images = [dino_load_image(img) for img in batch_images]
                batch_images = torch.cat(batch_images, dim=0)
                batch_images = batch_images.to(device)
                with torch.no_grad():
                    batch_embeddings = dino_encoder(batch_images)
                    batch_embeddings = batch_embeddings.cpu().detach().numpy()
                    dino_embeddings.append(batch_embeddings)
                
            dino_embeddings = np.concatenate(dino_embeddings, axis=0)
            embedding_output_file[key]['dino_embeddings'] = dino_embeddings
            # ================ Extract DINO embeddings ================
            
            embedding_output_file[key]['minlm_task_embedding'] = minlm_task_embedding
            embedding_output_file[key]['progress'] = np.arange(1, episode["action"].shape[0]+1) / episode["action"].shape[0]

    embedding_output_file.close()



if __name__ == '__main__':
    main(tyro.cli(Args))