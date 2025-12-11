from __future__ import annotations

import abc
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from reward_model.config import RewardBackboneConfig
from reward_model.util import dino_load_image, mean_pooling
from third_party.dinov2.model import vit_base
try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - optional dependency
    pass

try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except ImportError:  # pragma: no cover - optional dependency
    _policy_config = None  # type: ignore[assignment]
    _config = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractorInitParams:
    device: torch.device


class BaseEmbeddingExtractor(abc.ABC):
    """Base interface for extracting per-frame embeddings from raw video frames."""

    def __init__(self, backbone: RewardBackboneConfig, params: ExtractorInitParams):
        self.backbone = backbone
        self.params = params

    @abc.abstractmethod
    def extract_visual_embeddings(
        self,
        frames: np.ndarray,
        batch_size: int,
        prompt: str,
        other_frames: Optional[np.ndarray] = None,
        proprio_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert raw frames into visual embeddings."""

    def get_language_embedding(self, prompt: str) -> Optional[np.ndarray]:
        """Return a sentence-level embedding for ``prompt`` when available."""
        return None


class DinoMinilmExtractor(BaseEmbeddingExtractor):
    """Embedding extractor that follows the original DINO + MiniLM pipeline."""

    def __init__(self, backbone: RewardBackboneConfig, params: ExtractorInitParams):
        super().__init__(backbone, params)
        visual_config = backbone.visual_config
        language_config = backbone.language_config
        if visual_config.kind != "dinov2" or language_config is None or language_config.kind != "minilm":
            raise ValueError("DINO+MiniLM configuration is incomplete.")
        visual_params = visual_config.params
        language_params = language_config.params

        self.device = params.device
        self.dino_encoder = vit_base(
            img_size=int(visual_params.get("img_size", 518)),
            patch_size=int(visual_params.get("patch_size", 14)),
            init_values=float(visual_params.get("init_values", 1.0)),
            ffn_layer=str(visual_params.get("ffn_layer", "mlp")),
            block_chunks=int(visual_params.get("block_chunks", 0)),
            num_register_tokens=int(visual_params.get("num_register_tokens", 4)),
            interpolate_antialias=bool(visual_params.get("interpolate_antialias", True)),
            interpolate_offset=float(visual_params.get("interpolate_offset", 0.0)),
        ).to(self.device)
        assert "dino_ckpt_path" in visual_params, "DINO checkpoint path must be provided for the DINO+MiniLM extractor."
        self.dino_encoder.load_state_dict(
            torch.load(visual_params.get("dino_ckpt_path"), map_location=self.device),
            strict=True,
        )
        self.dino_encoder.eval()

        model_name = str(language_params.get("model_name", "sentence-transformers/all-MiniLM-L12-v2"))
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.minilm_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.minilm_model.eval()
        self._language_cache: Dict[str, np.ndarray] = {}

    def extract_visual_embeddings(
        self,
        frames: np.ndarray,
        batch_size: int,
        prompt: str,
        other_frames: Optional[np.ndarray] = None,
        proprio_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del prompt  # Unused but kept for API compatibility.
        embeddings: List[np.ndarray] = []
        total_steps = frames.shape[0]
        for start in range(0, total_steps, batch_size):
            end = min(start + batch_size, total_steps)
            batch_images = frames[start:end]
            batch_tensors = torch.cat([dino_load_image(img) for img in batch_images], dim=0).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.dino_encoder(batch_tensors).cpu().detach().numpy()
            embeddings.append(batch_embeddings)
        return np.concatenate(embeddings, axis=0)

    def get_language_embedding(self, prompt: str) -> np.ndarray:
        if prompt in self._language_cache:
            return self._language_cache[prompt]
        encoded_input = self.minilm_tokenizer(
            [prompt], padding=False, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.minilm_model(**encoded_input)
            prompt_embedding = (
                mean_pooling(model_output, encoded_input["attention_mask"]).cpu().detach().numpy()
            )
        self._language_cache[prompt] = prompt_embedding
        return prompt_embedding


class Qwen3VLExtractor(BaseEmbeddingExtractor):
    """Embedding extractor backed by Qwen3-VL for end-to-end VLM embeddings."""

    DEFAULT_PROGRESS_PROMPT = (
        "The preceding images show a full successful demonstration of the task "
        "from start (0.0) to completion (1.0). The numbered target frames that "
        "follow belong to a new episode. Provide progress values for each target "
        "frame in the same order. Respond with only comma-separated decimal numbers "
        "between 0 and 1 and do not include any reasoning, explanation, or thinking process."
    )

    def __init__(self, backbone: RewardBackboneConfig, params: ExtractorInitParams):
        super().__init__(backbone, params)
        visual_config = backbone.visual_config
        if visual_config.kind != "qwen3_vl":
            raise ValueError("Qwen3-VL configuration is missing.")
        qwen_params = visual_config.params
        try:
            from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration, snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "modelscope is required for Qwen3-VL extraction. Please install it before proceeding."
            ) from exc

        self.device = params.device
        model_name = str(qwen_params.get("model_name", "Qwen/Qwen3-VL-8B-Thinking"))
        model_dir = snapshot_download(model_name)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            attn_implementation=qwen_params.get("attn_implementation", "flash_attention_2"),
            device_map=qwen_params.get("device_map", "auto"),
        )
        self._default_progress_prompt = str(qwen_params.get("progress_prompt", self.DEFAULT_PROGRESS_PROMPT))
        self._progress_max_new_tokens = int(qwen_params.get("progress_max_new_tokens", 32))

    def extract_visual_embeddings(
        self,
        frames: np.ndarray,
        batch_size: int,
        prompt: str,
        other_frames: Optional[np.ndarray] = None,
        proprio_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        total_steps = frames.shape[0]
        for start in range(0, total_steps, batch_size):
            end = min(start + batch_size, total_steps)
            batch_frames = frames[start:end]
            messages = self._build_messages(batch_frames, prompt)
            text_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                padding=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            processor_outputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )

            model_inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in processor_outputs.items()
            }
            
            with torch.no_grad():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden_state = outputs.hidden_states[-1]
            last_hidden_state = last_hidden_state.mean(dim=1)
            pooled_embeddings = last_hidden_state.float().cpu().detach().numpy()
            embeddings.append(pooled_embeddings)
        return np.concatenate(embeddings, axis=0)

    def infer_episode_progress(
        self,
        reference_frames: np.ndarray,
        frames: np.ndarray,
        batch_size: int,
        progress_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> np.ndarray:
        if reference_frames.size == 0:
            raise ValueError("At least one reference frame is required for progress estimation.")
        if frames.size == 0:
            raise ValueError("Target frames are required for progress estimation.")
        prompt_text = progress_prompt if progress_prompt is not None else self._default_progress_prompt
        generation_tokens = max_new_tokens if max_new_tokens is not None else self._progress_max_new_tokens
        if generation_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")

        reference_images = [self._frame_to_pil(frame) for frame in reference_frames]
        progress_batches: List[np.ndarray] = []
        total_steps = frames.shape[0]
        for start in range(0, total_steps, batch_size):
            end = min(start + batch_size, total_steps)
            batch_frames = frames[start:end]
            batch_prompt = self._format_progress_prompt(prompt_text, batch_frames.shape[0])
            messages = [
                self._build_progress_message(reference_images, batch_frames, batch_prompt)
            ]
            text_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                padding=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            processor_outputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in processor_outputs.items()
            }
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=generation_tokens,
                )
            trimmed_ids = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
            ]
            decoded_outputs = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs = [self._strip_thought_process(text) for text in decoded_outputs]
            batch_progress = self._parse_progress_outputs(
                outputs[0],
                expected_count=batch_frames.shape[0],
            )
            progress_batches.append(np.asarray(batch_progress, dtype=np.float32))
        return np.concatenate(progress_batches, axis=0)

    @staticmethod
    def _build_messages(frames: np.ndarray, prompt: str) -> List[List[Dict[str, object]]]:
        messages: List[List[Dict[str, object]]] = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
        return messages

    @staticmethod
    def _build_progress_message(
        reference_images: Sequence[Image.Image],
        query_frames: Sequence[np.ndarray],
        prompt: str,
    ) -> List[Dict[str, object]]:
        content: List[Dict[str, object]] = [{"type": "text", "text": "Reference demonstration frames:"}]
        for idx, image in enumerate(reference_images, start=1):
            content.append({"type": "text", "text": f"Reference frame {idx}:"})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": "Target episode frames:"})
        for idx, frame in enumerate(query_frames, start=1):
            content.append({"type": "text", "text": f"Target frame {idx}:"})
            content.append({"type": "image", "image": Qwen3VLExtractor._frame_to_pil(frame)})
        content.append({"type": "text", "text": prompt})
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    @staticmethod
    def _frame_to_pil(frame: np.ndarray) -> Image.Image:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return Image.fromarray(frame)

    @staticmethod
    @staticmethod
    def _format_progress_prompt(base_prompt: str, num_frames: int) -> str:
        suffix = (
            f"There are {num_frames} target frame(s). Respond with {num_frames} decimal number(s) "
            "between 0 and 1, separated by commas, where each number corresponds to the target frames "
            "in order. Do not include any reasoning, explanations, or thinking process text—output only "
            "the numbers."
        )
        return f"{base_prompt.strip()} {suffix}"

    @staticmethod
    def _parse_progress_outputs(output_text: str, expected_count: int) -> List[float]:
        # LOGGER.info(output_text)
        matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*(%?)", output_text)
        values: List[float] = []
        for value_str, percent_flag in matches:
            value = float(value_str)
            is_percent = percent_flag == "%" or value > 1.0
            if is_percent:
                value /= 100.0
            values.append(float(np.clip(value, 0.0, 1.0)))
        if len(values) < expected_count:
            LOGGER.warning(
                "Expected %d progress values but only parsed %d from output: %s",
                expected_count,
                len(values),
                output_text,
            )
            values.extend([float("nan")] * (expected_count - len(values)))
        return values[:expected_count]

    @staticmethod
    def _strip_thought_process(output_text: str) -> str:
        cleaned = re.sub(r"(?is)<think>.*?</think>", " ", output_text)
        cleaned = re.sub(r"(?is)(?:thought|reasoning|think)\s*:\s*.*?(?=\n\s*\n|$)", " ", cleaned)
        return cleaned.strip()


class Pi0InternalExtractor(BaseEmbeddingExtractor):
    """Intrinsic visual embeddings from pretrained Pi0 / Pi0.5 models."""

    def __init__(self, backbone: RewardBackboneConfig, params: ExtractorInitParams):
        super().__init__(backbone, params)

        visual_config = backbone.visual_config
        if visual_config.kind != "pi0_internal":
            raise ValueError("Pi0/Pi0.5 internal configuration is missing.")
        pi0_params = visual_config.params

        if _policy_config is None or _config is None:
            raise ImportError(
                "Pi0 internal extractor requires openpi.policies.policy_config and openpi.training.config."
            )

        self.device = params.device
        self.policy = _policy_config.create_trained_policy(
            _config.get_config(pi0_params.get("policy_config")),
            pi0_params.get("policy_dir"),
        )

        self._device_count = jax.local_device_count()
        requested_multi_device = bool(pi0_params.get("enable_multi_device", True))
        self._multi_device_enabled = requested_multi_device and self._device_count > 1
        self._mesh: Mesh | None = None
        self._batch_sharding: NamedSharding | None = None
        if self._multi_device_enabled:
            devices = np.asarray(jax.local_devices())
            self._mesh = Mesh(devices, ("data",))
            self._batch_sharding = NamedSharding(self._mesh, PartitionSpec("data"))
            LOGGER.info(
                "Pi0 internal extractor using %d GPUs for batched embeddings.",
                self._device_count,
            )
        else:
            if requested_multi_device and self._device_count <= 1:
                LOGGER.warning(
                    "Multi-device mode requested but only %d GPU detected. Falling back to single GPU.",
                    self._device_count,
                )
            self._multi_device_enabled = False
            self._mesh = None
            self._batch_sharding = None

    def extract_visual_embeddings(
        self,
        frames: np.ndarray,
        batch_size: int,
        prompt: str,
        other_frames: Optional[np.ndarray] = None,
        proprio_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert (
            other_frames is not None and proprio_state is not None
        ), "Other frames and proprio state are required for Pi0 internal extractor."
        embeddings: List[np.ndarray] = []
        total_steps = frames.shape[0]
        for start in range(0, total_steps, batch_size):
            end = min(start + batch_size, total_steps)
            batch_frames = frames[start:end]
            batch_other_frames = other_frames[start:end]
            batch_proprio_state = proprio_state[start:end]
            original_batch = batch_frames.shape[0]
            batch_inputs = {
                "observation/image": batch_frames,
                "observation/wrist_image": batch_other_frames,
                "observation/state": batch_proprio_state,
                "prompt": np.array([prompt] * original_batch),
            }
            if self._multi_device_enabled and self._batch_sharding is not None:
                sharded_inputs, _ = self._prepare_multi_device_batch(batch_inputs)
                batch_embeddings = self.policy.extract_visual_embeddings(
                    sharded_inputs,
                    batch_sharding=self._batch_sharding,
                )
                batch_embeddings = batch_embeddings[:original_batch]
            else:
                batch_embeddings = self.policy.extract_visual_embeddings(batch_inputs)
            embeddings.append(batch_embeddings)
        return np.concatenate(embeddings, axis=0)

    def _prepare_multi_device_batch(self, batch_inputs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
        assert self._multi_device_enabled
        batch_size = next(iter(batch_inputs.values())).shape[0]
        remainder = batch_size % self._device_count
        if remainder == 0:
            return batch_inputs, 0
        pad = self._device_count - remainder
        padded_inputs = {key: self._pad_array(value, pad) for key, value in batch_inputs.items()}
        return padded_inputs, pad

    @staticmethod
    def _pad_array(array: np.ndarray, pad: int) -> np.ndarray:
        if pad == 0:
            return array
        if array.shape[0] == 0:
            raise ValueError("Cannot pad an empty batch for sharding.")
        pad_values = np.repeat(array[-1:], pad, axis=0)
        return np.concatenate([array, pad_values], axis=0)


def build_embedding_extractor(
    backbone: RewardBackboneConfig,
    params: ExtractorInitParams,
) -> BaseEmbeddingExtractor:
    """Factory that returns an extractor implementation for ``backbone``."""
    if backbone.visual_config.kind == "dinov2" and backbone.language_config is not None:
        return DinoMinilmExtractor(backbone, params)
    if backbone.visual_config.kind == "qwen3_vl":
        return Qwen3VLExtractor(backbone, params)
    if backbone.visual_config.kind == "pi0_internal":
        return Pi0InternalExtractor(backbone, params)
    raise ValueError(
        f"Unsupported backbone configuration: visual={backbone.visual_config.kind}, "
        f"language={getattr(backbone.language_config, 'kind', None)}."
    )


