import dataclasses
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class Pi05ValueWeightLoader(WeightLoader):
    """Loads pi05 checkpoints and initializes missing value-specific parameters."""

    base_params_path: str

    def load(self, params: at.Params) -> at.Params:
        base_params = _model.restore_params(download.maybe_download(self.base_params_path), restore_type=np.ndarray)
        # The pi05 checkpoint does not contain the new value tokens/head, so keep their initialization from `params`.
        return _merge_params(base_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class Pi05ValueExpertWeightLoader(WeightLoader):
    """Loads pi05 checkpoints and initializes missing value-specific parameters."""

    base_params_path: str

    def load(self, params: at.Params) -> at.Params:
        base_params = _model.restore_params(download.maybe_download(self.base_params_path), restore_type=np.ndarray)

        # Map SigLip and ActionExpert weights to ValueExpert
        flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
        new_params = {}
        for k, v in flat_base.items():
            # Copy SigLip weights
            if k.startswith("PaliGemma/img/"):
                new_key = k.replace("PaliGemma/img/", "ValuePaliGemma/img/")
                # Don't load the head (projector) if the shapes don't match
                if "head" in new_key:
                    continue
                new_params[new_key] = v
            # Copy ActionExpert (expert 1) weights to ValueExpert LLM
            elif k.startswith("PaliGemma/llm/"):
                parts = k.split("/")
                # Only process keys belonging to expert 1 (indicated by _1 suffix in names)
                if any(p.endswith("_1") and not p.isdigit() for p in parts):
                    # Remove _1 suffix from all parts
                    new_parts = [p[:-2] if (p.endswith("_1") and not p.isdigit()) else p for p in parts]
                    new_key = "/".join(new_parts).replace("PaliGemma/llm/", "ValuePaliGemma/llm/")
                    new_params[new_key] = v

        value_expert_params = flax.traverse_util.unflatten_dict(new_params, sep="/")

        # The pi05 checkpoint does not contain the new value tokens/head, so keep their initialization from `params`.
        # First merge base_params, keeping all other params initialized (regex=".*")
        params = _merge_params(base_params, params, missing_regex=".*")
        # Then merge extracted value expert params, again keeping all other params initialized
        return _merge_params(value_expert_params, params, missing_regex=".*")