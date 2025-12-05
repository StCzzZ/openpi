from __future__ import annotations

from typing import Dict, Optional

import h5py
import numpy as np
import torch
import torch.utils.data.dataset as dataset

from reward_model.util import dict_apply


class MultiStageDataset(dataset.Dataset):
    """
    Dataset class for training reward transformer.
    Each episode in the original dataset has:
        - visual embeddings stored under a configurable key (e.g. "dino_embeddings")
        - optional language embedding stored under a configurable key (e.g. "minlm_task_embedding")
        - progress: (N,) progress of the episode
        - stage: (N, ) ground-truth stage labels
        - subtask_progress: (N, ) subtask progress of each frame

    Args:
        dataset_path: Path to the dataset directory.
        num_stages: Number of stages in the specific task.
        max_seq_len: Maximum sequence length for reward modeling.
    """
    def __init__(
        self,
        dataset_path: str,
        num_stages: int,
        max_seq_len: int,
        video_rewind: bool,
        visual_embedding_key: str = "dino_embeddings",
        language_embedding_key: Optional[str] = "minlm_task_embedding",
        discrete: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.num_stages = num_stages
        self.max_seq_len = max_seq_len
        self.video_rewind = video_rewind
        self.visual_embedding_key = visual_embedding_key
        self.language_embedding_key = language_embedding_key

        self._load_dataset()
        self.visual_dim = self._infer_feature_dim(self.visual_embedding_key)
        self.language_dim = self._infer_feature_dim(self.language_embedding_key)
        self.discrete = discrete
        if discrete:
            self._calc_stage_prior()

    def _load_dataset(self) -> None:
        """
        Load the hdf5 dataset from the given path.
        Besides, calculates the cumulative timestep for random sampling.
        """
        self.h5_file = h5py.File(self.dataset_path, "r")
        self.episode_keys = list(self.h5_file.keys())
        self.cumulative_timestep = np.cumsum(
            [0]
            + [len(self.h5_file[key][self.visual_embedding_key]) for key in self.h5_file]
        )

    def _infer_feature_dim(self, key: Optional[str]) -> Optional[int]:
        if key is None or not self.episode_keys:
            return None
        first_episode = self.h5_file[self.episode_keys[0]]
        if key not in first_episode:
            return None
        sample = np.asarray(first_episode[key])
        if sample.ndim == 0:
            return 1
        return int(sample.shape[-1])

    def _calc_stage_prior(self) -> None:
        """
        Calculate the prior probability of each stage.
        """
        self.stage_prior = np.zeros(self.num_stages)
        for key in self.h5_file:
            stage_labels = self.h5_file[key]["stage"]
            for stage in np.unique(stage_labels):
                self.stage_prior[stage] += np.sum(stage_labels == stage)
        self.stage_prior /= np.sum(self.stage_prior)
        self.cumulative_stage_prior = np.cumsum(self.stage_prior)
        self.cumulative_stage_prior = np.concatenate(
            (np.array([0.0]), self.cumulative_stage_prior[:-1])
        )

    def sample_from_episode(self, episode_dict: h5py.Group) -> Dict[str, np.ndarray]:
        """
        Sample from a given episode to a fixed sequence length.
        """
        num_frames = len(episode_dict[self.visual_embedding_key])
        start_index = np.random.randint(0, num_frames - 3)
        end_index = np.random.randint(start_index + 3, num_frames)
        visual_embeddings = episode_dict[self.visual_embedding_key][start_index:end_index]
        visual_embeddings = self.padding_sequence(visual_embeddings)
        language_embedding = self._read_language_embedding(episode_dict)
        # Progress is defined to start from the start index of the sample, following ReWIND's convention
        sampled_progress = np.arange(end_index - start_index, dtype=np.float32) / (num_frames - start_index)
        progress = self.padding_sequence(sampled_progress)
        sample = {
            "visual_embeddings": visual_embeddings,
            "progress": progress,
        }
        if self.discrete:
            # Extracted from human / VLM annotations of high-level task stages
            stage = self.padding_sequence(episode_dict["stage"][start_index:end_index])
            subtask_progress = self.padding_sequence(
                episode_dict["subtask_progress"][start_index:end_index]
            )
            sample.update({
                "stage": stage,
                "subtask_progress": subtask_progress,
            })
        if language_embedding is not None:
            sample["language_embeddings"] = language_embedding
        return sample

    def sample_rewinded_video_from_episode(self, episode_dict: h5py.Group) -> Dict[str, np.ndarray]:
        """
        Sample from a given episode to a fixed sequence length, but randomly rewind the video.
        """
        num_frames = len(episode_dict[self.visual_embedding_key])
        start_index = np.random.randint(0, num_frames - 3)
        end_index = np.random.randint(start_index + 3, num_frames)
        split_index = np.random.randint(start_index, end_index - 2)
        seq_forward = episode_dict[self.visual_embedding_key][start_index:end_index]
        seq_backward = np.asarray(episode_dict[self.visual_embedding_key])[
            end_index - 2 : split_index : -1
        ]
        visual_embeddings = np.concatenate((seq_forward, seq_backward), axis=0)
        visual_embeddings = self.padding_sequence(visual_embeddings)
        language_embedding = self._read_language_embedding(episode_dict)
        sampled_progress = np.arange(end_index - start_index, dtype=np.float32) / (num_frames - start_index)
        progress = np.concatenate(
            (sampled_progress, sampled_progress[-2 : split_index - start_index : -1]), axis=0
        )
        progress = self.padding_sequence(progress)
        sample = {
            "visual_embeddings": visual_embeddings,
            "progress": progress,
        }
        if self.discrete:
            stage = np.concatenate(
                (
                    episode_dict["stage"][start_index:end_index],
                    np.asarray(episode_dict["stage"])[end_index - 2 : split_index : -1],
                ),
                axis=0,
            )
            stage = self.padding_sequence(stage)
            subtask_progress = np.concatenate(
                (
                    episode_dict["subtask_progress"][start_index:end_index],
                    np.asarray(episode_dict["subtask_progress"])[end_index - 2 : split_index : -1],
                ),
                axis=0,
            )
            subtask_progress = self.padding_sequence(subtask_progress)
            sample.update({
                "stage": stage,
                "subtask_progress": subtask_progress,
            })
        if language_embedding is not None:
            sample["language_embeddings"] = language_embedding
        return sample

    def _read_language_embedding(self, episode_dict: h5py.Group) -> Optional[np.ndarray]:
        if self.language_embedding_key is None:
            return None
        if self.language_embedding_key not in episode_dict:
            return None
        embedding = np.asarray(episode_dict[self.language_embedding_key])
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        return embedding

    def padding_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad a sequence to a fixed length.
        """
        seq_len = len(sequence)
        if seq_len == 0:
            feature_shape = sequence.shape[1:] if sequence.ndim > 1 else ()
            return np.zeros((self.max_seq_len,) + feature_shape, dtype=sequence.dtype)
        if seq_len < self.max_seq_len:
            padding_length = self.max_seq_len - seq_len
            last_element = sequence[-1]
            padding = np.array([last_element] * padding_length)
            return np.concatenate([sequence, padding], axis=0)
        else:
            sampled_indices = np.linspace(0, seq_len-1, self.max_seq_len).astype(int)
            return sequence[sampled_indices]

    def __len__(self) -> int:
        return self.cumulative_timestep[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetch the corresponding episode and sample from the dataset
        """
        episode_id = np.sum([1 for i in self.cumulative_timestep if i <= idx]) - 1
        output_dict = self.h5_file[self.episode_keys[episode_id]]
        if self.video_rewind:
            if np.random.random() < 0.8:
                return dict_apply(self.sample_rewinded_video_from_episode(output_dict), torch.from_numpy)
        return dict_apply(self.sample_from_episode(output_dict), torch.from_numpy)


if __name__ == "__main__":
    dataset = MultiStageDataset(
        dataset_path="/data/yuwenye/reward_modeling/data/sarm/1127_kitchen_test_pi0_internal_embeddings.hdf5",
        num_stages=7,
        max_seq_len=32,
        video_rewind=True,
        visual_embedding_key="pi0_internal_embeddings",
    )
    import pdb; pdb.set_trace()