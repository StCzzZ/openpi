import torch
import torch.utils.data.dataset as dataset
import h5py
import numpy as np

from reward_model.util import dict_apply

class MultiStageDataset(dataset.Dataset):
    """
    Dataset class for training reward transformer.
    Each episode in the original dataset has:
        - dino_embeddings: (N, 768) DINO embeddings of the video frames
        - minlm_task_embedding: (1, 384) task embedding
        - progress: (N,) progress of the episode
        - stage: (N, ) ground-truth stage labels
        - subtask_progress: (N, ) subtask progress of each frame

    Args:
        dataset_path: Path to the dataset directory.
        num_stages: Number of stages in the specific task.
        max_seq_len: Maximum sequence length for reward modeling.
    """
    def __init__(self, dataset_path, num_stages, max_seq_len):
        super().__init__()
        self.dataset_path = dataset_path
        self.num_stages = num_stages
        self.max_seq_len = max_seq_len

        self._load_dataset()
        self._calc_stage_prior()

    def _load_dataset(self):
        """
        Load the hdf5 dataset from the given path.
        Besides, calculates the cumulative timestep for random sampling.
        """
        self.h5_file = h5py.File(self.dataset_path, 'r')
        self.episode_keys = list(self.h5_file.keys())
        self.cumulative_timestep = np.cumsum([0] + \
            [len(self.h5_file[key]['dino_embeddings']) for key in self.h5_file])

    def _calc_stage_prior(self):
        """
        Calculate the prior probability of each stage.
        """
        self.stage_prior = np.zeros(self.num_stages)
        for key in self.h5_file:
            stage_labels = self.h5_file[key]['stage']
            for stage in np.unique(stage_labels):
                self.stage_prior[stage] += np.sum(stage_labels == stage)
        self.stage_prior /= np.sum(self.stage_prior)
        self.cumulative_stage_prior = np.cumsum(self.stage_prior)
        self.cumulative_stage_prior = np.concatenate((np.array([0.,]), self.cumulative_stage_prior[:-1]))

    def sample_from_episode(self, episode_dict):
        """
        Sample from a given episode to a fixed sequence length.
        """
        start_index = np.random.randint(0, len(episode_dict['dino_embeddings']) - 3)
        end_index = np.random.randint(start_index+3, len(episode_dict['dino_embeddings']))
        dino_embeddings = self.padding_sequence(episode_dict['dino_embeddings'][start_index:end_index])
        minlm_task_embedding = episode_dict['minlm_task_embedding']
        # Progress is defined to start from the start index of the sample, following ReWIND's convention
        sampled_progress = np.arange(end_index - start_index) / (len(episode_dict['dino_embeddings']) - start_index)
        progress = self.padding_sequence(sampled_progress)
        # Extracted from human / VLM annotations of high-level task stages
        stage = self.padding_sequence(episode_dict['stage'][start_index:end_index])
        subtask_progress = self.padding_sequence(episode_dict['subtask_progress'][start_index:end_index])
        return {
            'dino_embeddings': dino_embeddings,
            'minlm_task_embedding': np.asarray(minlm_task_embedding),
            'progress': progress,
            'stage': stage,
            'subtask_progress': subtask_progress,
        }
        
    def padding_sequence(self, sequence):
        """
        Pad a sequence to a fixed length.
        """
        seq_len = len(sequence)
        if seq_len < self.max_seq_len:
            padding_length = self.max_seq_len - seq_len
            last_element = sequence[-1]
            padding = np.array([last_element] * padding_length)
            return np.concatenate([sequence, padding], axis=0)
        else:
            sampled_indices = np.linspace(0, seq_len-1, self.max_seq_len).astype(int)
            return sequence[sampled_indices]

    def __len__(self):
        return self.cumulative_timestep[-1]

    def __getitem__(self, idx):
        """
        Fetch the corresponding episode and sample from the dataset
        """
        episode_id = np.sum([1 for i in self.cumulative_timestep if i <= idx]) - 1
        output_dict = self.h5_file[self.episode_keys[episode_id]]
        return dict_apply(self.sample_from_episode(output_dict), torch.from_numpy)


if __name__ == "__main__":
    dataset = MultiStageDataset(
        dataset_path="/data/yuwenye/reward_modeling/data/sarm/1113_kitchen_embeddings.hdf5",
        num_stages=7,
        max_seq_len=32
    )
    print(len(dataset))
    print(dataset[0])