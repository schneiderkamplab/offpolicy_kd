import random
import torch
from torch.utils.data import Sampler

__all__ = ["ProportionalSampler", "RandomSampler"]

class RandomSampler(Sampler):
    def __init__(self, datasets, seed=42, num_samples=None):
        self.total_length = sum(len(ds) for ds in datasets)
        self.num_samples = num_samples or self.total_length
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        perm = torch.randperm(self.total_length, generator=self.generator)[:self.num_samples]
        return iter(perm.tolist())

    def __len__(self):
        return self.num_samples

import torch
from torch.utils.data import Sampler

class ProportionalSampler(Sampler):
    def __init__(self, datasets, weights, seed=42, num_samples=None):
        assert len(datasets) == len(weights), "Each dataset must have a corresponding weight"
        self.datasets = datasets
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.weights /= self.weights.sum()

        self.lengths = [len(ds) for ds in datasets]
        self.total_length = sum(self.lengths)

        self.offsets = [0]
        for length in self.lengths[:-1]:
            self.offsets.append(self.offsets[-1] + length)
        max_total = min(self.lengths[i] / self.weights[i] for i in range(len(datasets)))
        self.num_samples = min(num_samples or self.total_length, int(max_total))
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        sampled_indices = []
        raw_counts = self.weights * self.num_samples
        counts = [min(int(c), self.lengths[i]) for i, c in enumerate(raw_counts)]
        for i, count in enumerate(counts):
            local_perm = torch.randperm(self.lengths[i], generator=self.generator)[:count]
            global_indices = local_perm + self.offsets[i]
            sampled_indices.extend(global_indices.tolist())
        final_perm = torch.randperm(len(sampled_indices), generator=self.generator)
        return iter([sampled_indices[i] for i in final_perm])

    def __len__(self):
        return self.num_samples
