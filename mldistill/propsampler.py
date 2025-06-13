import random
from torch.utils.data import Sampler

__all__ = ["ProportionalSampler"]

class ProportionalSampler(Sampler):
    def __init__(self, datasets, weights, seed=42, num_samples=None):
        assert len(datasets) == len(weights), "Each dataset must have a corresponding weight"
        self.datasets = datasets
        self.weights = [w / sum(weights) for w in weights]
        self.lengths = [len(ds) for ds in datasets]
        self.seed = seed
        self.offsets = []
        offset = 0
        for length in self.lengths:
            self.offsets.append(offset)
            offset += length
        max_total = min(self.lengths[i] / self.weights[i] for i in range(len(self.datasets)))
        self.num_samples = min(num_samples or int(sum(self.lengths)), int(max_total))
        self.rnd = random.Random(self.seed)

    def __iter__(self):
        raw_counts = [self.weights[i] * self.num_samples for i in range(len(self.datasets))]
        counts = [min(int(c), self.lengths[i]) for i, c in enumerate(raw_counts)]
        sampled_indices = []
        for i, count in enumerate(counts):
            length = self.lengths[i]
            offset = self.offsets[i]
            if count > length:
                raise ValueError(f"Cannot sample {count} items from dataset {i} of length {length}")
            indices = self.rnd.sample(range(length), count)
            sampled_indices.extend(offset + i for i in indices)
        self.rnd.shuffle(sampled_indices)
        return iter(sampled_indices)

    def __len__(self):
        return self.num_samples
