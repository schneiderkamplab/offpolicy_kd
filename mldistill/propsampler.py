import math
import random
from torch.utils.data import Dataset, Sampler

__all__ = ["ProportionalSampler", "IndexedMultiDataset"]

class ProportionalSampler(Sampler):
    def __init__(self, datasets, weights, seed=42, rank=0, world_size=1):
        assert len(datasets) == len(weights), "Each dataset must have a corresponding weight"
        self.datasets = datasets
        self.weights = [w / sum(weights) for w in weights]
        self.lengths = [len(ds) for ds in datasets]
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.index_mapping = self._generate_index_mapping()

    def _generate_index_mapping(self):
        rnd = random.Random(self.seed)
        # Total samples limited by most restrictive dataset given the weights
        max_total = min(self.lengths[i] / self.weights[i] for i in range(len(self.datasets)))
        max_total = int(max_total)
        print(f"Indexing {max_total} items")
        # Compute sample count per dataset
        counts = [int(w * max_total) for w in self.weights]
        # Sample indices per dataset
        index_mapping = []
        for i, count in enumerate(counts):
            indices = list(range(self.lengths[i]))
            selected = indices[:count]
            index_mapping.extend((i, idx) for idx in selected)
        # Global shuffle
        rnd.shuffle(index_mapping)
        # Shard for DDP (Accelerate-compatible)
        total = len(index_mapping)
        per_rank = math.ceil(total / self.world_size)
        start = self.rank * per_rank
        end = min(start + per_rank, total)
        return index_mapping[start:end]

    def __iter__(self):
        return iter(self.index_mapping)

    def __len__(self):
        return len(self.index_mapping)

class IndexedMultiDataset(Dataset):
    def __init__(self, datasets, index_mapping):
        self.datasets = datasets
        self.index_mapping = index_mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        dataset_id, sample_idx = self.index_mapping[idx]
        return self.datasets[dataset_id][sample_idx]
