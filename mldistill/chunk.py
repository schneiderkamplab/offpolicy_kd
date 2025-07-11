import torch

class ChunkingTransformation:
    flushable = True

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.buffer = None

    def __call__(self, sample: dict[str, torch.Tensor] | None) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor] | None:
        if sample is None:
            buffer = self.buffer
            self.buffer = None
            return {'input_ids': buffer} if buffer is not None and buffer.numel() > 0 else None
        tokens = sample['input_ids'] if self.buffer is None else torch.cat([self.buffer, sample['input_ids']])
        n_full = tokens.size(0) // self.chunk_size
        chunks = [{'input_ids': tokens[i * self.chunk_size : (i + 1) * self.chunk_size]} for i in range(n_full)]
        self.buffer = tokens[n_full * self.chunk_size:]
        return chunks if chunks else None
