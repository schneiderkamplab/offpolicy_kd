import torch

class ChunkingTransformation:
    flushable = True

    def __init__(self, chunk_size, eos_token_id):
        self.chunk_size = chunk_size
        self.eos_token_id = eos_token_id
        self.buffer = None

    def __call__(self, sample: dict[str, torch.Tensor] | None) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor] | None:
        if sample is None:
            buffer = self.buffer
            self.buffer = None
            return {'input_ids': buffer} if buffer is not None and buffer.numel() > 0 else None
        input_ids = sample['input_ids']
        cat_inputs = [self.buffer, input_ids] if self.buffer is not None else [input_ids]
        if input_ids[-1].item() != self.eos_token_id:
            cat_inputs.append(torch.tensor([self.eos_token_id], dtype=input_ids.dtype))
        tokens = torch.cat(cat_inputs) if len(cat_inputs) > 1 else input_ids
        n_full = tokens.size(0) // self.chunk_size
        chunks = [{'input_ids': tokens[i * self.chunk_size : (i + 1) * self.chunk_size]} for i in range(n_full)]
        self.buffer = tokens[n_full * self.chunk_size:]
        return chunks if chunks else None
