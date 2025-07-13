import torch
import math

class RopeND:
    def __init__(self, head_dim=64, nd=3, max_lens=[1024, 64, 64], nd_split=[2, 1, 1], bases=[1000, 1000, 1000],
                 auto_base=True, cache_longer=1):
        self.nd = nd
        self.head_dim = head_dim
        self.max_lens = max_lens
        self.nd_split = nd_split
        self.split_dims = [2 * i * (head_dim // 2 // sum(nd_split)) for i in nd_split]
        assert sum(self.split_dims) == head_dim
        self.auto_base = auto_base
        if auto_base:
            # empirical, make cos(theta) = -1 when length is kL. base = kL/pi
            # And L=1 the difference (1/base)**(1/32) ~ 0.7-0.8 ~ pi/4
            # for traditional L = 4096, 8L/pi = 10.4k, base is set to 10k
            self.bases = [(int(8 * l / math.pi) // 100 + 1) * 100 for l in self.max_lens]
            print(f"Bases for rope: {self.bases}")
        else:
            self.bases = bases
        self.cache_longer = cache_longer

    def generated_cos_sin_mix2d(self, max_len, dim, device, base=1000):
        inv_freq = 1.0 / (base ** \
                          (torch.linspace(start=0, end=self.head_dim, steps=dim // 2,
                                          device=device).float() / self.head_dim))
        assert inv_freq.size(0) * 2 == dim, f"inv_freq.size(0) = {inv_freq.size(0)}, required dim = {dim}"

        t = torch.arange(max_len * self.cache_longer, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=1)
        return freqs.cos().to(torch.float), freqs.sin().to(torch.float)

    def generate_pos_embs_mix2d(self, position_ids, device=None):
        if device is None:
            device = position_ids.device

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        cos_emb_all, sin_emb_all = [], []
        for i in range(self.nd):
            dim_i = self.split_dims[i]
            base_i = self.bases[i]
            max_len_i = self.max_lens[i]
            if not hasattr(self, f"cos_{i}"):
                _cos, _sin = self.generated_cos_sin_mix2d(max_len=max_len_i, dim=dim_i, device=device, base=base_i)
                setattr(self, f"cos_{i}", _cos)
                setattr(self, f"sin_{i}", _sin)
            cos_emb_all.append(getattr(self, f'cos_{i}')[position_ids[i, :], :])
            sin_emb_all.append(getattr(self, f'sin_{i}')[position_ids[i, :], :])
        cos_emb = torch.cat(cos_emb_all, dim=-1)
        sin_emb = torch.cat(sin_emb_all, dim=-1)
        return cos_emb, sin_emb

    def __call__(self, q, k, position_ids):
        '''q: N N_head L C
        '''
        cos_emb, sin_emb = self.generate_pos_embs_mix2d(position_ids, device=q.device)

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos, sin):
            """Applies Rotary Position Embedding to the query and key tensors.

            Args:
                q (`torch.Tensor`): The query tensor.
                k (`torch.Tensor`): The key tensor.
                cos (`torch.Tensor`): The cosine part of the rotary embedding.
                sin (`torch.Tensor`): The sine part of the rotary embedding.
            Returns:
                `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
            """
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            dtype = q.dtype
            q = q.to(torch.float)
            k = k.to(torch.float)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            q_embed = q_embed.to(dtype)
            k_embed = k_embed.to(dtype)
            return q_embed, k_embed

        q, k = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
        return q, k