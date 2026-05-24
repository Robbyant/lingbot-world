import math
import torch
import torch.nn.functional as torch_F
from einops import rearrange

from ..modules.attention import flash_attention
from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import all_to_all, gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    dit_cond_dict=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # cam
    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
        c2ws_plucker_emb = [
            rearrange(
                i,
                '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                c1=self.patch_size[0],
                c2=self.patch_size[1],
                c3=self.patch_size[2],
            ) for i in c2ws_plucker_emb
        ]
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb,
                                     dim=1)  # [1, (L1+...+Ln), C]
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden_states = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
        c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states

        cam_len = c2ws_plucker_emb.size(1)
        if cam_len < seq_len:
            pad_len = seq_len - cam_len
            pad = c2ws_plucker_emb.new_zeros(
                c2ws_plucker_emb.size(0), pad_len, c2ws_plucker_emb.size(2))
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, pad], dim=1)
        elif cam_len > seq_len:
            c2ws_plucker_emb = c2ws_plucker_emb[:, :seq_len, :]

        if get_world_size() > 1:
            c2ws_plucker_emb = torch.chunk(
                c2ws_plucker_emb, get_world_size(), dim=1)[get_rank()]
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        dit_cond_dict=dit_cond_dict)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def sp_dit_forward_causal(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    dit_cond_dict=None,
    kv_cache=None,
    crossattn_cache=None,
    current_start=0,
    max_attention_size=1_000_000,
    frame_seqlen=None,
    cross_attn_first_call=None,
):
    """
    x:                  A list of videos each with shape [C, T, H, W].
    t:                  [B].
    context:            A list of text embeddings each with shape [L, C].
    seq_len:            Maximum sequence length for positional encoding.
    y:                  Conditional video inputs for image-to-video mode, same shape as x.
    dit_cond_dict:      Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                        with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
    kv_cache:           Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                        (Tensor of shape [B, kv_size, local_heads, head_dim]), ``global_end_index``,
                        and ``local_end_index`` (scalar Tensors tracking cache position).
    crossattn_cache:    Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                        (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
    current_start:      Token offset of the current chunk in the full sequence. Used to index
                        into the KV cache and compute positional embeddings correctly.
    max_attention_size: Maximum number of KV tokens each query can attend to. Limits the
                        effective context window of self-attention to control memory usage.

    Note: This function uses sequence parallel (SP) with Ulysses attention.
    The sequence is chunked across GPUs. Inside attention, all-to-all redistributes
    from sequence-parallel to head-parallel, applies causal RoPE on the full sequence,
    updates the KV cache, runs attention, then all-to-all back.
    """

    assert len(x) == 1

    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)[0]
    assert seq_lens.max() <= seq_len
    x = torch.cat(x)

    # Pad sequence to be divisible by world_size for SP chunking.
    # int(seq_lens) is one cudaStreamSynchronize; cache it once so each
    # attention layer can reuse via the seq_lens_int kwarg instead of
    # re-casting per-layer (32x).
    sp_size = get_world_size()
    seq_lens_int = int(seq_lens)
    padded_seq_lens = ((seq_lens_int + sp_size - 1) // sp_size) * sp_size
    sp_pad_len = padded_seq_lens - seq_lens_int
    if sp_pad_len > 0:
        x = torch.cat([x, x.new_zeros(x.size(0), sp_pad_len, x.size(2))], dim=1)

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), padded_seq_lens)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, padded_seq_lens)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # cam
    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
        c2ws_plucker_emb = [
            rearrange(
                i,
                '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                c1=self.patch_size[0],
                c2=self.patch_size[1],
                c3=self.patch_size[2],
            ) for i in c2ws_plucker_emb
        ]
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb,
                                     dim=1)  # [1, (L1+...+Ln), C]
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden_states = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
        c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states

        cam_len = c2ws_plucker_emb.size(1)
        if cam_len < padded_seq_lens:
            pad_len_cam = padded_seq_lens - cam_len
            pad = c2ws_plucker_emb.new_zeros(
                c2ws_plucker_emb.size(0), pad_len_cam, c2ws_plucker_emb.size(2))
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, pad], dim=1)
        elif cam_len > padded_seq_lens:
            c2ws_plucker_emb = c2ws_plucker_emb[:, :padded_seq_lens, :]

        if get_world_size() > 1:
            c2ws_plucker_emb = torch.chunk(
                c2ws_plucker_emb, get_world_size(), dim=1)[get_rank()]
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        dit_cond_dict=dit_cond_dict,
        max_attention_size=max_attention_size,
        frame_seqlen=frame_seqlen,
        cross_attn_first_call=cross_attn_first_call,
        seq_lens_int=seq_lens_int)

    for block_index, block in enumerate(self.blocks):
        kwargs.update(
            {
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
            }
        )
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    return [u.float() for u in x]


def sp_attn_forward_causal(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    kv_cache=None,
    current_start=0,
    max_attention_size=1_000_000,
    frame_seqlen=None,
    seq_lens_int=None):
    r"""
    Sequence-parallel causal self-attention using Ulysses all-to-all.

    Input x is sequence-chunked (each GPU holds padded_seq_lens/world_size tokens).
    all-to-all gathers the full sequence and scatters heads, so each GPU
    operates on the full sequence with local_heads.  After RoPE, padding
    tokens are sliced off so that only the actual ``seq_lens`` tokens enter
    the KV cache and flash attention.  The output is padded back to
    ``padded_seq_lens`` before the reverse all-to-all.

    Args:
        x(Tensor):              Shape [B, padded_seq_lens // world_size, C]
        seq_lens(Tensor):       Total number of valid tokens (full sequence, not per-rank).
        grid_sizes(Tensor):     Shape [B, 3], the second dimension contains (F, H, W).
        freqs(Tensor):          Rope freqs, shape [1024, C / num_heads / 2].
        kv_cache(dict):         Self-attention KV cache. Contains keys ``k``, ``v``
                                (Tensor of shape [B, kv_size, local_heads, head_dim]),
                                ``global_end_index``, and ``local_end_index``
                                (scalar Tensors tracking cache position).
        current_start(int):     Token offset of the current chunk in the full sequence.
                                Used to index into the KV cache and compute positional
                                embeddings correctly.
        max_attention_size(int): Maximum number of KV tokens each query can attend to.
                                Limits the effective context window of self-attention
                                to control memory usage.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    sp_size = get_world_size()

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    # all-to-all: gather sequence, scatter heads
    # [B, s/p, N, d] -> [B, s, N/p, d]
    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    # padded_seq_lens = s * sp_size may exceed seq_lens due to SP padding
    padded_seq_lens = s * sp_size
    if seq_lens_int is None:
        seq_lens_int = int(seq_lens)

    if frame_seqlen is None:
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    current_start_frame = current_start // frame_seqlen

    # Apply causal RoPE on full (padded) sequence with local heads
    # causal_rope_apply only processes the first f*h*w tokens internally,
    # leaving any SP padding tokens unchanged
    query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
    key   = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

    # Slice to actual seq_lens — discard SP padding for cache and attention
    query = query[:, :seq_lens_int]
    key = key[:, :seq_lens_int]
    v = v[:, :seq_lens_int]

    current_end = current_start + seq_lens
    kv_cache_size = kv_cache["k"].shape[1]
    if self.local_attn_size == -1:
        # Fast path (no eviction possible — cache is global). Both indices
        # advance identically every forward, so local_end_index ==
        # current_end and local_start_index == current_start. Python ints
        # via current_start (kwarg) + seq_lens_int — no .item() syncs.
        local_end_index = current_start + seq_lens_int
        local_start_index = current_start
        kv_cache["k"][:, local_start_index:local_end_index] = key
        kv_cache["v"][:, local_start_index:local_end_index] = v
    elif (current_end > kv_cache["global_end_index"].item()) and (
            seq_lens + kv_cache["local_end_index"].item() > kv_cache_size):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        # Clone the source slice to avoid overlapping memory error
        sink_tokens = self.sink_size * frame_seqlen
        num_evicted_tokens = seq_lens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        # Insert the new keys/values at the end
        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - seq_lens
        kv_cache["k"][:, local_start_index:local_end_index] = key
        kv_cache["v"][:, local_start_index:local_end_index] = v
    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - seq_lens
        kv_cache["k"][:, local_start_index:local_end_index] = key
        kv_cache["v"][:, local_start_index:local_end_index] = v

    k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
    v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]

    # Attention on local heads, full key/value cache for this rank
    x_local = flash_attention(query, k_cache, v_cache)  # [B, seq_lens, local_heads, d]

    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    # Pad back to padded_seq_lens for all-to-all
    local_heads = x_local.size(2)
    sp_pad = padded_seq_lens - seq_lens_int
    if sp_pad > 0:
        x_local = torch.cat([
            x_local,
            x_local.new_zeros(b, sp_pad, local_heads, d)
        ], dim=1)

    # all-to-all back: gather heads, scatter sequence
    # [B, s, N/p, d] -> [B, s/p, N, d]
    x_full = all_to_all(x_local, scatter_dim=1, gather_dim=2)

    # output
    x_full = x_full.flatten(2)
    x_full = self.o(x_full)
    return x_full
