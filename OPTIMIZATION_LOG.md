# Optimization Log

This is the running journal of inference-optimization experiments on the LingBot-World Fast pipeline. Modeled after `karpathy/autoresearch`'s methodology (one canonical metric, single-file diffs, git as the lab notebook). The metric is `bench.py` and its `bench/results/<sha>.json` output.

## Ground rules

1. **One change per commit.** Each commit must isolate a single optimization. Mixed commits are not allowed in this branch.
2. **Bench every commit.** Run `bench.py` after each change. Commit the resulting `bench/results/<sha>.json` together with the code change.
3. **Bit-identical bar (default).** The output MD5 must match the baseline `ed2f82628308a3f8acd9b7935bb84401`. If a commit cannot preserve it, that's flagged explicitly in the entry below and only landed with explicit user OK to relax the bar.
4. **Append-only.** Append entries below; never edit a landed entry. If a commit gets reverted, mark it `REVERTED` and add a follow-up entry.

## Canonical benchmark

Fixed inputs (see `bench.py::CONFIG`):

- Task: `i2v-A14B`, size `480*832`, 81 frames, seed 42
- Image: `examples/03/image.jpg` · Action path: `examples/03`
- Prompt: lakeside scene (see CONFIG)
- Hardware: 8× H100 80GB

Run it:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  .venv/bin/torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29500 bench.py
```

## Entry template

```
### <commit-sha> — <one-line title>

- Branch: opt/<short-name>
- Hypothesis: <what we expected to happen and why>
- Bit-identical bar: kept ✓ / relaxed (PSNR Z dB) / broke ✗
- Δ generate_ms: <baseline-ms> → <new-ms>  (<sign><pct>%)
- Phases of note: <which phase shrank / grew>
- Lesson: <one sentence on what we learned>
```

---

## Entries

### 18d7565 — Baseline + phase timers (no perf change)

- Branch: optimization/baseline-profile-instrumentation
- Hypothesis: Add `_phase(name)` instrumentation around T5 encode, VAE encode/decode, each chunk's denoise loop and KV-cache update, so we can see where time goes without changing behavior.
- Bit-identical bar: kept ✓ (instrumentation only)
- Δ generate_ms: — → 21459 (reference point)
- Phases of note: chunk0 denoise = 7196 ms (warmup tax), steady-state chunks ~1000 ms each, VAE encode 2478 ms, VAE decode 3128 ms, KV-cache update ~260 ms/chunk (×7 = 1855 ms).
- Lesson: chunk 0 is a 6+ second outlier; without warmup elimination we can't see real per-chunk numbers.

## Future-work backlog (not in current bit-identical sequence)

Listed here so we don't forget them; each would require relaxing the bit-identical bar:

- **FlashAttn 2 → 3 upgrade.** Code already auto-detects; just install `flash-attn-interface`. Expected: faster attention, but FP-order may shift MD5.
- **`torch.compile` on DiT forward.** Triton kernels reorder accumulations. Likely 1.3–2× on attention/MLP, but MD5-changing.
- **Multi-GPU VAE decode.** Currently rank-0 only (3.1 s, 7 GPUs idle). Splitting changes reduction order.
- **Sage Attention** (INT8 attention). ~9% in Voltage Park's measurement. Quality lossy.
- **TeaCache reformulated chunk-to-chunk.** Voltage Park's biggest win (51%) was step-to-step; the Fast variant has only 4 steps but 7 chunks. Open research question.
- **fp8 weights / activations.** H100 has native fp8 support via TransformerEngine.
- **NF4 4-bit weights** (community ckpt exists at cahlen/lingbot-world-base-cam-nf4).
- **Persistent server mode.** ~104 s cold-start dominates total wall-clock for short clips.
- **Long-video specific** (961 frames): tune `--local_attn_size` to cap KV cache growth.
