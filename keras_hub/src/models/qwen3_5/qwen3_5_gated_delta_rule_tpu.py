"""TPU-oriented chunked gated delta rule for Qwen3.5 (JAX backend only).

Same arguments, results and float32 internals as `_chunk_gated_delta_rule` in
`qwen3_5_gated_delta_net.py`, restructured so the TPU runs it as a few large
batched ops instead of thousands of small dependent ones:

- Each chunk's `T = (I - A)^-1` (`A` strictly lower triangular) is built by
  blocked forward substitution written as six levels of two full-size masked
  matmuls (`_inverse_by_masked_doubling`), instead of a 63-step row loop that
  rewrites the whole chunk x chunk matrix each step. It never forms powers of
  `A`, so it is as stable as the loop.
- The intra-chunk work of every chunk is computed at once; only the
  chunk-to-chunk state recurrence stays sequential, as a `lax.scan` instead of
  an unrolled Python loop over chunks.

Selected with `KERAS_HUB_QWEN3_5_GDN_IMPL=tpu` (see
`qwen3_5_gated_delta_net._chunk_gated_delta_rule`).
"""

import jax
import jax.numpy as jnp

from keras_hub.src.models.qwen3_5 import qwen3_5_gated_delta_net as gdn


def _inverse_by_masked_doubling(a):
    """`(I - a)^-1` for strictly lower triangular `a` (..., C, C), C a power
    of two.

    While `t` is block diagonal with (width x width) blocks, `t + t @ a_pair @
    t` (with `a_pair` keeping only the lower-left (width x width) quarter of
    every diagonal (2 width)-block of `a`) is block diagonal with (2 width)
    blocks `[[T11, 0], [T22 a21 T11, T22]]`. Starting from `t = I`, log2(C)
    such levels give the full inverse: LAPACK's blocked forward substitution,
    run as full-size matmuls. Like every other matmul in the model, they use
    JAX's global matmul precision (`jax_default_matmul_precision`).
    """
    size = a.shape[-1]
    if size & (size - 1):
        raise ValueError(f"Chunk size {size} is not a power of two.")
    index = jnp.arange(size)
    inverse = jnp.broadcast_to(jnp.eye(size, dtype = a.dtype), a.shape)
    width = 1
    while width < size:
        same_pair = (index[:, None] // (2 * width)) == (
            index[None, :] // (2 * width)
        )
        lower_left = ((index[:, None] // width) % 2 == 1) & (
            (index[None, :] // width) % 2 == 0
        )
        pair_a = jnp.where(same_pair & lower_left, a, 0.0)
        inverse = inverse + jnp.matmul(jnp.matmul(inverse, pair_a), inverse)
        width *= 2
    return inverse


def chunk_gated_delta_rule_tpu(
    query,
    key,
    value,
    g,
    beta,
    chunk_size = 64,
    initial_state = None,
    output_final_state = False,
    padding_mask = None,
):
    """Chunked gated delta rule, batched over chunks.

    Args:
        query: (B, seq, num_heads, head_k_dim)
        key: (B, seq, num_heads, head_k_dim)
        value: (B, seq, num_heads, head_v_dim)
        g: (B, seq, num_heads) — decay gates (log-space)
        beta: (B, seq, num_heads) — write gates (sigmoid-space)
        chunk_size: Chunk size, a power of two (default 64).
        initial_state: Optional (B, num_heads, head_k_dim, head_v_dim)
            recurrent state.
        output_final_state: Whether to return the final state.
        padding_mask: Optional (B, seq) mask; masked tokens leave the state
            untouched.
    Returns:
        output: (B, seq, num_heads, head_v_dim) in the input dtype
        final_state: recurrent state or None
    """
    input_dtype = query.dtype
    query = gdn._l2norm(query, axis = -1)
    key = gdn._l2norm(key, axis = -1)
    query, key, value = (
        jnp.transpose(x, (0, 2, 1, 3)).astype(jnp.float32)
        for x in (query, key, value)
    )
    beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
    g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)
    beta, g = gdn._mask_state_updates(beta, g, padding_mask)

    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query, key, value = (
            jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0)))
            for x in (query, key, value)
        )
        beta, g = (jnp.pad(x, ((0, 0), (0, 0), (0, pad))) for x in (beta, g))
    num_chunks = (seq_len + pad) // chunk_size

    query = query * (1.0 / k_dim**0.5)
    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]

    def chunked(x):
        return x.reshape(
            batch_size, num_heads, num_chunks, chunk_size, *x.shape[3:]
        )

    query, key, value, v_beta, k_beta, g = map(
        chunked, (query, key, value, v_beta, k_beta, g)
    )
    g = jnp.cumsum(g, axis = -1)
    lower_incl = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.float32))
    decay_mask = (
        jnp.exp((g[..., :, None] - g[..., None, :]) * lower_incl) * lower_incl
    )
    strictly_lower = jnp.tril(jnp.ones((chunk_size, chunk_size), bool), k = -1)

    # Intra-chunk work for all chunks at once.
    a = jnp.where(
        strictly_lower,
        -(jnp.einsum("bhncd,bhnjd->bhncj", k_beta, key) * decay_mask),
        0.0,
    )
    with jax.named_scope("gdn_inverse"):
        t = _inverse_by_masked_doubling(a)
    with jax.named_scope("gdn_chunk_mix"):
        value = jnp.einsum("bhncj,bhnjd->bhncd", t, v_beta)
        k_cumdecay = jnp.einsum(
            "bhncj,bhnjd->bhncd", t, k_beta * jnp.exp(g)[..., None]
        )
        intra = jnp.where(
            lower_incl.astype(bool),
            jnp.einsum("bhncd,bhnjd->bhncj", query, key) * decay_mask,
            0.0,
        )

    g_last = g[..., -1]
    query_decayed = query * jnp.exp(g)[..., None]
    key_weighted = key * jnp.exp(g_last[..., None] - g)[..., None]
    state = (
        jnp.zeros((batch_size, num_heads, k_dim, v_dim), jnp.float32)
        if initial_state is None
        else jnp.asarray(initial_state).astype(jnp.float32)
    )

    # Only the chunk-to-chunk state recurrence is sequential.
    def step(state, chunk):
        k_cd, v, q_decayed, k_w, decay = chunk
        v_new = v - jnp.einsum("bhcd,bhdv->bhcv", k_cd, state)
        inter = jnp.einsum("bhcd,bhdv->bhcv", q_decayed, state)
        state = state * decay[..., None, None] + jnp.einsum(
            "bhcd,bhcv->bhdv", k_w, v_new
        )
        return state, (v_new, inter)

    def chunk_major(x):
        return jnp.moveaxis(x, 2, 0)

    with jax.named_scope("gdn_scan"):
        state, (v_new, inter) = jax.lax.scan(
            step,
            state,
            (
                chunk_major(k_cumdecay),
                chunk_major(value),
                chunk_major(query_decayed),
                chunk_major(key_weighted),
                chunk_major(jnp.exp(g_last)),
            ),
        )
    v_new, inter = jnp.moveaxis(v_new, 0, 2), jnp.moveaxis(inter, 0, 2)
    with jax.named_scope("gdn_output"):
        output = inter + jnp.einsum("bhncj,bhnjv->bhncv", intra, v_new)
    output = output.reshape(
        batch_size, num_heads, num_chunks * chunk_size, v_dim
    )[:, :, :seq_len]
    output = jnp.transpose(output, (0, 2, 1, 3)).astype(input_dtype)
    return output, (state if output_final_state else None)
