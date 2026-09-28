"""J by state for the exploratory GJ row of C1-E2: forward mode over the split path (image+language K/V once,
state token + flow differentiated), all chunk positions in one batch-8 pass. CUDA/CPU only (no forward AD on MPS).

The state is the last prefix token and the prefix-LM mask keeps image and language tokens from attending to it,
so their K/V do not depend on the state. Checked against the stock path in tests/test_c1e2_j.py."""

import torch
import torch.autograd.forward_ad as fwAD
from transformers.cache_utils import DynamicCache

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def split_path(policy, batch, noise):
    """-> (s, f): s (1, sd) the normalised state; f(s8) the normalised chunk (B, H, A) with s8 (B, sd) in place of s."""
    m, vwe, cfg = policy.model, policy.model.vlm_with_expert, policy.config
    # predict_action_chunk would post-process otherwise; the split path mirrors only the cached stock route
    assert not cfg.adapt_to_pi_aloha and cfg.rtc_config is None and cfg.use_cache
    sd = cfg.robot_state_feature.shape[0]
    state = policy.prepare_state(batch)
    with torch.no_grad():
        embs, pad, att = m.embed_prefix(*policy.prepare_images(batch), batch[OBS_LANGUAGE_TOKENS],
                                        batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
        assert att[0, -1] and not att[0, :-1].any()  # state = last token (no prefix_length padding), unseen by the rest
        att2d, pos = make_att_2d_masks(pad, att), torch.cumsum(pad, dim=1) - 1
        _, il = vwe.forward(attention_mask=att2d[:, :-1, :-1], position_ids=pos[:, :-1],
                            inputs_embeds=[embs[:, :-1], None], use_cache=True)
    layers, dt = vwe.get_vlm_model().text_model.layers, -1.0 / cfg.num_steps  # layers already cut to num_vlm_layers

    def f(s8):
        b = s8.shape[0]
        h = m.state_proj(torch.cat([s8, state[:, sd:].expand(b, -1)], 1))[:, None]
        p, mask, cache = pos[:, -1:].expand(b, -1), pad[:, None].expand(b, -1, -1), DynamicCache()
        for i, layer in enumerate(layers):  # vwe.forward_attn_layer + the residual/MLP part of vwe.forward, 1 query
            at = layer.self_attn
            x = layer.input_layernorm(h).to(at.q_proj.weight.dtype)
            shp = (b, 1, -1, at.head_dim)
            q, k = apply_rope(at.q_proj(x).view(shp), p), apply_rope(at.k_proj(x).view(shp), p)
            K = torch.cat([il.layers[i].keys.transpose(1, 2).expand(b, -1, -1, -1), k], 1)
            V = torch.cat([il.layers[i].values.transpose(1, 2).expand(b, -1, -1, -1), at.v_proj(x).view(shp)], 1)
            o = vwe.eager_attention_forward(mask, b, at.head_dim, q, K, V)
            h = (at.o_proj(o.to(at.o_proj.weight.dtype)) + h).to(at.o_proj.weight.dtype)  # stock adds in place
            h = layer.mlp(layer.post_attention_layernorm(h)) + h
            cache.update(K.transpose(1, 2), V.transpose(1, 2), i)  # post-RoPE, as the stock prefill stores them
        x = noise.expand(b, -1, -1)
        for step in range(cfg.num_steps):  # euler_integrate: from t = 1, dt = -1/num_steps
            t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=x.device).expand(b)
            x = x + dt * m.denoise_step(prefix_pad_masks=pad.expand(b, -1), past_key_values=cache, x_t=x, timestep=t)
        return x[..., : cfg.action_feature.shape[0]]  # unpadded, as predict_action_chunk

    return state[:, :sd], f


@torch.no_grad()  # no reverse graph needed; forward AD ignores grad mode (inference_mode would disable it)
def jacobian_all_positions(policy, batch, noise):
    """-> (J, a_norm) for one env's preprocessed batch (B = 1) and its noise: J (H, A, sd) d a_norm / d s_norm at every
    chunk position; a_norm (H, A) the split path's own normalised chunk, for the caller to check against the stock one."""
    s, f = split_path(policy, batch, noise)
    sd = s.shape[1]
    with fwAD.dual_level():
        s8 = fwAD.make_dual(s.expand(sd, -1).contiguous(), torch.eye(sd, dtype=s.dtype, device=s.device))
        out = fwAD.unpack_dual(f(s8))
        return out.tangent.permute(1, 2, 0), out.primal[0]
