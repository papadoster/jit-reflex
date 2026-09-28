"""C1-E2 GJ row: scripts/c1e2_j.py against the stock SmolVLA path, on a tiny random SmolVLA built by lerobot's own
classes from the real checkpoint config with the network shrunk (CPU, fp32, no weights loaded)."""

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
swe = pytest.importorskip("lerobot.policies.smolvla.smolvlm_with_expert")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e2_j  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402


def test_split_path_value_and_forward_jacobian(monkeypatch):
    real = swe.AutoConfig.from_pretrained

    def shrunk(name, **kw):  # the real SmolVLM2 config, small; 4 layers: self-attn (0, 2) and cross-attn (1, 3)
        c = real(name, **kw)
        c.text_config.update(dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=16))
        c.vision_config.update(dict(num_hidden_layers=1, hidden_size=32, intermediate_size=64, num_attention_heads=2,
                                    image_size=128))
        return c

    monkeypatch.setattr(swe, "AutoConfig", types.SimpleNamespace(from_pretrained=shrunk))
    cfg = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
    cfg.device, cfg.load_vlm_weights, cfg.resize_imgs_with_padding = "cpu", False, (128, 128)
    torch.manual_seed(0)
    policy = SmolVLAPolicy(cfg).float().eval().requires_grad_(False)  # as the runner
    batch = {"observation.images.image": torch.rand(1, 3, 96, 96), "observation.images.image2": torch.rand(1, 3, 96, 96),
             OBS_STATE: torch.randn(1, 8), OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (1, 7)),
             OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)}  # right padding
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim)

    s, f = c1e2_j.split_path(policy, batch, noise)
    with torch.no_grad():
        err_value = float((f(s)[0] - policy.predict_action_chunk(dict(batch), noise=noise)[0]).abs().max())
    J = c1e2_j.jacobian_all_positions(policy, batch, noise)
    # reference: reverse mode through the whole stock path; fp32 central differences are roundoff-bound here
    # (rel. 2.4e-2 at step 1e-3, 2.3e-3 at 1e-2: this random net's J is small, ~1e-3 per entry)
    st = policy.prepare_state(batch).requires_grad_(True)
    out = policy.model.sample_actions(*policy.prepare_images(batch), batch[OBS_LANGUAGE_TOKENS],
                                      batch[OBS_LANGUAGE_ATTENTION_MASK], st, noise=noise)[0, :, :7].reshape(-1)
    J_rev = torch.autograd.grad(out, st, torch.eye(out.numel()), is_grads_batched=True)[0][:, 0, :8]
    err_J = float((J.reshape(-1, 8) - J_rev).norm() / J_rev.norm())
    print(f"value max abs {err_value:.2e}, J fwd vs rev rel {err_J:.2e}, |J| {float(J.norm()):.3f}")
    assert J.shape == (cfg.chunk_size, 7, 8)
    assert err_value <= 1e-4
    assert err_J <= 1e-4
