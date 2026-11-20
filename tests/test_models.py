"""
Smoke tests for the model zoo: every architecture builds, runs a forward pass
on a small gather and is recognised by jtr_marine.model_loader from its
weights alone. CPU only, no data, no checkpoints.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config
from jtr_marine.model_loader import detect_architecture, load_checkpoint
from jtr_marine.models.unet import build_model
from jtr_marine.models.unet_no_film import build_model_no_film
from jtr_marine.models.unet_ablation import build_ablation_model
from jtr_marine.models.unet_attn import SeismicUNetAttn
from jtr_marine.models.chai2020_unet import Chai2020Wrapper
from jtr_marine.models.chai2020_cond import Chai2020CondWrapper


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def attn_model(cfg, kind):
    mc = cfg.model
    return SeismicUNetAttn(
        in_channels=mc.in_channels, out_channels=mc.out_channels,
        enc_channels=tuple(mc.enc_channels),
        bottleneck_channels=mc.bottleneck_channels,
        encoder1_kernel=tuple(mc.encoder1_kernel),
        film_hidden_dim=mc.film_hidden_dim, norm_type=mc.norm_type,
        dropout=mc.dropout, n_groups=getattr(mc, "n_groups", 8),
        attention_type=kind, attention_kwargs=None)


# name -> (builder, expected architecture tag, expected parameter count or None)
MODELS = {
    "compact_unet":       (lambda c: build_model(c), "none", 8_360_865),
    "chai2020":           (lambda c: Chai2020Wrapper(), "chai2020", 87_149_953),
    "chai2020_mask_film": (lambda c: Chai2020CondWrapper(in_channels=2, use_film=True),
                           "chai2020_cond", 87_295_041),
    "chai2020_mask_only": (lambda c: Chai2020CondWrapper(in_channels=2, use_film=False),
                           "chai2020_cond", None),
    "ablation_k5_none":   (lambda c: build_ablation_model(c, kernel=5, norm_type="none"),
                           "v6_ablation", None),
    "ablation_k3_none":   (lambda c: build_ablation_model(c, kernel=3, norm_type="none"),
                           "v6_ablation", None),
    "attn_channel":       (lambda c: attn_model(c, "channel"), "channel", None),
}


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.mark.parametrize("name", list(MODELS))
def test_build_forward_and_reload(name, cfg, tmp_path):
    builder, arch, expected = MODELS[name]
    torch.manual_seed(0)
    model = builder(cfg).eval()
    if expected is not None:
        assert n_params(model) == expected

    # small gather: (batch, [gather, mask], time, offset)
    x = torch.randn(1, 2, 128, 48)
    geom = torch.tensor([3000.0 / cfg.data.max_streamer_length])
    with torch.no_grad():
        y = model(x, geom)
    assert y.shape[-2:] == x.shape[-2:]
    assert torch.isfinite(y).all()

    state = model.state_dict()
    assert detect_architecture(state) == arch

    ckpt = tmp_path / "best.pt"
    torch.save({"model_state": state, "epoch": 0}, ckpt)
    reloaded, _ = load_checkpoint(ckpt, cfg, torch.device("cpu"), verbose=False)
    assert n_params(reloaded) == n_params(model)
    with torch.no_grad():
        assert torch.allclose(reloaded(x, geom), y)


def test_no_film_ignores_geometry(cfg):
    model = build_model_no_film(cfg).eval()
    x = torch.randn(1, 2, 64, 32)
    with torch.no_grad():
        a = model(x, torch.tensor([0.33]))
        b = model(x, torch.tensor([1.0]))
    assert torch.equal(a, b)
