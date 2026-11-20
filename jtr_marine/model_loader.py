#!/usr/bin/env python3
"""
jtr_marine/model_loader.py — Load any project checkpoint, detecting its architecture
automatically from the stored weights.

Why
---
Evaluation code that hard-codes `build_model` from jtr_marine.models.unet
fails on the attention variants with

    RuntimeError: Unexpected key(s) in state_dict: "attention...."

This helper inspects the state_dict and builds the right class, so the same
evaluation code can be applied unchanged to:

    - compact U-Net (SeismicUNet, no attention)
    - channel attention (squeeze-and-excitation)
    - spatial attention (multi-head attention)
    - transformer (2 transformer blocks)
    - the GAN baseline (generator is a plain SeismicUNet)
    - the U-Net of Chai et al. (2020), independent published baseline
      (Chai2020Wrapper from models/chai2020_unet.py, parameters stored as "net.*")
    - the same backbone with mask channel and/or FiLM
      (Chai2020CondWrapper from models/chai2020_cond.py, "net.*" and "film.*")
    - compact U-Net with other kernel size and/or no normalisation
      (SeismicUNetAblation from models/unet_ablation.py, same keys as the compact U-Net)

Usage in an existing script
---------------------------
Replace these three lines:

    model = build_model(config).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

with:

    from jtr_marine.model_loader import load_checkpoint
    model, ckpt = load_checkpoint(checkpoint_path, config, device)

Standalone check (from the repository root)
-------------------------------------------
    python -m jtr_marine.model_loader checkpoints/<run>/best.pt
"""

import torch


def detect_architecture(state_dict) -> str:
    """
    Infer the architecture from the parameter names.

    Returns one of: 'chai2020', 'chai2020_cond', 'v6_ablation', 'none',
    'channel', 'spatial', 'transformer'
    """
    keys = list(state_dict.keys())

    # Chai et al. (2020) backbone: stored under "net."
    if any(k.startswith("net.encoders.") for k in keys):
        in_ch = state_dict["net.encoders.0.block.0.weight"].shape[1]
        has_film = any(k.startswith("film.") for k in keys)
        if in_ch == 1 and not has_film:
            return "chai2020"            # published baseline (Chai2020Wrapper)
        return "chai2020_cond"           # + mask channel and/or FiLM

    attn_keys = [k for k in keys if k.startswith("attention.")]

    if not attn_keys:
        # ablation model: kernel other than 3x3 at level 2, or no norm parameters
        w2 = state_dict.get("enc2.conv.block.0.weight")
        has_norm = "enc2.conv.block.1.weight" in state_dict
        if w2 is not None and (w2.shape[-1] != 3 or not has_norm):
            return "v6_ablation"
        return "none"
    if any(k.startswith("attention.blocks.") for k in attn_keys):
        return "transformer"
    if any("mha." in k for k in attn_keys):
        return "spatial"
    if any(k.startswith("attention.mlp.") for k in attn_keys):
        return "channel"

    raise ValueError(
        f"Unrecognised attention parameters: {attn_keys[:5]} ... "
        "please extend detect_architecture()."
    )


def _materialise_lazy_parameters(model, config, device, state):
    """
    The spatial/transformer attention blocks create their positional embedding
    lazily, on the first forward pass (see models/attention_blocks.py,
    _maybe_init_pos_emb). A freshly built model therefore lacks
    'attention.pos_emb' / 'attention.pos_emb_param', and load_state_dict fails.

    We run one dummy forward pass so those parameters are created with the
    correct shape before the weights are loaded.
    """
    needs_warmup = any(k.startswith("attention.pos_emb") for k in state)
    if not needs_warmup:
        return

    dc = config.data
    t_samples = getattr(dc, "n_time_samples", 2250)
    n_offsets = getattr(dc, "n_offsets", 181)
    in_ch = config.model.in_channels

    model.train()  # some lazy inits are guarded by module mode
    with torch.no_grad():
        dummy_x = torch.zeros(1, in_ch, t_samples, n_offsets, device=device)
        dummy_geom = torch.zeros(1, device=device)
        model(dummy_x, dummy_geom)


def _fix_lazy_parameter_shapes(model, state, device):
    """
    Fallback for the lazily created positional embeddings.

    If the dummy forward pass did not create them, or created them with a
    different number of tokens than the checkpoint (e.g. because the gather
    size in the current config differs from the one used at training time),
    we create/resize them directly from the checkpoint shapes. The values are
    overwritten by load_state_dict immediately afterwards, so only the shape
    matters here.
    """
    import torch.nn as nn

    lazy_keys = [k for k in state if k.startswith("attention.")
                 and "pos_emb" in k]
    if not lazy_keys:
        return

    for key in lazy_keys:
        target_shape = state[key].shape
        # navigate to the owning module, e.g. "attention.pos_emb_param"
        parts = key.split(".")
        module = model
        for p in parts[:-1]:
            module = getattr(module, p)
        name = parts[-1]

        current = getattr(module, name, None)
        if current is None or tuple(current.shape) != tuple(target_shape):
            new_param = nn.Parameter(
                torch.zeros(*target_shape, device=device)
            )
            if hasattr(module, name):
                delattr(module, name)
            module.register_parameter(name, new_param)
            # keep the plain attribute in sync (the class uses self.pos_emb)
            if name == "pos_emb_param" and hasattr(module, "pos_emb"):
                module.pos_emb = new_param
            if name == "pos_emb":
                module._max_tokens = target_shape[1]


def load_checkpoint(checkpoint_path, config, device, verbose: bool = True):
    """
    Load a checkpoint and return (model, ckpt_dict).

    The model is returned in eval() mode on the requested device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model_state"]
    arch = detect_architecture(state)

    if arch == "chai2020":
        from jtr_marine.models.chai2020_unet import Chai2020Wrapper
        model = Chai2020Wrapper()
    elif arch == "chai2020_cond":
        from jtr_marine.models.chai2020_cond import Chai2020CondWrapper
        use_film = "film.mlp.0.weight" in state
        model = Chai2020CondWrapper(
            in_channels     = state["net.encoders.0.block.0.weight"].shape[1],
            use_film        = use_film,
            film_hidden_dim = state["film.mlp.0.weight"].shape[0] if use_film else 64,
        )
    elif arch == "v6_ablation":
        from jtr_marine.models.unet_ablation import build_ablation_model
        k = int(state["enc2.conv.block.0.weight"].shape[-1])
        norm = "instance" if "enc2.conv.block.1.weight" in state else "none"
        model = build_ablation_model(config, kernel=k, norm_type=norm)
    elif arch == "none":
        from jtr_marine.models.unet import build_model
        model = build_model(config)
    else:
        from jtr_marine.models.unet_attn import SeismicUNetAttn
        mc = config.model
        model = SeismicUNetAttn(
            in_channels         = mc.in_channels,
            out_channels        = mc.out_channels,
            enc_channels        = tuple(mc.enc_channels),
            bottleneck_channels = mc.bottleneck_channels,
            encoder1_kernel     = tuple(mc.encoder1_kernel),
            film_hidden_dim     = mc.film_hidden_dim,
            norm_type           = mc.norm_type,
            dropout             = mc.dropout,
            n_groups            = getattr(mc, "n_groups", 8),
            attention_type      = arch,
            attention_kwargs    = None,
        )

    model = model.to(device)
    _materialise_lazy_parameters(model, config, device, state)
    _fix_lazy_parameter_shapes(model, state, device)
    model.load_state_dict(state)
    model.eval()

    if verbose:
        extra = ""
        if arch == "chai2020_cond":
            extra = f" (in_channels={model.in_channels}, FiLM={model.use_film})"
        elif arch == "v6_ablation":
            extra = f" (kernel={model.kernel}, norm={model.norm_type})"
        print(f"  Architecture:  {arch}{extra}")
        print(f"  Parameters:    {model.count_parameters():,}")
        print(f"  Epoch:         {ckpt.get('epoch', 'N/A')}")
        if "val_snr_db" in ckpt:
            print(f"  Val SNR:       {ckpt['val_snr_db']:.2f} dB")

    return model, ckpt


if __name__ == "__main__":
    import sys
    from config import get_config

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cfg = get_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in sys.argv[1:]:
        print(f"\n{path}")
        try:
            load_checkpoint(path, cfg, dev)
        except Exception as e:
            print(f"  FAILED: {e}")
