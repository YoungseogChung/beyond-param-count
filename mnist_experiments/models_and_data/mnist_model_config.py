import argparse

args = argparse.Namespace(
    method="soft",
    setting="mnist",
    num_epochs=15,
    lr=1e-3,
    batch_size=256,
    optimizer="adam",
    flip_pixels=False,
    patch_size=14,
    num_experts=-1,  # HACK: make sure I overwrite it
    routing_type="soft_moe",
    phi_init="uniform",
    num_slots=1,
    expert_out_dim=196,
    expert_fn_multiplier=1,
    uniform_d=False,
    uniform_c=False,
    freeze_phi=False,
    encoder=None,
    encoder_out_dim=196,
    decoder="linear",
    num_dummy=0,
    type_dummy="min",
    seed=0,
    gpu="0",
    save_dir="logs/smoe_mnist_constant_params",
    save_interim=True,
    save_x=False,
    dbg=False,
    dry=False,
    name=None,  # HACK: make sure I overwrite it
    save_every=5,
    normalize_moe_input=True
)
