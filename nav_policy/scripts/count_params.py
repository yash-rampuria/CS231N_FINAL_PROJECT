"""Helper: print parameter counts for the finalized RGBVelocityPolicy."""

from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy

m = RGBVelocityPolicy(
    T=4, H=10, cmd_dim=4,
    gru_hidden=256, gru_layers=1,
    mlp_hidden=(256, 128),
    freeze_stem_and_layer1=True,
)
total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"total parameters     = {total:,}")
print(f"trainable parameters = {trainable:,}")
print(f"frozen parameters    = {total - trainable:,}")
