"""Verify imports work end-to-end inside the container."""

import figs
from figs.control.velocity_controller import VelocityController

import nav_policy
from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy, count_parameters
from nav_policy.model.losses import bc_loss, per_component_mse
from nav_policy.data.build_dataset import build
from nav_policy.data.rgb_horizon_dataset import RGBHorizonDataset
from nav_policy.deploy.policy_controller import RGBVelocityController
from nav_policy.deploy.frame_buffer import FrameBuffer

import torch

print(f"figs            : {figs.__file__}")
print(f"nav_policy      : {nav_policy.__file__}")
print(f"torch           : {torch.__version__}")
print(f"cuda available  : {torch.cuda.is_available()}")
print(f"cuda device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# Tiny shape sanity check on the model itself.
m = RGBVelocityPolicy(T=4, H=10, cmd_dim=4)
dummy = torch.zeros(2, 4, 3, 224, 224)
out = m(dummy)
print(f"model forward   : in={tuple(dummy.shape)} -> out={tuple(out.shape)}  "
      f"trainable_params={count_parameters(m):,}")
assert out.shape == (2, 10, 4), out.shape

print("ALL IMPORTS + MODEL SHAPE CHECK OK")
