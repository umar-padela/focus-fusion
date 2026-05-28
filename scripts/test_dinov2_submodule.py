"""Quick smoke test: verify dinov2 submodule loads and patch forward works."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for smoke test

import sys
import pathlib

repo_root = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(repo_root / "third_party" / "dinov2"))

import torch

print("Loading dinov2_vits14 from submodule (pretrained=False for speed)...")
model = torch.hub.load(
    str(repo_root / "third_party" / "dinov2"),
    "dinov2_vits14",
    source="local",
    pretrained=False,
)
model.eval()

x = torch.randn(2, 3, 448, 448)
with torch.no_grad():
    out = model.get_intermediate_layers(x, n=1, return_class_token=False)[0]

print(f"patch output shape: {tuple(out.shape)}")  # expect (2, 1024, 384)
assert out.shape == (2, 1024, 384), f"Unexpected shape: {out.shape}"
assert not out.isnan().any(), "NaN in output"
print("dinov2_vits14 submodule smoke test PASSED.")
