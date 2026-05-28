# Third-party submodules

Do not commit edits inside `third_party/*` on `main`. Apply fixes in our wrappers under `focus_fusion/models/backbones/`.

## Pinned SHAs

| Path | Upstream repo | Pinned commit | Notes |
|------|--------------|---------------|-------|
| `third_party/dinov2/` | https://github.com/facebookresearch/dinov2 | `7b187bd` | Latest stable main (2024); no release tags exist |
| `third_party/ptv3/` | https://github.com/Pointcept/PointTransformerV3 | TBD | To be added by Person 2 |

## Setup

```bash
# Fresh clone
git clone --recurse-submodules <repo-url>

# Existing clone (after a pull that bumped a submodule)
git submodule update --init --recursive
```

## Updating a submodule

```bash
git -C third_party/dinov2 fetch
git -C third_party/dinov2 checkout <new-sha>
# Then stage and commit the bump:
git add third_party/dinov2
git commit -m "Bump dinov2 submodule to <new-sha>"
# Update the SHA in this README too.
```

## Pretrained weights

Weights are **not** stored in submodules — they go under `checkpoints/` (gitignored).

| Model | Download |
|-------|----------|
| `dinov2_vitb14` | `checkpoints/dinov2/dinov2_vitb14_pretrain.pth` — downloaded via torch.hub on first use, or manually from https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth |
| `ptv3` | TBD |
