# Third-party dependencies

Do not commit edits inside `third_party/*`. Apply fixes in our wrappers under `focus_fusion/models/backbones/`.

## Submodules

| Path | Upstream | Pinned commit | Notes |
|------|----------|---------------|-------|
| `third_party/dinov2/` | https://github.com/facebookresearch/dinov2 | `7b187bd` | Vision backbone (Person 2) |

```bash
# Fresh clone
git clone --recurse-submodules <repo-url>

# Existing clone after submodule bump
git submodule update --init --recursive
```

## Pointcept / LitePT (Person 1)

LitePT runs through [Pointcept](https://github.com/Pointcept/Pointcept). Clone it separately — it is **not** a git submodule (too large).

```bash
git clone https://github.com/Pointcept/Pointcept.git third_party/pointcept
```

Record the exact commit SHA used for final experiments here before submission.

Expected layout:
```
third_party/
  dinov2/       ← submodule (tracked by .gitmodules)
  pointcept/    ← manual clone (gitignored)
```

## Pretrained weights

Weights go under `checkpoints/` (gitignored — do not commit).

| Model | Path | Source |
|-------|------|--------|
| DINOv2 ViT-S/14 | downloaded via torch.hub on first use | fbaipublicfiles |
| LitePT small | `/experiments/checkpoints/litept/model_best.pth` (Modal volume) | [`prs-eth/LitePT`](https://huggingface.co/prs-eth/LitePT) — run `modal run modal/modal_setup.py` |
