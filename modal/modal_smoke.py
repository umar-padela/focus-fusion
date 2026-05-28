"""
Local entrypoint — Modal infrastructure smoke test.

  modal run modal/modal_smoke.py

Verifies:
  - GPU available (CUDA)
  - torch 2.1.0 installed
  - DINOv2 ViT-S/14 forward pass: (1, 6, 3, 448, 448) → (1, 6, 1024, 384)
  - Volume write (focus-fusion-experiments/smoke_ok.txt)

Does not run training. Use modal_train.py for that.
"""

import sys
sys.path.insert(0, "modal")

from modal_config import app, smoke, spawn_modal_function


@app.local_entrypoint()
def main():
    result = spawn_modal_function(smoke, label="smoke")
    print("\nSmoke finished.")
    print(f"  torch:           {result.get('torch')}")
    print(f"  cuda_available:  {result.get('cuda_available')}")
    print(f"  cuda_device:     {result.get('cuda_device')}")
    print(f"  dinov2_shape:    {result.get('dinov2_output_shape')}")
    print("Volume focus-fusion-experiments/smoke_ok.txt written.")
