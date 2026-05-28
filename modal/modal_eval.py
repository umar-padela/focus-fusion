"""
Local entrypoint — eval FocusFusion on Modal.

  modal run modal/modal_eval.py --experiment e1
  modal run modal/modal_eval.py --experiment e2 --split mini_val
  modal run modal/modal_eval.py --experiment e1 --checkpoint /experiments/e1/epoch_40.pt

Remote: ``modal_config.evaluate`` → ``python -m focus_fusion.eval.metrics`` on GPU.
Results written to ``/experiments/<experiment>/eval/`` on Volume ``focus-fusion-experiments``.

Browse results locally (paths are relative to volume root — omit /experiments):
  modal volume ls   focus-fusion-experiments
  modal volume ls   focus-fusion-experiments e1
  modal volume get  focus-fusion-experiments e1/eval/results.json results.json
"""

import sys
sys.path.insert(0, "modal")

from modal_config import app, evaluate, spawn_modal_function


@app.local_entrypoint()
def main(
    experiment: str = "e1",
    config: str = "configs/default.yaml",
    checkpoint: str | None = None,
    split: str = "mini_val",
    extra_args: str = "",
) -> None:
    """
    Launch a FocusFusion eval run on Modal.

    Parameters
    ----------
    experiment
        Which experiment to evaluate: ``e1`` or ``e2``.
    config
        Path to the YAML config file (relative to repo root).
    checkpoint
        Path to a checkpoint on the experiments volume.
        Defaults to ``/experiments/<experiment>/best.pt``.
    split
        nuScenes split to evaluate on: ``mini_val`` (default) or ``mini_train``.
    extra_args
        Space-separated extra CLI flags forwarded verbatim to the eval script.
    """
    extra = [part for part in extra_args.split() if part.strip()] if extra_args else None

    invoke = f"modal run modal/modal_eval.py --experiment {experiment} --split {split}"
    if checkpoint:
        invoke += f" --checkpoint {checkpoint}"
    if config != "configs/default.yaml":
        invoke += f" --config {config}"
    print(f"Invoke command: {invoke}")

    result = spawn_modal_function(
        evaluate,
        label=f"evaluate:{experiment}/{split}",
        wait=True,
        experiment=experiment,
        config=config,
        checkpoint=checkpoint,
        split=split,
        extra_args=extra,
    )

    print("\n--- eval result ---")
    print(f"experiment: {result.get('experiment')}")
    print(f"checkpoint: {result.get('checkpoint')}")
    print(f"split:      {result.get('split')}")
    print("Done. See /experiments/<experiment>/eval/ on Volume focus-fusion-experiments.")
