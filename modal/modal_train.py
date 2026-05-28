"""
Local entrypoint — train FocusFusion on Modal.

  modal run modal/modal_train.py --experiment e1                  # live logs; job dies if terminal closes
  modal run --detach modal/modal_train.py --experiment e1         # live logs; job survives terminal close
  modal run --detach modal/modal_train.py --experiment e2         # T=6 temporal run (Person 2)

The ``--detach`` flag MUST come before the script path to be a Modal CLI flag.
Placed after the script it becomes a function argument and has no effect.

``--detach`` behaviour:
  without: Modal stops the ephemeral app when this entrypoint exits → job killed.
  with:    Modal keeps the spawned job running after the local client disconnects.
           The entrypoint still blocks (showing live logs) until you close the
           terminal or Ctrl-C; the remote job then continues independently.

Remote: ``modal_config.train`` → ``python -m focus_fusion.train.trainer`` on GPU.
Checkpoints: ``/experiments/<experiment>/`` on Volume ``focus-fusion-experiments``.

Override GPU without editing code (Windows):
  $env:MODAL_GPU_TRAIN="A100-80GB"; modal run modal/modal_train.py --experiment e1
"""

import sys
sys.path.insert(0, "modal")

from modal_config import app, spawn_modal_function, train


@app.local_entrypoint()
def main(
    experiment: str = "e1",
    config: str = "configs/default.yaml",
    extra_args: str = "",
) -> None:
    """
    Launch a FocusFusion training run on Modal.

    Parameters
    ----------
    experiment
        Which experiment to run: ``e1`` (T=1, single frame) or ``e2`` (T=6, temporal).
    config
        Path to the YAML config file (relative to repo root).
    extra_args
        Space-separated extra CLI flags forwarded verbatim to the trainer, e.g.
        ``"--batch-size 2 --epochs 50 --lr 1e-4"``.
    """
    extra = [part for part in extra_args.split() if part.strip()] if extra_args else None

    invoke = f"modal run --detach modal/modal_train.py --experiment {experiment}"
    if config != "configs/default.yaml":
        invoke += f" --config {config}"
    if extra_args.strip():
        invoke += f' --extra-args "{extra_args.strip()}"'
    print(f"Invoke command: {invoke}")

    spawn_modal_function(
        train,
        label=f"train:{experiment}",
        wait=True,
        experiment=experiment,
        config=config,
        extra_args=extra,
    )
