# Third-Party Code

The LitePT baseline is run through Pointcept.

Recommended layout:

```text
third_party/
  pointcept/   # git clone https://github.com/Pointcept/Pointcept.git
```

Record the Pointcept commit SHA used for experiments here before final runs.
Do not commit edits inside third-party repositories.

The LitePT nuScenes checkpoint should live under:

```text
checkpoints/LitePT/nuscenes-semseg-litept-small-v1m1/model/model_best.pth
```
