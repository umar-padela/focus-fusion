import modal

app = modal.App("ptv3-nuscenes-mini")

# Paths inside Modal container.
NUSCENES_DIR = "/data/nuscenes"
CKPT_DIR = "/checkpoints"
OUT_DIR = "/outputs"
POINTCEPT_DIR = "/root/Pointcept"
PROJECT_DIR = "/root/project"
BASELINE_DIR = f"{PROJECT_DIR}/ptv3_nuscenes_baseline"
LITEPT_CONFIG = f"{POINTCEPT_DIR}/configs/nuscenes/semseg-litept-v1m1-0-small.py"
LITEPT_DEFAULT_CHECKPOINT = (
    "LitePT/nuscenes-semseg-litept-small-v1m1/model/model_best.pth"
)

# Modal volumes.
nuscenes_vol = modal.Volume.from_name("nuscenes-data")
ckpt_vol = modal.Volume.from_name("ptv3-checkpoints")
out_vol = modal.Volume.from_name("ptv3-outputs")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "clang",
        "gcc",
        "g++",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .pip_install(
        # Core torch stack.
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",

        # General scientific/data deps.
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "h5py",
        "pyyaml",
        "tqdm",
        "open3d",

        # nuScenes.
        "nuscenes-devkit",
        "pyquaternion",

        # Pointcept common deps.
        "addict",
        "yapf",
        "tensorboard",
        "tensorboardX",
        "termcolor",
        "sharedarray",
        "einops",
        "plyfile",
        "timm",
        "wandb",

        # Newer Pointcept optional imports.
        "transformers==4.41.2",
        "peft==0.11.1",
        "huggingface_hub==0.23.4",
        "accelerate==0.31.0",

        # Sparse conv backend.
        "spconv-cu120",
    )
    .pip_install(
        # Prebuilt FlashAttention wheel matching Python 3.10 + PyTorch 2.1 + CUDA 12.1.
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.1.1/"
        "flash_attn-2.1.1+cu121torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands(
        "python -m pip install "
        "torch-scatter torch-cluster torch-sparse torch-spline-conv torch-geometric "
        "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"
    )
    .run_commands(
        f"git clone https://github.com/Pointcept/Pointcept.git {POINTCEPT_DIR}",
    )
    .run_commands(
        # Build Pointcept CUDA extension.
        f"cd {POINTCEPT_DIR}/libs/pointops && "
        "TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' "
        "CUDA_HOME=/usr/local/cuda "
        "python setup.py install"
    )
    .env(
        {
            "PYTHONPATH": f"{POINTCEPT_DIR}:/root/project",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9",
            "CUDA_HOME": "/usr/local/cuda",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
    )
    .run_commands(
        # Build-time smoke test.
        "python - <<'PY'\n"
        "import sys\n"
        f"sys.path.insert(0, '{POINTCEPT_DIR}')\n"
        "import torch\n"
        "import torch_scatter\n"
        "import torch_cluster\n"
        "import torch_sparse\n"
        "import torch_geometric\n"
        "import spconv.pytorch as spconv\n"
        "import wandb\n"
        "import peft\n"
        "import flash_attn\n"
        "import pointops\n"
        "from pointcept.models import build_model\n"
        "print('Pointcept dependency smoke test passed')\n"
        "PY"
    )
    # IMPORTANT: add_local_dir must stay last.
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "datasets",
            "runs",
            "outputs",
            "checkpoints",
        ],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={
        "/data": nuscenes_vol,
        "/outputs": out_vol,
    },
)
def check_data(split: str = "mini_val"):
    import subprocess
    from pathlib import Path

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        f"{BASELINE_DIR}/check_nuscenes_lidarseg.py",
        "--dataroot",
        NUSCENES_DIR,
        "--version",
        "v1.0-mini",
        "--split",
        split,
    ]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    out_vol.commit()


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={
        "/data": nuscenes_vol,
    },
)
def extract_data(
    mini_archive: str = "/data/archives/v1.0-mini.tgz",
    lidarseg_archive: str = "/data/archives/nuScenes-lidarseg-mini-v1.0.tar.bz2",
):
    import shutil
    import subprocess
    from pathlib import Path

    target = Path(NUSCENES_DIR)
    mini_path = Path(mini_archive)
    lidarseg_path = Path(lidarseg_archive)

    if not mini_path.exists():
        raise FileNotFoundError(f"Missing mini archive: {mini_path}")
    if not lidarseg_path.exists():
        raise FileNotFoundError(f"Missing lidarseg archive: {lidarseg_path}")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {mini_path} into {target}")
    subprocess.run(["tar", "-xzf", str(mini_path), "-C", str(target)], check=True)

    print(f"Extracting {lidarseg_path} into {target}")
    subprocess.run(["tar", "-xjf", str(lidarseg_path), "-C", str(target)], check=True)

    print("Extracted dataset tree:")
    subprocess.run(["find", str(target), "-maxdepth", "2", "-type", "d"], check=True)

    required = [
        target / "samples" / "LIDAR_TOP",
        target / "sweeps" / "LIDAR_TOP",
        target / "lidarseg" / "v1.0-mini",
        target / "v1.0-mini",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Expected extracted path is missing: {path}")

    nuscenes_vol.commit()
    print(f"Committed extracted nuScenes mini dataset to {target}")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={
        "/data": nuscenes_vol,
        "/checkpoints": ckpt_vol,
        "/outputs": out_vol,
    },
)
def eval_ptv3_mini(
    split: str = "mini_val",
    checkpoint_name: str = LITEPT_DEFAULT_CHECKPOINT,
    use_flash: bool = True,
):
    import subprocess
    from pathlib import Path

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    config = LITEPT_CONFIG
    checkpoint = f"{CKPT_DIR}/{checkpoint_name}"

    cmd = [
        "python",
        f"{BASELINE_DIR}/eval_pointcept_ptv3.py",
        "--dataroot",
        NUSCENES_DIR,
        "--version",
        "v1.0-mini",
        "--split",
        split,
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--device",
        "cuda",
        "--save-json",
        f"{OUT_DIR}/litept_{split}_metrics.json",
        "--save-predictions",
        f"{OUT_DIR}/litept_{split}_predictions",
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    out_vol.commit()
    print(f"Saved metrics to {OUT_DIR}/litept_{split}_metrics.json")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={
        "/data": nuscenes_vol,
        "/checkpoints": ckpt_vol,
        "/outputs": out_vol,
    },
)
def pointcept_test_mini(
    split: str = "mini_val",
    checkpoint_name: str = LITEPT_DEFAULT_CHECKPOINT,
    fast: bool = True,
    use_flash: bool = True,
):
    import pickle
    import subprocess
    from pathlib import Path

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    if split != "mini_val":
        raise ValueError("Pointcept official test wrapper currently supports split='mini_val'.")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    processed_root = Path("/tmp/pointcept_nuscenes_mini")
    info_dir = processed_root / "info"
    raw_link = processed_root / "raw"
    info_dir.mkdir(parents=True, exist_ok=True)
    if raw_link.exists() or raw_link.is_symlink():
        raw_link.unlink()
    raw_link.symlink_to(NUSCENES_DIR)

    nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_DIR, verbose=True)
    split_to_scenes = create_splits_scenes()
    scene_names = set(split_to_scenes[split])
    scene_token_to_name = {scene["token"]: scene["name"] for scene in nusc.scene}

    def sensor_to_ego(calibrated_sensor):
        return transform_matrix(
            calibrated_sensor["translation"],
            Quaternion(calibrated_sensor["rotation"]),
            inverse=False,
        )

    def ego_to_sensor(calibrated_sensor):
        return transform_matrix(
            calibrated_sensor["translation"],
            Quaternion(calibrated_sensor["rotation"]),
            inverse=True,
        )

    def ego_to_global(ego_pose):
        return transform_matrix(
            ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=False
        )

    def global_to_ego(ego_pose):
        return transform_matrix(
            ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=True
        )

    def collect_lidar_sweeps(ref_sd_record, max_sweeps=9):
        ref_cs = nusc.get("calibrated_sensor", ref_sd_record["calibrated_sensor_token"])
        ref_pose = nusc.get("ego_pose", ref_sd_record["ego_pose_token"])
        ref_from_global = ego_to_sensor(ref_cs) @ global_to_ego(ref_pose)

        sweeps = []
        prev_token = ref_sd_record["prev"]
        while prev_token and len(sweeps) < max_sweeps:
            sweep_sd = nusc.get("sample_data", prev_token)
            sweep_cs = nusc.get(
                "calibrated_sensor", sweep_sd["calibrated_sensor_token"]
            )
            sweep_pose = nusc.get("ego_pose", sweep_sd["ego_pose_token"])
            global_from_sweep = ego_to_global(sweep_pose) @ sensor_to_ego(sweep_cs)
            sweeps.append(
                {
                    "lidar_path": sweep_sd["filename"],
                    "sample_data_token": sweep_sd["token"],
                    "transform_matrix": ref_from_global @ global_from_sweep,
                    "time_lag": (ref_sd_record["timestamp"] - sweep_sd["timestamp"])
                    / 1e6,
                }
            )
            prev_token = sweep_sd["prev"]
        return sweeps

    infos = []
    for sample in nusc.sample:
        scene_name = scene_token_to_name[sample["scene_token"]]
        if scene_name not in scene_names:
            continue
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_record = nusc.get("sample_data", lidar_token)
        if not sd_record.get("is_key_frame", False):
            continue
        lidarseg_record = nusc.get("lidarseg", lidar_token)
        ref_cs = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        infos.append(
            {
                "token": sample["token"],
                "lidar_token": lidar_token,
                "lidar_path": sd_record["filename"],
                "gt_segment_path": lidarseg_record["filename"],
                "sweeps": collect_lidar_sweeps(sd_record),
                "ref_from_car": ego_to_sensor(ref_cs),
                "scene_name": scene_name,
                "timestamp": int(sample.get("timestamp", 0)),
            }
        )
    infos.sort(key=lambda x: (x["scene_name"], x["timestamp"], x["lidar_token"]))

    # Pointcept's NuScenesDataset expects nuscenes_infos_10sweeps_{split}.pkl.
    info_path = info_dir / "nuscenes_infos_10sweeps_val.pkl"
    with open(info_path, "wb") as f:
        pickle.dump(infos, f)
    print(f"[pointcept-test] wrote {len(infos)} mini-val infos to {info_path}")

    checkpoint = f"{CKPT_DIR}/{checkpoint_name}"
    run_name = "litept_fast" if fast else "litept_tta"
    save_path = Path(OUT_DIR) / f"pointcept_test_mini_val_{run_name}"
    if save_path.exists():
        import shutil

        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    config = processed_root / "semseg-litept-v1m1-0-small-mini.py"
    test_cfg_override = ""
    if fast:
        test_cfg_override = """
            test_cfg=dict(
                aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
            ),
"""
    with open(config, "w") as f:
        f.write(
            f"""
_base_ = ["{LITEPT_CONFIG}"]

save_path = "{save_path}"
weight = "{checkpoint}"
data_root = "{processed_root}"
test_epoch = 1
find_unused_parameters = False

data = dict(
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(
        data_root=data_root,
{test_cfg_override}
    ),
)
"""
        )
    print(f"[pointcept-test] wrote config to {config}")

    cmd = [
        "python",
        f"{POINTCEPT_DIR}/tools/test.py",
        "--config-file",
        str(config),
    ]

    print("Running Pointcept official tester:")
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=POINTCEPT_DIR, check=True)

    out_vol.commit()
    print(f"Saved Pointcept tester outputs to {save_path}")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={
        "/data": nuscenes_vol,
        "/checkpoints": ckpt_vol,
        "/outputs": out_vol,
    },
)
def compare_pointcept_dataset(sample_index: int = 0):
    import pickle
    import subprocess
    from pathlib import Path

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    processed_root = Path("/tmp/pointcept_nuscenes_mini_compare")
    info_dir = processed_root / "info"
    raw_link = processed_root / "raw"
    info_dir.mkdir(parents=True, exist_ok=True)
    if raw_link.exists() or raw_link.is_symlink():
        raw_link.unlink()
    raw_link.symlink_to(NUSCENES_DIR)

    nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_DIR, verbose=True)
    split_to_scenes = create_splits_scenes()
    scene_names = set(split_to_scenes["mini_val"])
    scene_token_to_name = {scene["token"]: scene["name"] for scene in nusc.scene}

    infos = []
    for sample in nusc.sample:
        scene_name = scene_token_to_name[sample["scene_token"]]
        if scene_name not in scene_names:
            continue
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_record = nusc.get("sample_data", lidar_token)
        if not sd_record.get("is_key_frame", False):
            continue
        lidarseg_record = nusc.get("lidarseg", lidar_token)
        infos.append(
            {
                "token": sample["token"],
                "lidar_token": lidar_token,
                "lidar_path": sd_record["filename"],
                "gt_segment_path": lidarseg_record["filename"],
                "sweeps": [],
                "scene_name": scene_name,
                "timestamp": int(sample.get("timestamp", 0)),
            }
        )
    infos.sort(key=lambda x: (x["scene_name"], x["timestamp"], x["lidar_token"]))
    info_path = info_dir / "nuscenes_infos_10sweeps_val.pkl"
    with open(info_path, "wb") as f:
        pickle.dump(infos, f)

    script_path = processed_root / "compare_dataset.py"
    with open(script_path, "w") as f:
        f.write(
            f'''
import numpy as np
import sys
import torch

sys.path.insert(0, "{POINTCEPT_DIR}")
sys.path.insert(0, "/root/project/ptv3_nuscenes_baseline")
sys.path.insert(0, "/root/project")

from pointcept.datasets.nuscenes import NuScenesDataset
from ptv3_nuscenes_baseline.nuscenes_lidarseg_dataset import NuScenesLidarSegDataset

def describe(name, item):
    print(f"\\n{{name}}")
    for key in ("name", "coord", "grid_coord", "feat", "segment", "inverse", "origin_segment"):
        if key not in item:
            continue
        value = item[key]
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if arr.dtype.kind in "fiu" and arr.size:
            print(
                f"{{key}}: shape={{arr.shape}} dtype={{arr.dtype}} "
                f"min={{np.nanmin(arr):.6g}} max={{np.nanmax(arr):.6g}} mean={{np.nanmean(arr):.6g}}"
            )
        else:
            print(f"{{key}}: {{value}}")
    for key in ("segment", "origin_segment"):
        if key in item:
            arr = item[key].detach().cpu().numpy() if hasattr(item[key], "detach") else np.asarray(item[key])
            vals, counts = np.unique(arr, return_counts=True)
            print(f"{{key}} hist: {{list(zip(vals.tolist(), counts.tolist()))[:40]}}")
    if "feat" in item:
        feat = item["feat"].detach().cpu().numpy() if hasattr(item["feat"], "detach") else np.asarray(item["feat"])
        print(f"feat first5: {{feat[:5].tolist()}}")
    if "coord" in item:
        coord = item["coord"].detach().cpu().numpy() if hasattr(item["coord"], "detach") else np.asarray(item["coord"])
        print(f"coord first5: {{coord[:5].tolist()}}")

sample_index = {int(sample_index)}

pc_ds = NuScenesDataset(
    split="val",
    data_root="{processed_root}",
    transform=[
        dict(type="Copy", keys_dict=dict(segment="origin_segment")),
        dict(
            type="GridSample",
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
            feat_keys=("coord", "strength"),
        ),
    ],
    test_mode=True,
    ignore_index=-1,
    loop=1,
)
our_ds = NuScenesLidarSegDataset(
    dataroot="{NUSCENES_DIR}",
    version="v1.0-mini",
    split="mini_val",
    voxel_size=0.05,
    verbose=True,
)

pc_item = pc_ds[sample_index]
our_item = our_ds[sample_index]

print(f"Pointcept dataset len: {{len(pc_ds)}}")
print(f"Our dataset len: {{len(our_ds)}}")
print(f"Pointcept sample name: {{pc_ds.data_list[sample_index].get('lidar_token', pc_ds.data_list[sample_index].get('token'))}}")
print(f"Our lidar token: {{our_item['lidar_token']}}")
describe("POINTCEPT ITEM", pc_item)
describe("OUR ITEM", our_item)

for key in ("coord", "grid_coord", "feat", "segment", "origin_segment", "inverse"):
    if key in pc_item and key in our_item:
        a = pc_item[key].detach().cpu().numpy() if hasattr(pc_item[key], "detach") else np.asarray(pc_item[key])
        b = our_item[key].detach().cpu().numpy() if hasattr(our_item[key], "detach") else np.asarray(our_item[key])
        print(f"\\ncompare {{key}}: pc={{a.shape}} ours={{b.shape}}")
        if a.shape == b.shape and a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print(f"  max_abs={{diff.max() if diff.size else 0:.6g}} mean_abs={{diff.mean() if diff.size else 0:.6g}} equal={{np.array_equal(a, b)}}")
'''
        )

    subprocess.run(["python", str(script_path)], check=True)


@app.local_entrypoint()
def main(
    mode: str = "check",
    split: str = "mini_val",
    checkpoint_name: str = LITEPT_DEFAULT_CHECKPOINT,
    fast: bool = True,
    use_flash: bool = True,
    sample_index: int = 0,
):
    if mode == "check":
        check_data.remote(split)
    elif mode == "extract_data":
        extract_data.remote()
    elif mode == "eval":
        eval_ptv3_mini.remote(split, checkpoint_name, use_flash)
    elif mode == "pointcept_test":
        pointcept_test_mini.remote(split, checkpoint_name, fast, use_flash)
    elif mode == "compare_dataset":
        compare_pointcept_dataset.remote(sample_index)
    else:
        raise ValueError(
            "mode must be 'check', 'extract_data', 'eval', 'pointcept_test', "
            "or 'compare_dataset'"
        )
