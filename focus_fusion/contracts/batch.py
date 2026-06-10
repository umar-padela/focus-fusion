import torch

def fake_lidar_batch(num_points: int = 1024, num_classes: int = 16):
    coord = torch.randn(num_points, 3)
    strength = torch.rand(num_points, 1)
    return {
        "coord": coord,
        "grid_coord": torch.floor((coord - coord.min(dim=0).values) / 0.05).int(),
        "feat": torch.cat([coord, strength], dim=1),
        "offset": torch.tensor([num_points], dtype=torch.long),
        "segment": torch.randint(0, num_classes, (num_points,), dtype=torch.long),
    }
