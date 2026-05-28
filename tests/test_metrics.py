import numpy as np

from focus_fusion.eval.metrics import SegmentationMeter


def test_segmentation_meter_perfect_predictions():
    meter = SegmentationMeter(num_classes=3, class_names=("a", "b", "c"))
    target = np.array([0, 1, 1, 2, -1])
    pred = np.array([0, 1, 1, 2, 0])
    meter.update(pred, target)
    result = meter.compute()
    assert result.miou == 1.0
    assert result.all_acc == 1.0
