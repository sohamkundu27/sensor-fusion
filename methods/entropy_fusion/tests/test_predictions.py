import numpy as np
import torch
from nuscenes.eval.detection.data_classes import DetectionBox
from models.predictions import decode_predictions, merge_views


def test_export_validates_schema_and_cross_view_deduplication():
    prediction = {k: torch.zeros(1, 1, n) for k,n in [('logits',11), ('boxes',4), ('boxes3d',10), ('attributes',8)]}
    prediction['logits'][0, 0, 1] = 10
    prediction['boxes3d'][0, 0, 2] = np.log(10)
    prediction['boxes3d'][0, 0, 7] = 1
    prediction['anchors'] = torch.tensor([[0., 0, 10, 10]])
    meta = dict(global_to_camera=np.eye(4), ego_to_global=np.eye(4), intrinsic=np.eye(3), sample_token='test')
    record = decode_predictions(prediction, [meta])[0][0]
    box = DetectionBox.deserialize(record)
    assert box.detection_name == 'car' and box.attribute_name.startswith('vehicle.')
    assert np.isclose(np.linalg.norm(box.rotation), 1)
    duplicate = dict(record, detection_score=.1)
    assert merge_views([duplicate, record]) == [record]
    distant = dict(record, translation=[1000, 1000, 10])
    assert len(merge_views([record, distant])) == 2
    different_class = dict(record, detection_name='pedestrian', attribute_name='pedestrian.moving')
    assert len(merge_views([record, different_class])) == 2
