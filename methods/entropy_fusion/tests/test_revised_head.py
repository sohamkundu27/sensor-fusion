import math
import numpy as np
import torch
from models.revised_head import anchor_depth_reference
from models.predictions import metric_nms


def test_depth_reference_measures_only_present_lidar():
    depth = torch.zeros(1,1,16,16)
    mask = torch.zeros_like(depth,dtype=torch.bool)
    depth[0,0,7,7] = 10
    mask[0,0,7,7] = True
    anchors = torch.tensor([[0.,0.,16.,16.],[0.,0.,2.,2.]])
    prior = anchor_depth_reference(depth,mask,anchors)
    assert torch.allclose(prior,torch.tensor([[math.log(10),math.log(20)]]),atol=1e-5)
    assert torch.allclose(anchor_depth_reference(depth,torch.zeros_like(mask),anchors),
                          torch.full_like(prior,math.log(20)))


def test_metric_suppression_retains_depth_separated_objects():
    g2c = np.eye(4)
    g2c[:3,:3] = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    meta = dict(intrinsic=np.eye(3),global_to_camera=g2c)
    anchors = torch.tensor([[-1.,-1.,1.,1.]]*3)
    code = torch.zeros(3,10)
    code[:,2] = torch.tensor([10.,20.,10.1]).log()
    keep = metric_nms(code,anchors,torch.tensor([.9,.8,.7]),torch.zeros(3,dtype=torch.long),meta,100)
    assert keep.tolist() == [0,1]
